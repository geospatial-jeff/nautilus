"""Operator-instance actors: the loops that drive a :class:`~nautilus.core.operator.Operator`.

An :class:`Output` routes one upstream instance's frames to a downstream operator's instances: data
batches go through a partitioner, control frames are broadcast to *all* downstream instances.

There are five actor-loop entry points. ``run_source`` drives a source; ``run_transform`` (one input) and
``run_two_input`` (a join's two inputs) are thin wrappers over the shared ``_run_operator_loop`` —
differing only in how each data batch is dispatched to the operator (``process`` vs
``process_left``/``process_right`` by the input's side). Two more let an operator ``await``, each its own
loop (the structured-concurrency shape shares nothing with the proven synchronous one): ``run_async_sink``
drives an :class:`~nautilus.core.operator.AsyncSink`, issuing each batch as one of several in-flight
``write`` tasks (a :class:`~asyncio.Semaphore` bounds them, an :class:`~asyncio.TaskGroup` owns them) so
their I/O overlaps while the actor reads on; ``run_async_transform`` drives an
:class:`~nautilus.core.operator.AsyncOneInputOperator`, overlapping its awaiting ``fetch`` across batches
while integrating and emitting each result synchronously and in input order — so it needs an ordered
reorder buffer the sink does not, a ``deque`` whose fetches wake the loop through one ``asyncio.Event``.
``_run_operator_loop`` encodes the core streaming semantics:

* dispatch each data batch to the operator (``process`` for one input, or ``process_left`` /
  ``process_right`` by the input's side for a join),
* once *every* input has sent EOS, call ``on_eos`` to flush pending per-key state, then send EOS.

The operator's ``process``/``on_eos`` are synchronous; the loop performs every ``await`` (backpressured
send) *between* those calls, so each operator step is a race-free critical section.

Telemetry: each actor holds one :class:`~nautilus.telemetry.recorder.Recorder`, the sole writer of its
built-in metrics, with backpressure timed inside :class:`Output`. A no-op recorder skips timing
entirely.
"""

from __future__ import annotations

import asyncio
import traceback
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from time import perf_counter_ns
from typing import Any

import pyarrow as pa

from nautilus.core.operator import (
    AsyncOneInputOperator,
    AsyncSink,
    Collector,
    ListCollector,
    OneInputOperator,
    OperatorContext,
    SourceOperator,
    TwoInputOperator,
)
from nautilus.core.records import EOS, EOS_FRAME, Batch, Frame
from nautilus.runtime.channel import Channel
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.partition import Partitioner
from nautilus.telemetry.model import Counter
from nautilus.telemetry.recorder import NOOP_COUNTER, NULL_RECORDER, Recorder


def _source_location(op: object) -> str:
    t = type(op)
    return f"{t.__module__}:{t.__qualname__}"


def _record_operator_error(
    recorder: Recorder,
    op_id: str,
    op_class: str,
    location: str,
    phase: str,
    exc: BaseException,
    *,
    frame_kind: str | None = None,
    input_index: int | None = None,
    batch_rows: int | None = None,
) -> None:
    """Record one operator-lifecycle exception (counter + rich event). Factored out of :func:`_capture`
    so a caller that holds an exception value it did not catch in an ``except`` block — the async sink,
    reaping a finished write task's ``.exception()`` — records it the same way."""
    recorder.incr("operator.errors", 1, operator_id=op_id, exc_type=type(exc).__name__)
    recorder.event(
        "operator.error",
        operator_id=op_id,
        op_class=op_class,
        phase=phase,
        exc_type=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback.format_exception(exc)),
        frame_kind=frame_kind,
        input_index=input_index,
        batch_rows=batch_rows,
        source_location=location,
    )


@contextmanager
def _capture(
    recorder: Recorder,
    op_id: str,
    op_class: str,
    location: str,
    phase: str,
    *,
    frame_kind: str | None = None,
    input_index: int | None = None,
    batch_rows: int | None = None,
) -> Iterator[None]:
    """Record an exception (counter + rich event), then re-raise it unchanged (fail-fast preserved)."""
    try:
        yield
    except Exception as e:
        _record_operator_error(
            recorder,
            op_id,
            op_class,
            location,
            phase,
            e,
            frame_kind=frame_kind,
            input_index=input_index,
            batch_rows=batch_rows,
        )
        raise


class _MicrosAccumulator:
    """Adds nanosecond durations into a microsecond Counter, carrying the sub-microsecond remainder so
    the running total stays accurate even when each step is under a microsecond. Truncating every add
    with ``// 1000`` would floor a sub-µs step to zero, so a high-rate stream of tiny steps would read
    as idle; carrying the remainder recovers it while keeping the counter in whole microseconds."""

    __slots__ = ("_counter", "_carry_ns")

    def __init__(self, counter: Counter) -> None:
        self._counter = counter
        self._carry_ns = 0

    def add_ns(self, ns: int) -> None:
        self._carry_ns += ns
        micros = self._carry_ns // 1000
        if micros:
            self._counter.add(micros)
            self._carry_ns -= micros * 1000


class Output:
    """Routes an upstream instance's frames to the downstream instance channels and records the
    send-side edge metrics (backpressure, frames/rows sent, queue depth)."""

    def __init__(
        self,
        channels: list[Channel],
        partitioner: Partitioner,
        *,
        recorder: Recorder = NULL_RECORDER,
        edge_src: str = "",
        edge_dst: str = "",
        capacity: int = 0,
    ) -> None:
        self.channels = channels
        self.partitioner = partitioner
        self._rec = recorder
        self._on = recorder is not NULL_RECORDER
        self._src = edge_src
        self._dst = edge_dst
        self._capacity = capacity
        # Hoisted once (the recorder warns against per-call resolution on the hot path): the time the
        # keyed shuffle spends routing is otherwise unattributed wall between process and send.
        self._route_hist = recorder.histogram(
            "partition.route_micros", operator_id=edge_src, edge_dst=edge_dst
        )
        # Per-channel queue-depth histograms, hoisted: the depth gauge gives the high-water level, this
        # gives the distribution (how often near capacity). Built only for channels that report a depth
        # (in-process), so a socket edge gets no empty series — matching the lazily-set depth gauge.
        self._depth_hists = {
            i: recorder.histogram(
                "edge.queue_depth_hist",
                operator_id=edge_src,
                edge_src=edge_src,
                edge_dst=edge_dst,
                channel_index=i,
            )
            for i, ch in enumerate(channels)
            if ch.depth() is not None
        }
        # queue_capacity is constant per channel — set it once here, not on every send. (Read back by
        # report._build_edges; only in-process channels report a depth and thus carry a capacity.)
        if self._on:
            for i in self._depth_hists:
                recorder.gauge(
                    "edge.queue_capacity",
                    operator_id=edge_src,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    channel_index=i,
                ).set(capacity)
        # Transport accounting is per-channel and only for cross-process edges: an in-process channel
        # reports None, so its index is absent here and its sends skip the bookkeeping entirely. The
        # SocketChannel reports cumulative totals; we record the per-send delta against these.
        self._transport_idx = {i for i, ch in enumerate(channels) if ch.bytes_written() is not None}
        self._prev_bytes = [0] * len(channels)
        self._prev_credit_wait = [0] * len(channels)
        self._prev_encode = [0] * len(channels)

    async def emit(self, batch: pa.RecordBatch) -> None:
        if self._on:
            t0 = perf_counter_ns()
            routed = self.partitioner.route(batch, len(self.channels))
            self._route_hist.observe((perf_counter_ns() - t0) // 1000)
        else:
            routed = self.partitioner.route(batch, len(self.channels))
        for idx, sub in routed:
            if sub.num_rows:
                await self._send(idx, Batch(sub), "data", sub.num_rows)

    async def broadcast(self, frame: Frame) -> None:
        for idx in range(len(self.channels)):
            await self._send(idx, frame, "control", 0)

    async def _send(self, idx: int, frame: Frame, frame_type: str, rows: int) -> None:
        ch = self.channels[idx]
        if not self._on:
            await ch.send(frame)
            return
        t0 = perf_counter_ns()
        await ch.send(frame)
        dt = (perf_counter_ns() - t0) // 1000
        rec = self._rec
        base: dict[str, object] = {
            "operator_id": self._src,
            "edge_src": self._src,
            "edge_dst": self._dst,
            "channel_index": idx,
        }
        rec.incr("edge.send_wait_micros", dt, **base)
        rec.incr("edge.frames_sent", 1, frame_type=frame_type, **base)
        if frame_type == "data":
            rec.incr("edge.batches_sent", 1, **base)
            rec.incr("edge.rows_sent", rows, **base)
        depth = ch.depth()
        if depth is not None:
            rec.set_gauge("edge.queue_depth", depth, **base)  # capacity is set once in __init__
            self._depth_hists[idx].observe(depth)
        if (
            idx in self._transport_idx
        ):  # cross-process edge: record wire bytes + serialize/stall deltas
            written = ch.bytes_written() or 0
            rec.incr("transport.bytes_sent", written - self._prev_bytes[idx], **base)
            self._prev_bytes[idx] = written
            waited = ch.credit_wait_micros() or 0
            rec.incr("edge.credit_wait_micros", waited - self._prev_credit_wait[idx], **base)
            self._prev_credit_wait[idx] = waited
            encoded = ch.encode_micros() or 0
            rec.incr("transport.encode_micros", encoded - self._prev_encode[idx], **base)
            self._prev_encode[idx] = encoded


async def _flush(
    collector: ListCollector,
    outputs: list[Output],
    rows_out: Counter,
    batches_out: Counter,
    bytes_out: Counter | None,
) -> None:
    for batch in collector.drain():
        batches_out.add(1)
        rows_out.add(batch.num_rows)
        if bytes_out is not None:  # FULL tier only — the buffer-size walk is the expensive part
            bytes_out.add(int(batch.get_total_buffer_size()))
        for out in outputs:
            await out.emit(batch)


async def _broadcast(frame: Frame, outputs: list[Output]) -> None:
    for out in outputs:
        await out.broadcast(frame)


async def run_source(
    source: SourceOperator,
    ctx: OperatorContext,
    outputs: list[Output],
    *,
    recorder: Recorder = NULL_RECORDER,
) -> None:
    """Drive a source: forward each generated frame downstream (data routed, control broadcast)."""
    op_id, opcls, loc = ctx.operator_id, type(source).__name__, _source_location(source)
    sub = ctx.subtask_index
    rows_out = recorder.counter("operator.rows_out", operator_id=op_id, subtask_index=sub)
    batches_out = recorder.counter("operator.batches_out", operator_id=op_id, subtask_index=sub)
    bytes_out = recorder.counter("operator.bytes_out", operator_id=op_id, subtask_index=sub)
    bytes_on = bytes_out is not NOOP_COUNTER  # FULL tier only — skip the Arrow buffer-size walk
    # A source has no process/on_eos, so without this it shows zero self-time and reads as idle
    # even when generation is the bottleneck. Time each frame's production (the generator step).
    step = _MicrosAccumulator(
        recorder.counter("runtime.step_micros", operator_id=op_id, subtask_index=sub)
    )
    started = perf_counter_ns()

    async def emit(frame: Frame) -> None:
        if isinstance(frame, Batch):
            batches_out.add(1)
            rows_out.add(frame.num_rows)
            if bytes_on:
                bytes_out.add(int(frame.data.get_total_buffer_size()))
            for out in outputs:
                await out.emit(frame.data)
        elif isinstance(frame, Frame):  # a control frame — broadcast to every downstream instance
            await _broadcast(frame, outputs)
        else:
            raise TypeError(
                f"source {op_id!r} yielded a non-Frame {type(frame).__name__}; wrap data in Batch"
            )

    with _capture(recorder, op_id, opcls, loc, "open"):
        source.open(ctx)
    recorder.event(
        "operator.lifecycle.open",
        operator_id=op_id,
        op_class=opcls,
        source_location=loc,
        num_inputs=0,
    )
    frames: AsyncIterator[Frame] | None = None
    try:
        with _capture(recorder, op_id, opcls, loc, "process"):
            frames = source.frames()
            gen0 = perf_counter_ns()
            async for frame in frames:
                # Time to produce this frame (the generator body); for a self-pacing source this
                # includes its await. The send that follows is timed separately as send/route.
                step.add_ns(perf_counter_ns() - gen0)
                await emit(frame)
                gen0 = perf_counter_ns()
    finally:
        # Finalize the frames() async generator BEFORE source.close(). Python does not call aclose()
        # on a CancelledError unwind, so a user's `async with` / try-finally inside frames() would only
        # run at GC; closing it here makes that cleanup prompt. aclose() is a no-op on an exhausted
        # generator (the normal EOS path) and throws GeneratorExit — not the in-flight CancelledError —
        # so fail-fast / unchanged re-raise is preserved. (getattr-guarded for the rare class-based
        # async iterator that is not a generator and so has no aclose.)
        if frames is not None:
            aclose = getattr(frames, "aclose", None)
            if aclose is not None:
                await aclose()
        with _capture(recorder, op_id, opcls, loc, "close"):
            source.close()
        recorder.event(
            "operator.lifecycle.close",
            operator_id=op_id,
            rows_in=0,
            rows_out=rows_out.value,
            wall_micros=(perf_counter_ns() - started) // 1000,
        )


async def _run_operator_loop(
    op: OneInputOperator | TwoInputOperator,
    ctx: OperatorContext,
    mailbox: Mailbox,
    outputs: list[Output],
    dispatch: Callable[[int, pa.RecordBatch, Collector], None],
    *,
    recorder: Recorder = NULL_RECORDER,
) -> None:
    """Drive a one- or two-input operator to completion, then forward EOS.

    The one- and two-input loops are identical but for how a data batch is handled, so they share this
    core: ``dispatch(input_index, batch, collector)`` routes the batch to the operator's handler —
    ``process`` for one input, ``process_left``/``process_right`` chosen by the input's side for two. EOS
    is forwarded only once every input — both ports — has closed, calling ``on_eos`` first to flush.
    """
    op_id, opcls, loc = ctx.operator_id, type(op).__name__, _source_location(op)
    sub, n = ctx.subtask_index, mailbox.num_inputs

    rows_in = recorder.counter("operator.rows_in", operator_id=op_id, subtask_index=sub)
    batches_in = recorder.counter("operator.batches_in", operator_id=op_id, subtask_index=sub)
    rows_out = recorder.counter("operator.rows_out", operator_id=op_id, subtask_index=sub)
    batches_out = recorder.counter("operator.batches_out", operator_id=op_id, subtask_index=sub)
    bytes_in = recorder.counter("operator.bytes_in", operator_id=op_id, subtask_index=sub)
    bytes_out = recorder.counter("operator.bytes_out", operator_id=op_id, subtask_index=sub)
    # FULL tier only: if byte accounting is disabled these resolve to the shared no-op, and we skip
    # the (expensive) Arrow buffer-size walk entirely.
    bytes_on = bytes_in is not NOOP_COUNTER
    bytes_out_arg = bytes_out if bytes_on else None
    proc_hist = recorder.histogram("operator.process_micros", operator_id=op_id, subtask_index=sub)
    batch_rows_hist = recorder.histogram(
        "operator.batch_rows", operator_id=op_id, subtask_index=sub
    )
    eos_hist = recorder.histogram("operator.on_eos_micros", operator_id=op_id, subtask_index=sub)
    proc_calls = recorder.counter("operator.process_calls", operator_id=op_id, subtask_index=sub)
    eos_calls = recorder.counter("operator.on_eos_calls", operator_id=op_id, subtask_index=sub)
    step = _MicrosAccumulator(
        recorder.counter("runtime.step_micros", operator_id=op_id, subtask_index=sub)
    )
    awaits = recorder.counter("runtime.await_count", operator_id=op_id, subtask_index=sub)
    input_wait = recorder.counter("edge.input_wait_micros", operator_id=op_id)
    recorder.set_gauge("eos.expected", n, operator_id=op_id)

    collector = ListCollector()
    closed = [False] * n
    started = perf_counter_ns()
    state_on = recorder is not NULL_RECORDER  # gate the state-size walk; OFF/no-op runs skip it

    def _sample_state() -> None:
        # Sampled once at EOS — the high-water point, before on_eos flushes buffered state. sizes() is
        # O(state-names), not a store walk.
        for (sop_id, name), (entries, keys) in ctx.state_backend.sizes().items():
            recorder.set_gauge("state.entries", entries, operator_id=sop_id, state_name=name)
            recorder.set_gauge("state.keys", keys, operator_id=sop_id, state_name=name)

    def _flush_state() -> None:
        """At EOS on every input: sample state high-water, then run ``on_eos`` to emit final results."""
        if state_on:
            _sample_state()
        w0 = perf_counter_ns()
        with _capture(recorder, op_id, opcls, loc, "on_eos"):
            op.on_eos(collector)
        dt_ns = perf_counter_ns() - w0
        eos_hist.observe(dt_ns // 1000)
        eos_calls.add(1)
        step.add_ns(dt_ns)  # on_eos is a synchronous critical section too — see runtime.step_micros

    with _capture(recorder, op_id, opcls, loc, "open"):
        op.open(ctx)
    recorder.event(
        "operator.lifecycle.open",
        operator_id=op_id,
        op_class=opcls,
        source_location=loc,
        num_inputs=n,
    )

    try:
        while not mailbox.exhausted:
            t0 = perf_counter_ns()
            i, frame = await mailbox.get()
            input_wait.add((perf_counter_ns() - t0) // 1000)
            awaits.add(1)

            if isinstance(frame, Batch):
                rows = frame.num_rows
                batches_in.add(1)
                rows_in.add(rows)
                batch_rows_hist.observe(rows)
                if bytes_on:
                    bytes_in.add(int(frame.data.get_total_buffer_size()))
                p0 = perf_counter_ns()
                with _capture(
                    recorder,
                    op_id,
                    opcls,
                    loc,
                    "process",
                    frame_kind="batch",
                    input_index=i,
                    batch_rows=rows,
                ):
                    dispatch(i, frame.data, collector)
                dt_ns = perf_counter_ns() - p0
                proc_hist.observe(dt_ns // 1000)
                step.add_ns(dt_ns)
                proc_calls.add(1)
                await _flush(collector, outputs, rows_out, batches_out, bytes_out_arg)

            elif isinstance(frame, EOS):
                recorder.incr("eos.received", 1, operator_id=op_id, input_index=i)
                closed[i] = True
                mailbox.close_input(i)
                if all(closed):
                    # Every input closed: flush buffered state via on_eos, then break to forward EOS.
                    _flush_state()
                    await _flush(collector, outputs, rows_out, batches_out, bytes_out_arg)
                    break

            else:  # an unknown/unhandled frame must fail loudly, never silently vanish
                raise TypeError(
                    f"operator {op_id!r} received an unhandled frame on input {i}: "
                    f"{type(frame).__name__}"
                )
    finally:
        mailbox.close()  # cancel any recvs still armed if the actor unwound mid-fan-in (fail-fast)
        with _capture(recorder, op_id, opcls, loc, "close"):
            op.close()
        # Inbound Arrow IPC decode happens in the channels' background read loops; total it once here.
        # Zero unless an input crossed a socket, so single-process runs record nothing.
        decoded = mailbox.decode_micros()
        if decoded:
            recorder.incr("transport.decode_micros", decoded, operator_id=op_id)
        recorder.event(
            "operator.lifecycle.close",
            operator_id=op_id,
            rows_in=rows_in.value,
            rows_out=rows_out.value,
            wall_micros=(perf_counter_ns() - started) // 1000,
        )

    recorder.event(
        "eos.forwarded", operator_id=op_id, wall_micros=(perf_counter_ns() - started) // 1000
    )
    await _broadcast(EOS_FRAME, outputs)


async def run_transform(
    op: OneInputOperator,
    ctx: OperatorContext,
    mailbox: Mailbox,
    outputs: list[Output],
    *,
    recorder: Recorder = NULL_RECORDER,
) -> None:
    """Drive a one-input operator to completion, then forward EOS — every batch goes to ``process``."""
    await _run_operator_loop(
        op, ctx, mailbox, outputs, lambda _i, batch, out: op.process(batch, out), recorder=recorder
    )


async def run_two_input(
    op: TwoInputOperator,
    ctx: OperatorContext,
    mailbox: Mailbox,
    outputs: list[Output],
    *,
    left_input_count: int,
    recorder: Recorder = NULL_RECORDER,
) -> None:
    """Drive a two-input operator (a join) to completion, then forward EOS.

    The mailbox concatenates the left input's channels before the right's, so input indices
    ``[0, left_input_count)`` are the left side (``process_left``) and the rest are the right
    (``process_right``). The shared loop forwards EOS only after every channel of both sides has
    closed, calling ``on_eos`` first so the join can flush any buffered state."""

    def dispatch(i: int, batch: pa.RecordBatch, out: Collector) -> None:
        if i < left_input_count:
            op.process_left(batch, out)
        else:
            op.process_right(batch, out)

    await _run_operator_loop(op, ctx, mailbox, outputs, dispatch, recorder=recorder)


async def run_async_sink(
    sink: AsyncSink,
    ctx: OperatorContext,
    mailbox: Mailbox,
    *,
    recorder: Recorder = NULL_RECORDER,
) -> None:
    """Drive an async sink to completion: write each inbound batch to its external store, then close. The
    sink is the graph's terminal — no outputs, so it emits and forwards nothing.

    Two asyncio primitives carry the contract. A :class:`~asyncio.Semaphore` bounds the writes in flight;
    at the bound the loop stops reading, so the upstream channel fills and stalls the producer (the
    backpressure). An :class:`~asyncio.TaskGroup` owns the write tasks, which gives the rest for free:
    leaving the group awaits every outstanding write (the end-of-stream drain), and the first write to
    raise — or to exceed ``timeout_micros`` — cancels and awaits its siblings and propagates as an
    ``ExceptionGroup`` (fail-fast with prompt cleanup).

    Each write task records its own ``async.*`` and error metrics. That is still single-writer in the
    sense the recorder means: every task shares this actor's one event loop, and a counter add never
    ``await``s, so two tasks' updates can never interleave."""
    op_id, opcls, loc = ctx.operator_id, type(sink).__name__, _source_location(sink)
    sub, n = ctx.subtask_index, mailbox.num_inputs

    rows_in = recorder.counter("operator.rows_in", operator_id=op_id, subtask_index=sub)
    batches_in = recorder.counter("operator.batches_in", operator_id=op_id, subtask_index=sub)
    input_wait = recorder.counter("edge.input_wait_micros", operator_id=op_id)
    requests = recorder.counter("async.requests", operator_id=op_id, subtask_index=sub)
    req_micros = recorder.counter("async.request_micros", operator_id=op_id, subtask_index=sub)
    timeouts = recorder.counter("async.timeouts", operator_id=op_id, subtask_index=sub)
    in_flight_gauge = recorder.gauge("async.in_flight", operator_id=op_id, subtask_index=sub)
    recorder.set_gauge("eos.expected", n, operator_id=op_id)

    cap = sink.max_in_flight()
    if cap < 1:
        raise ValueError(f"async sink {op_id!r} max_in_flight() returned {cap}; it must be >= 1")
    recorder.set_gauge("async.capacity", cap, operator_id=op_id, subtask_index=sub)
    timeout_us = sink.timeout_micros()
    timeout_s = None if timeout_us is None else timeout_us / 1_000_000
    sem = asyncio.Semaphore(cap)
    in_flight = 0
    started = perf_counter_ns()

    async def write_one(batch: pa.RecordBatch) -> None:
        nonlocal in_flight
        try:
            t0 = perf_counter_ns()
            with _capture(recorder, op_id, opcls, loc, "write"):
                try:
                    if timeout_s is None:
                        await sink.write(batch)
                    else:
                        await asyncio.wait_for(sink.write(batch), timeout_s)
                except TimeoutError:
                    timeouts.add(1)
                    raise
            requests.add(1)
            req_micros.add((perf_counter_ns() - t0) // 1000)
        finally:
            in_flight -= 1
            sem.release()

    with _capture(recorder, op_id, opcls, loc, "open"):
        sink.open(ctx)
    recorder.event(
        "operator.lifecycle.open",
        operator_id=op_id,
        op_class=opcls,
        source_location=loc,
        num_inputs=n,
    )

    try:
        async with asyncio.TaskGroup() as tg:
            while not mailbox.exhausted:
                t0 = perf_counter_ns()
                i, frame = await mailbox.get()
                input_wait.add((perf_counter_ns() - t0) // 1000)
                if isinstance(frame, Batch):
                    batches_in.add(1)
                    rows_in.add(frame.num_rows)
                    # acquire after the read, not before: a pre-read acquire would idle a write slot
                    # whenever the loop is blocked waiting for input.
                    await sem.acquire()
                    in_flight += 1
                    in_flight_gauge.set(
                        in_flight
                    )  # gauge keeps the max — this is the peak in flight
                    tg.create_task(write_one(frame.data))
                elif isinstance(frame, EOS):
                    recorder.incr("eos.received", 1, operator_id=op_id, input_index=i)
                    mailbox.close_input(i)
                elif frame.is_control:
                    pass  # reserved control frames (e.g. Barrier) carry no sink action
                else:
                    raise TypeError(
                        f"async sink {op_id!r} received an unhandled frame on input {i}: "
                        f"{type(frame).__name__}"
                    )
            # leaving the group awaits every outstanding write — the end-of-stream drain
    finally:
        mailbox.close()
        with _capture(recorder, op_id, opcls, loc, "close"):
            await sink.close()
        decoded = mailbox.decode_micros()
        if decoded:
            recorder.incr("transport.decode_micros", decoded, operator_id=op_id)
        recorder.event(
            "operator.lifecycle.close",
            operator_id=op_id,
            rows_in=rows_in.value,
            rows_out=0,
            wall_micros=(perf_counter_ns() - started) // 1000,
        )


class _Data:
    """A data slot in the async transform's ordered reorder buffer: one batch and the in-flight ``fetch``
    task computing its result. Input order is the deque position, so no sequence number is needed.
    """

    __slots__ = ("batch", "task")

    def __init__(self, batch: pa.RecordBatch, task: asyncio.Future[tuple[object, int]]) -> None:
        self.batch = batch
        self.task = task


async def run_async_transform(
    op: AsyncOneInputOperator,
    ctx: OperatorContext,
    mailbox: Mailbox,
    outputs: list[Output],
    *,
    recorder: Recorder = NULL_RECORDER,
) -> None:
    """Drive an async one-input transform (the fetch/integrate contract, ``DESIGN.md`` mechanism 9) to
    completion, then forward EOS.

    Realizing that contract means reordering out-of-order fetch completions back into input order — which
    the sink's TaskGroup cannot, so this is a ``deque`` of slots the driver reaps as fetches complete:

    * The ``deque`` holds DATA slots (a batch + its in-flight fetch task, in input order). The head is the
      reorder point: ``integrate``/emit drains a DATA head once its fetch is done — yet later fetches keep
      running behind it. A fetch's wall duration is read off its task when the actor reaps it, feeding
      ``async.request_micros``.
    * ``max_in_flight`` bounds the buffer: ``buffered`` counts DATA slots and is decremented on pop, not on
      completion, so a slow head cannot let the buffer grow without limit; a full buffer stalls reads — the
      backpressure to upstream. ``awaiting`` separately counts fetches still in flight (the
      ``async.in_flight`` gauge), ``<= buffered`` once completed fetches queue behind a slow head.
    * Each fetch and the armed read carry a done-callback that sets one ``asyncio.Event``, so the loop
      blocks once per completion instead of re-registering the whole in-flight set every turn. The fetch
      callback also records a failure the instant it happens — regardless of reorder position — so the loop
      fails fast on any fetch (not only the head) without ever waiting on a blocked head. On failure or
      teardown the ``finally`` cancels and awaits every pending task, running each ``fetch``'s own
      ``try/finally`` cleanup promptly.
    * Terminal: only once the mailbox is exhausted *and* nothing is buffered does it call ``on_eos`` to
      flush buffered state, then forward EOS — strictly after the last batch is emitted, so EOS never
      overtakes in-flight data.

    The driver installs the :class:`~nautilus.core.operator.OperatorContext` guard that keeps a concurrent
    ``fetch`` out of keyed state, opening it only for the synchronous ``open``/``integrate``/``on_eos``/
    ``close`` calls; ``runtime.step_micros`` here is those calls' self-time only, never the awaited I/O.
    """
    op_id, opcls, loc = ctx.operator_id, type(op).__name__, _source_location(op)
    sub, n = ctx.subtask_index, mailbox.num_inputs

    rows_in = recorder.counter("operator.rows_in", operator_id=op_id, subtask_index=sub)
    batches_in = recorder.counter("operator.batches_in", operator_id=op_id, subtask_index=sub)
    rows_out = recorder.counter("operator.rows_out", operator_id=op_id, subtask_index=sub)
    batches_out = recorder.counter("operator.batches_out", operator_id=op_id, subtask_index=sub)
    bytes_in = recorder.counter("operator.bytes_in", operator_id=op_id, subtask_index=sub)
    bytes_out = recorder.counter("operator.bytes_out", operator_id=op_id, subtask_index=sub)
    bytes_on = bytes_in is not NOOP_COUNTER  # FULL tier only — skip the Arrow buffer-size walk
    bytes_out_arg = bytes_out if bytes_on else None
    proc_hist = recorder.histogram("operator.process_micros", operator_id=op_id, subtask_index=sub)
    batch_rows_hist = recorder.histogram(
        "operator.batch_rows", operator_id=op_id, subtask_index=sub
    )
    eos_hist = recorder.histogram("operator.on_eos_micros", operator_id=op_id, subtask_index=sub)
    proc_calls = recorder.counter("operator.process_calls", operator_id=op_id, subtask_index=sub)
    eos_calls = recorder.counter("operator.on_eos_calls", operator_id=op_id, subtask_index=sub)
    step = _MicrosAccumulator(
        recorder.counter("runtime.step_micros", operator_id=op_id, subtask_index=sub)
    )
    awaits = recorder.counter("runtime.await_count", operator_id=op_id, subtask_index=sub)
    input_wait = recorder.counter("edge.input_wait_micros", operator_id=op_id)
    requests = recorder.counter("async.requests", operator_id=op_id, subtask_index=sub)
    req_micros = recorder.counter("async.request_micros", operator_id=op_id, subtask_index=sub)
    timeouts = recorder.counter("async.timeouts", operator_id=op_id, subtask_index=sub)
    in_flight_gauge = recorder.gauge("async.in_flight", operator_id=op_id, subtask_index=sub)
    recorder.set_gauge("eos.expected", n, operator_id=op_id)

    if not op.ordered():
        raise NotImplementedError(
            f"async transform {op_id!r} requested ordered()=False; only ordered (input-order) emission "
            "is implemented — unordered is a planned addition"
        )
    cap = op.max_in_flight()
    if cap < 1:
        raise ValueError(
            f"async transform {op_id!r} max_in_flight() returned {cap}; it must be >= 1"
        )
    recorder.set_gauge("async.capacity", cap, operator_id=op_id, subtask_index=sub)
    timeout_us = op.timeout_micros()
    timeout_s = None if timeout_us is None else timeout_us / 1_000_000

    collector = ListCollector()
    state_on = recorder is not NULL_RECORDER  # gate the state-size walk; OFF/no-op runs skip it
    backend = (
        ctx._install_async_guard()
    )  # returns the raw backend for engine sampling (bypasses the guard)
    started = perf_counter_ns()

    pending: deque[_Data] = deque()
    buffered = (
        0  # DATA slots awaiting drain, in input order; max_in_flight caps it — the backpressure
    )
    awaiting = (
        0  # fetches launched but not yet completed — the true count for the async.in_flight gauge
    )
    failed: list[asyncio.Future[Any]] = []  # fetches that raised, appended by their done-callback
    progress = (
        asyncio.Event()
    )  # a fetch/read completion sets it, so the loop blocks once per event, not per task
    get_task: asyncio.Future[tuple[int, Frame]] | None = None
    armed_ns = 0
    read_done_ns = 0

    def _sample_state() -> None:
        for (sop_id, name), (entries, keys) in backend.sizes().items():
            recorder.set_gauge("state.entries", entries, operator_id=sop_id, state_name=name)
            recorder.set_gauge("state.keys", keys, operator_id=sop_id, state_name=name)

    def _flush_state() -> None:
        """At EOS: sample state high-water, then run ``on_eos`` to emit final results."""
        if state_on:
            _sample_state()
        w0 = perf_counter_ns()
        with (
            _capture(recorder, op_id, opcls, loc, "on_eos"),
            ctx._state_section(),
        ):
            op.on_eos(ctx, collector)
        dt_ns = perf_counter_ns() - w0
        eos_hist.observe(dt_ns // 1000)
        eos_calls.add(1)
        step.add_ns(dt_ns)  # on_eos is a synchronous critical section too

    async def _timed(batch: pa.RecordBatch) -> tuple[object, int]:
        """Run one ``fetch`` under its per-request timeout; return its result and the wall time it took."""
        t0 = perf_counter_ns()
        if timeout_s is None:
            result = await op.fetch(batch)
        else:
            result = await asyncio.wait_for(op.fetch(batch), timeout_s)
        return result, perf_counter_ns() - t0

    def _on_fetch_done(task: asyncio.Future[tuple[object, int]]) -> None:
        """Done-callback on every fetch task: keep the true in-flight count, and record a failure the
        instant it happens. Because the callback fires regardless of the task's reorder-buffer position, a
        failure behind a slow head — or on any sibling — is seen at once, so the loop below can fail fast
        without ever waiting on the blocking head (and a timeout/error is recorded even if it is later
        surfaced by a drain rather than the loop's abort)."""
        nonlocal awaiting
        awaiting -= 1
        if not task.cancelled() and (exc := task.exception()) is not None:
            if isinstance(exc, TimeoutError):
                timeouts.add(1)
            _record_operator_error(recorder, op_id, opcls, loc, "fetch", exc)
            failed.append(task)
        progress.set()

    def _on_read_done(_task: asyncio.Future[tuple[int, Frame]]) -> None:
        """Done-callback on the armed read: stamp when it actually completed, so edge.input_wait_micros
        counts suspension in ``mailbox.get`` and not the arm-to-processing span the loop interleaves with
        fetch reaps and integrate."""
        nonlocal read_done_ns
        read_done_ns = perf_counter_ns()
        progress.set()

    def _classify(i: int, frame: Frame) -> None:
        """Place one read frame into the reorder buffer: launch a fetch for a data batch, or record EOS on
        its input."""
        nonlocal buffered, awaiting
        if isinstance(frame, Batch):
            rows = frame.num_rows
            batches_in.add(1)
            rows_in.add(rows)
            batch_rows_hist.observe(rows)
            if bytes_on:
                bytes_in.add(int(frame.data.get_total_buffer_size()))
            task = asyncio.ensure_future(_timed(frame.data))
            task.add_done_callback(_on_fetch_done)
            pending.append(_Data(frame.data, task))
            buffered += 1
            awaiting += 1
            in_flight_gauge.set(
                awaiting
            )  # gauge keeps the max — the peak fetches awaiting I/O at once
        elif isinstance(frame, EOS):
            recorder.incr("eos.received", 1, operator_id=op_id, input_index=i)
            mailbox.close_input(i)
        else:  # an unknown/unhandled frame must fail loudly, never silently vanish
            raise TypeError(
                f"async transform {op_id!r} received an unhandled frame on input {i}: "
                f"{type(frame).__name__}"
            )

    async def _drain_head() -> None:
        """Emit ready data in input order, stopping at the first DATA slot whose fetch is still running (it
        gates everything behind it). Bails on a failed fetch so the loop's fail-fast abort — which also
        records it — surfaces the error, not a reap here."""
        nonlocal buffered
        while pending and not failed:
            head = pending[0]
            if not head.task.done():
                break
            pending.popleft()
            buffered -= (
                1  # on pop, not completion — keeps the buffer bounded (see the buffered note above)
            )
            result, dur_ns = head.task.result()
            requests.add(1)
            req_micros.add(dur_ns // 1000)
            p0 = perf_counter_ns()
            with (
                _capture(
                    recorder,
                    op_id,
                    opcls,
                    loc,
                    "integrate",
                    frame_kind="batch",
                    batch_rows=head.batch.num_rows,
                ),
                ctx._state_section(),
            ):
                op.integrate(head.batch, result, ctx, collector)
            dt_ns = perf_counter_ns() - p0
            proc_hist.observe(dt_ns // 1000)
            proc_calls.add(1)
            step.add_ns(dt_ns)
            await _flush(collector, outputs, rows_out, batches_out, bytes_out_arg)

    def _ready() -> bool:
        """Whether the loop can make progress without blocking: a fetch failed, the armed read completed,
        or the head fetch has finished."""
        if failed or (get_task is not None and get_task.done()):
            return True
        return bool(pending) and pending[0].task.done()

    with _capture(recorder, op_id, opcls, loc, "open"), ctx._state_section():
        op.open(ctx)
    recorder.event(
        "operator.lifecycle.open",
        operator_id=op_id,
        op_class=opcls,
        source_location=loc,
        num_inputs=n,
    )

    try:
        while True:
            if (
                failed
            ):  # a fetch raised (recorded in its callback); abort — the finally cancels the rest
                exc = failed[0].exception()
                assert exc is not None
                raise exc

            if get_task is None and buffered < cap and not mailbox.exhausted:
                get_task = asyncio.ensure_future(mailbox.get())
                get_task.add_done_callback(_on_read_done)
                armed_ns = perf_counter_ns()

            # Terminal: the mailbox is exhausted (else a read would be armed above) and nothing is buffered
            # — every fetch reaped and emitted by the drain below. Call on_eos to flush buffered state, then
            # break to forward EOS strictly after the last batch is emitted, so EOS never overtakes data.
            if get_task is None and buffered == 0:
                _flush_state()
                await _flush(collector, outputs, rows_out, batches_out, bytes_out_arg)
                break

            # Block until the next fetch or read completion — one wakeup per event via `progress`, set by
            # their done-callbacks, instead of re-registering the whole in-flight set each turn. Clear then
            # re-check so a completion racing the clear is never lost.
            if not _ready():
                progress.clear()
                if not _ready():
                    await progress.wait()
            if failed:
                continue  # surfaced by the abort at the loop top; don't launch more work

            if get_task is not None and get_task.done():
                i, frame = get_task.result()
                input_wait.add((read_done_ns - armed_ns) // 1000)
                awaits.add(1)
                get_task = None
                _classify(i, frame)

            await _drain_head()
    finally:
        # Cancel every still-pending fetch (and the armed read) BEFORE awaiting them — gathering a
        # blocked fetch without cancelling it would hang teardown forever.
        leftover: list[asyncio.Future[Any]] = [s.task for s in pending]
        if get_task is not None:
            leftover.append(get_task)
        for fut in leftover:
            fut.cancel()
        if leftover:
            # Await — not a bare cancel — so each fetch's own try/finally (release the client, abort the
            # in-flight request) runs promptly. CancelledError from the cancels is suppressed.
            await asyncio.gather(*leftover, return_exceptions=True)
        mailbox.close()
        with _capture(recorder, op_id, opcls, loc, "close"), ctx._state_section():
            await op.close()
        decoded = mailbox.decode_micros()
        if decoded:
            recorder.incr("transport.decode_micros", decoded, operator_id=op_id)
        recorder.event(
            "operator.lifecycle.close",
            operator_id=op_id,
            rows_in=rows_in.value,
            rows_out=rows_out.value,
            wall_micros=(perf_counter_ns() - started) // 1000,
        )

    recorder.event(
        "eos.forwarded", operator_id=op_id, wall_micros=(perf_counter_ns() - started) // 1000
    )
    await _broadcast(EOS_FRAME, outputs)
