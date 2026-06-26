"""Operator-instance actors: the loops that drive a :class:`~nautilus.core.operator.Operator`.

An :class:`Output` routes one upstream instance's frames to a downstream operator's instances: data
batches go through a partitioner, control frames are broadcast to *all* downstream instances.

``run_source`` and ``run_transform`` are the two actor loops. The transform loop encodes the
core streaming semantics:

* per-input watermark combination (min over non-idle inputs),
* fire windows/timers when the combined watermark advances, *then* forward the watermark,
* on EOS of all inputs, advance to ``WATERMARK_MAX`` to flush every pending window, then send EOS.

The operator's ``process``/``on_watermark`` are synchronous; this loop performs every ``await``
(backpressured send) *between* those calls, so each operator step is a race-free critical section.

Telemetry: each actor holds one :class:`~nautilus.telemetry.recorder.Recorder`, the sole writer of its
built-in metrics, with backpressure timed inside :class:`Output`. A no-op recorder skips timing
entirely.
"""

from __future__ import annotations

import traceback
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from time import perf_counter_ns

import pyarrow as pa

from nautilus.core.operator import (
    ListCollector,
    OneInputOperator,
    OperatorContext,
    SourceOperator,
)
from nautilus.core.records import (
    EOS,
    EOS_FRAME,
    WATERMARK_MAX,
    Batch,
    Frame,
    StatusActive,
    StatusIdle,
    Watermark,
)
from nautilus.core.time import WatermarkTracker
from nautilus.runtime.channel import Channel
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.partition import Partitioner
from nautilus.telemetry.model import Counter
from nautilus.telemetry.recorder import NOOP_COUNTER, NULL_RECORDER, Recorder


def _source_location(op: object) -> str:
    t = type(op)
    return f"{t.__module__}:{t.__qualname__}"


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
        recorder.incr(
            "operator.errors", 1, operator_id=op_id, op_class=op_class, exc_type=type(e).__name__
        )
        recorder.event(
            "operator.error",
            operator_id=op_id,
            op_class=op_class,
            phase=phase,
            exc_type=type(e).__name__,
            message=str(e),
            traceback=traceback.format_exc(),
            frame_kind=frame_kind,
            input_index=input_index,
            batch_rows=batch_rows,
            source_location=location,
        )
        raise


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

    async def emit(self, batch: pa.RecordBatch) -> None:
        for idx, sub in self.partitioner.route(batch, len(self.channels)):
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
            rec.set_gauge("edge.queue_depth", depth, **base)
            rec.set_gauge("edge.queue_capacity", self._capacity, **base)


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
    started = perf_counter_ns()

    async def emit(frame: Frame) -> None:
        if isinstance(frame, Batch):
            batches_out.add(1)
            rows_out.add(frame.num_rows)
            if bytes_on:
                bytes_out.add(int(frame.data.get_total_buffer_size()))
            for out in outputs:
                await out.emit(frame.data)
        else:
            await _broadcast(frame, outputs)

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
            async for frame in frames:
                await emit(frame)
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


async def run_transform(
    op: OneInputOperator,
    ctx: OperatorContext,
    mailbox: Mailbox,
    outputs: list[Output],
    *,
    recorder: Recorder = NULL_RECORDER,
) -> None:
    """Drive a one-input operator to completion, then forward EOS."""
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
    proc_hist = recorder.histogram(
        "operator.process_micros", operator_id=op_id, op_class=opcls, subtask_index=sub
    )
    batch_rows_hist = recorder.histogram(
        "operator.batch_rows", operator_id=op_id, subtask_index=sub
    )
    wm_hist = recorder.histogram(
        "operator.on_watermark_micros", operator_id=op_id, subtask_index=sub
    )
    proc_calls = recorder.counter("operator.process_calls", operator_id=op_id, subtask_index=sub)
    wm_calls = recorder.counter("operator.on_watermark_calls", operator_id=op_id, subtask_index=sub)
    step = recorder.counter("runtime.step_micros", operator_id=op_id, subtask_index=sub)
    awaits = recorder.counter("runtime.await_count", operator_id=op_id, subtask_index=sub)
    input_wait = recorder.counter("edge.input_wait_micros", operator_id=op_id)
    wm_gauge = recorder.gauge("watermark.combined_micros", operator_id=op_id, subtask_index=sub)
    advances = recorder.counter("watermark.advances", operator_id=op_id)
    wm_final = recorder.gauge("watermark.final_micros", operator_id=op_id)
    recorder.set_gauge("eos.expected", n, operator_id=op_id)

    tracker = WatermarkTracker(n)
    collector = ListCollector()
    closed = [False] * n
    started = perf_counter_ns()

    def _fire(t: int, frame_kind: str) -> None:
        w0 = perf_counter_ns()
        with _capture(recorder, op_id, opcls, loc, "on_watermark", frame_kind=frame_kind):
            op.on_watermark(t, collector)
        wm_hist.observe((perf_counter_ns() - w0) // 1000)
        wm_calls.add(1)

    async def _advance(advanced: int | None) -> None:
        """On a strict watermark advance: record it, fire due windows/timers, flush, then forward the
        watermark downstream. A no-op when the combined watermark did not move."""
        if advanced is None:
            return
        wm_gauge.set(advanced)
        advances.add(1)
        _fire(advanced, "watermark")
        await _flush(collector, outputs, rows_out, batches_out, bytes_out_arg)
        await _broadcast(Watermark(advanced), outputs)

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
                    op.process(frame.data, collector)
                dt = (perf_counter_ns() - p0) // 1000
                proc_hist.observe(dt)
                step.add(dt)
                proc_calls.add(1)
                await _flush(collector, outputs, rows_out, batches_out, bytes_out_arg)

            elif isinstance(frame, Watermark):
                await _advance(tracker.update(i, frame.t))

            elif isinstance(frame, StatusIdle):
                recorder.incr("watermark.input_idle", 1, operator_id=op_id, input_index=i)
                await _advance(tracker.set_idle(i))

            elif isinstance(frame, StatusActive):
                recorder.incr("watermark.input_active", 1, operator_id=op_id, input_index=i)
                tracker.set_active(i)

            elif isinstance(frame, EOS):
                recorder.incr("eos.received", 1, operator_id=op_id, input_index=i)
                closed[i] = True
                mailbox.close_input(i)
                advanced = tracker.close_input(i)
                if all(closed):
                    # Distinct terminal path: flush every pending window at WATERMARK_MAX, then break
                    # to forward EOS — no watermark broadcast, so not an _advance() call.
                    _fire(WATERMARK_MAX, "eos")
                    await _flush(collector, outputs, rows_out, batches_out, bytes_out_arg)
                    break
                await _advance(advanced)
    finally:
        with _capture(recorder, op_id, opcls, loc, "close"):
            op.close()
        wm_final.set(tracker.combined)
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
