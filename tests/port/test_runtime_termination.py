"""Termination, EOS ordering, and backpressure invariants a different executor must preserve.

These pin the *observable* rules a Python->Rust rewrite has to reproduce, at the layer each rule lives:

* the mailbox's per-channel FIFO and the actor's ``close_input``-after-EOS suppression (items a, b),
* the in-process pipeline's freedom from deadlock at capacity 1 (item c),
* the synchronous operator loop's EOS-after-``on_eos``-flush ordering (items d, e),
* the source loop's non-Frame guard (item f),
* the collecting sink's control-frame skip (item g).

Every golden here was produced by running the real runtime — the frame sequences and the rotation
order are what the current code emits, not an assumption about what it should.
"""

from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

from nautilus.core.operator import Collector, OneInputOperator, OperatorContext, SourceOperator
from nautilus.core.records import EOS, EOS_FRAME, Barrier, Batch, Frame
from nautilus.runtime.actor import Output, run_source, run_transform
from nautilus.runtime.channel import InProcChannel
from nautilus.runtime.execute import _collect_sink
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.partition import Forward
from nautilus.telemetry.recorder import NULL_RECORDER

# --- helpers -----------------------------------------------------------------------------------


def _b(v: int) -> Batch:
    """A one-row data batch carrying ``v`` in column ``v`` — a distinguishable marker per frame."""
    return Batch(pa.record_batch({"v": [v]}))


def _val(frame: Frame) -> int | None:
    """The single ``v`` cell of a data batch, or ``None`` for a control frame — so an output frame
    sequence reads as a list of ints and ``None``s that pins both order and payload."""
    return frame.data.column("v")[0].as_py() if isinstance(frame, Batch) else None


async def _drive_transform(op: OneInputOperator, frames: list[Frame]) -> list[Frame]:
    """Drive one one-input operator over ``frames`` through the real ``run_transform`` loop and return
    the exact downstream frame sequence (data batches then the terminal EOS). Feeds and collects on
    generous-capacity in-process channels so the run itself never stalls — the collector stops at EOS.
    This is the same source->transform->collect harness ``tests/test_async_transform.py`` uses."""
    in_chan, out_chan = InProcChannel(64), InProcChannel(64)
    captured: list[Frame] = []

    async def feed() -> None:
        for f in frames:
            await in_chan.send(f)

    async def collect() -> None:
        while True:
            fr = await out_chan.recv()
            captured.append(fr)
            if isinstance(fr, EOS):
                return

    async with asyncio.TaskGroup() as tg:
        tg.create_task(
            run_transform(
                op,
                OperatorContext("op0"),
                Mailbox([in_chan]),
                [Output([out_chan], Forward())],
            )
        )
        tg.create_task(feed())
        tg.create_task(collect())
    return captured


class _EmitOnEos(OneInputOperator):
    """Forwards each batch, then emits one final marker batch from ``on_eos`` — used to pin that the
    engine forwards EOS strictly after an ``on_eos`` emission, and that ``on_eos`` fires even when the
    only frame seen is EOS."""

    def __init__(self, sentinel: int) -> None:
        self._sentinel = sentinel

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        out.emit(batch)

    def on_eos(self, out: Collector) -> None:
        out.emit(pa.record_batch({"v": [self._sentinel]}))


class _NonFrameSource(SourceOperator):
    """A source whose generator yields a bare ``int`` (not a :class:`Frame`) — the misuse the source
    loop must reject loudly rather than route as data."""

    async def frames(self):  # type: ignore[override]
        yield 12345


# --- (a) mailbox: a post-EOS batch on a channel is never returned once its EOS is consumed ---------


async def test_mailbox_post_eos_batch_is_never_returned() -> None:
    # Channel 0 carries [Batch, EOS, Batch]; the trailing batch sits *after* that channel's EOS. Driving
    # the mailbox the way the actor loop does — close_input(i) the instant an EOS is consumed, never
    # re-arm — the post-EOS batch is unreachable: the run terminates having seen only the pre-EOS batch
    # and the two EOS frames.
    a, b = InProcChannel(64), InProcChannel(64)
    mb = Mailbox([a, b])
    await a.send(_b(1))
    await a.send(EOS_FRAME)
    await a.send(_b(999))  # after channel 0's own EOS — must never surface

    seen: list[tuple[int, int | None]] = []
    while not mb.exhausted:
        i, frame = await mb.get()
        seen.append((i, _val(frame)))
        if isinstance(frame, EOS):
            mb.close_input(i)  # exactly what the actor does on EOS — stop receiving on that input
            if i == 0:
                await b.send(EOS_FRAME)  # release channel 1 so the mailbox can exhaust

    assert seen == [(0, 1), (0, None), (1, None)]  # Batch(1), EOS(ch0), EOS(ch1) — never 999
    assert mb.exhausted


async def test_mailbox_without_close_input_would_surface_post_eos_batch() -> None:
    # The complement of the above, pinning *where* the invariant is enforced: the mailbox's own contract
    # is per-channel FIFO, so without close_input(0) after EOS it re-arms channel 0 and *does* return the
    # post-EOS batch (999). Suppressing it is the actor's close_input, not a mailbox drop — a rewrite that
    # moves the check must keep this split.
    a, b = InProcChannel(64), InProcChannel(64)
    mb = Mailbox([a, b])
    await a.send(_b(1))
    await a.send(EOS_FRAME)
    await a.send(_b(999))
    await b.send(EOS_FRAME)

    got = [await mb.get() for _ in range(4)]
    assert [(i, _val(f)) for i, f in got] == [(0, 1), (1, None), (0, None), (0, 999)]


# --- (b) fan-in rotation over three always-ready channels yields 0,1,2,0,1,2 -----------------------


async def test_fan_in_rotation_over_three_ready_channels() -> None:
    # All three inputs are always ready, so the pick is a pure fairness tie-break. The rotating start
    # begins at index 0 and advances one past each choice, giving the exact cycle 0,1,2,0,1,2 — pinned so
    # a rewrite reproduces both the start (0) and the rotation direction, not merely "no starvation".
    a, b, c = InProcChannel(64), InProcChannel(64), InProcChannel(64)
    mb = Mailbox([a, b, c])
    for _ in range(6):
        await a.send(Barrier(0))
        await b.send(Barrier(1))
        await c.send(Barrier(2))

    seq = [(await mb.get())[0] for _ in range(6)]
    assert seq == [0, 1, 2, 0, 1, 2]


async def test_fan_in_rotation_start_is_independent_of_fill_order() -> None:
    # The start is state (``_next_start`` = 0), not "whichever channel was ready first". Filling the
    # channels in reverse (c, b, a) still yields 0,1,2,0,1,2 — the rotation is deterministic regardless
    # of arrival order, which is what makes the digest reproducible.
    a, b, c = InProcChannel(64), InProcChannel(64), InProcChannel(64)
    mb = Mailbox([a, b, c])
    for _ in range(6):
        await c.send(Barrier(2))
        await b.send(Barrier(1))
        await a.send(Barrier(0))

    seq = [(await mb.get())[0] for _ in range(6)]
    assert seq == [0, 1, 2, 0, 1, 2]


# --- (c) in-process source->map->sink at capacity 1 completes without deadlock ---------------------


async def test_in_process_pipeline_at_capacity_one_conserves_all_rows() -> None:
    # capacity=1 is the tightest backpressure: every channel holds a single frame, so producer and
    # consumer must interleave step-for-step or the run deadlocks. ~100 one-row batches through a
    # keyless map must all reach the sink inside the timeout, none lost. Imported here so the module's
    # top stays runtime-only (this is the only test needing the driver).
    from nautilus.core.time import TestClock
    from nautilus.driver.local import run_local_chain
    from nautilus.operators import InMemorySource, MapBatch

    frames: list[Frame] = [_b(i) for i in range(100)] + [EOS_FRAME]
    result = await asyncio.wait_for(
        run_local_chain(
            InMemorySource(list(frames)),
            [MapBatch(lambda batch: batch)],
            capacity=1,
            clock=TestClock(),
        ),
        timeout=20,
    )
    assert sum(rb.num_rows for rb in result) == 100


# --- (d) the sync loop forwards EOS strictly after an on_eos emission ------------------------------


async def test_sync_loop_forwards_eos_after_on_eos_batch() -> None:
    # The operator emits its input batch (7), then a final batch (-1) from on_eos. The loop must flush
    # that on_eos emission downstream *before* forwarding EOS, so the terminal EOS is the last frame —
    # never overtaking the batch produced during flush.
    out = await _drive_transform(_EmitOnEos(sentinel=-1), [_b(7), EOS_FRAME])

    assert [_val(f) for f in out] == [7, -1, None]
    assert isinstance(out[-1], EOS)  # EOS is strictly last, after the on_eos batch


# --- (e) on_eos fires and flushes on an EOS-only input --------------------------------------------


async def test_on_eos_flushes_on_eos_only_input() -> None:
    # An input that carries only EOS (no data) must still trigger on_eos: the operator emits a sentinel
    # (-42) there, and it must be delivered before the terminal EOS. This is the empty-partition case a
    # skewed shuffle produces on the instances that receive no rows.
    out = await _drive_transform(_EmitOnEos(sentinel=-42), [EOS_FRAME])

    assert [_val(f) for f in out] == [-42, None]
    assert isinstance(out[-1], EOS)


# --- (f) the source loop rejects a non-Frame yield ------------------------------------------------


async def test_source_loop_rejects_non_frame_yield() -> None:
    # A source generator that yields a bare value instead of a Batch is a contract violation the loop
    # must surface loudly, with a message that tells the author how to fix it (wrap the data in a Batch).
    out_chan = InProcChannel(64)
    with pytest.raises(TypeError, match="wrap data in Batch"):
        await run_source(
            _NonFrameSource(),
            OperatorContext("src0"),
            [Output([out_chan], Forward())],
        )


# --- (g) the collecting sink skips non-data control frames ----------------------------------------


async def test_collect_sink_ignores_non_data_control_frames() -> None:
    # The synthesized collecting sink appends only data batches. A reserved control frame (Barrier)
    # interleaved with data must be skipped, not collected and not fatal: [b1, Barrier, b2, EOS] yields
    # exactly [b1, b2].
    ch = InProcChannel(64)
    mb = Mailbox([ch])
    frames: list[Frame] = [_b(1), Barrier(0), _b(2), EOS_FRAME]
    collected: list[pa.RecordBatch] = []

    async def feed() -> None:
        for f in frames:
            await ch.send(f)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_collect_sink(mb, collected, NULL_RECORDER))
        tg.create_task(feed())

    assert [rb.column("v")[0].as_py() for rb in collected] == [1, 2]
