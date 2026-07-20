"""Executor-independent async semantics — the contract a tokio port of ``run_async_transform`` must keep.

A future Rust port will complete futures in a different order than asyncio does, so a suite that leans on
sleep timing to force out-of-order completion would pin the *executor*, not the *semantics*. These tests
instead force completion order **deterministically** via per-batch :class:`asyncio.Event` gates the test
opens in a chosen order (the pattern ``test_async_transform.py::_GatedMap`` uses), so what they assert is
true of any executor that honors the fetch/integrate contract:

* ordered mode (``ordered()=True``): emission is input-order-identical however fetches complete;
* unordered mode (``ordered()=False``, stateless-only): emission follows completion order;
* the reorder buffer stays bounded: with a slow head and many blocked followers, ``max_in_flight`` caps
  the fetches in flight at once, and the results still reassemble in input order.

Driven directly through :func:`run_async_transform` over in-process channels (no DSL, no wall-clock
sleeps in the assertions) so the observed order is the driver's, reproducibly. Goldens were derived by
running the real driver — see the module notes in the accompanying report, not invented.
"""

from __future__ import annotations

import asyncio

import pyarrow as pa

from nautilus import AsyncOneInputOperator, OperatorContext
from nautilus.core.operator import Collector
from nautilus.core.records import EOS, EOS_FRAME, Batch
from nautilus.runtime.actor import Output, run_async_transform
from nautilus.runtime.channel import InProcChannel
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.partition import Forward

# --- gated operators (module-level so they deep-copy/cloudpickle cleanly, like the sibling suite) ---


class _GatedDouble(AsyncOneInputOperator):
    """A stateless async double whose every fetch blocks on a per-value :class:`asyncio.Event`. Opening the
    gates in a chosen order forces a deterministic *completion* order, so a test proves ordered/unordered
    emission without depending on sleep timing — the same trick as ``_GatedMap`` in
    ``test_async_transform.py``, kept executor-neutral here on purpose."""

    def __init__(
        self, gates: dict[int, asyncio.Event], *, ordered: bool = True, max_in_flight: int = 8
    ) -> None:
        self._gates = gates
        self._ordered = ordered
        self._cap = max_in_flight

    def max_in_flight(self) -> int:
        return self._cap

    def ordered(self) -> bool:
        return self._ordered

    async def fetch(self, batch: pa.RecordBatch) -> object:
        v = batch.column("v")[0].as_py()
        await self._gates[v].wait()
        return pa.record_batch({"v": [v * 2]})

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        out.emit(result)  # type: ignore[arg-type]


class _BoundTracker(AsyncOneInputOperator):
    """A stateless async double that blocks every fetch on a gate and counts how many fetches are inside
    :meth:`fetch` at once (``peak``). The head (``v==0``) blocks on ``head_gate``, every follower on
    ``follower_gate``; with both shut, the fetches in flight pile up to exactly ``max_in_flight`` and no
    further — the reorder buffer's bound, made observable. Releasing the followers then the head drains
    everything in input order (ordered mode)."""

    def __init__(
        self,
        head_gate: asyncio.Event,
        follower_gate: asyncio.Event,
        *,
        max_in_flight: int = 4,
    ) -> None:
        self._head_gate = head_gate
        self._follower_gate = follower_gate
        self._cap = max_in_flight
        self.in_flight = 0
        self.peak = 0

    def max_in_flight(self) -> int:
        return self._cap

    async def fetch(self, batch: pa.RecordBatch) -> object:
        self.in_flight += 1  # no await between the increment and the peak read — a race-free sample
        self.peak = max(self.peak, self.in_flight)
        try:
            v = batch.column("v")[0].as_py()
            await (self._head_gate if v == 0 else self._follower_gate).wait()
            return pa.record_batch({"v": [v * 2]})
        finally:
            self.in_flight -= 1

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        out.emit(result)  # type: ignore[arg-type]


# --- drive helpers ------------------------------------------------------------------------------


async def _pump(coro: object) -> None:
    """Yield to the event loop enough times that every currently-runnable task (feeds, fetch launches)
    reaches its next block. A fixed yield budget is deterministic — no wall-clock — and generous for the
    handful of tasks these tests run."""
    for _ in range(200):
        await asyncio.sleep(0)


async def _drive(
    op: AsyncOneInputOperator,
    vals: list[int],
    control: object,
) -> list[int]:
    """Feed ``vals`` (one row per batch) into ``op`` through :func:`run_async_transform`, run ``control``
    to open gates, and return the emitted ``v`` values in emission order. ``control`` is an async callable
    taking ``(emitted, progress)`` — ``emitted`` grows as batches arrive, ``progress`` is set on each — so
    a controller can wait for one release to drain before opening the next gate."""
    in_chan, out_chan = InProcChannel(256), InProcChannel(256)
    emitted: list[int] = []
    progress = asyncio.Event()

    async def feed() -> None:
        for v in vals:
            await in_chan.send(Batch(pa.record_batch({"v": [v]})))
        await in_chan.send(EOS_FRAME)

    async def collect() -> None:
        while True:
            fr = await out_chan.recv()
            if isinstance(fr, Batch):
                emitted.append(fr.data.column("v")[0].as_py())
                progress.set()
            elif isinstance(fr, EOS):
                return

    async def run() -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                run_async_transform(
                    op, OperatorContext("op0"), Mailbox([in_chan]), [Output([out_chan], Forward())]
                )
            )
            tg.create_task(feed())
            tg.create_task(collect())
            tg.create_task(control(emitted, progress))  # type: ignore[operator]

    await asyncio.wait_for(run(), timeout=5)
    return emitted


# --- (a) ordered mode: input-order emission regardless of completion order -----------------------


async def test_ordered_emits_input_order_despite_out_of_order_completion() -> None:
    # Every fetch is gated shut, then the gates are opened in a deliberately non-input order (later batches
    # complete first). Ordered mode holds each out-of-order completion behind the in-order frontier, so the
    # emitted sequence is the INPUT order — the reorder buffer put them back. This is the property a tokio
    # port must keep: its completion order will differ, its emission order must not.
    vals = [10, 11, 12, 13, 14]
    release_order = [14, 12, 10, 13, 11]  # deterministically out of input order
    gates = {v: asyncio.Event() for v in vals}
    op = _GatedDouble(gates, ordered=True)

    async def control(emitted: list[int], progress: asyncio.Event) -> None:
        await _pump(None)  # let all five fetches launch and block on their gates
        for v in release_order:
            gates[v].set()
            await asyncio.sleep(0)  # let this completion propagate before opening the next gate

    emitted = await _drive(op, vals, control)
    assert emitted == [v * 2 for v in vals]  # input order, not [28, 24, 20, 26, 22]


async def test_ordered_head_gated_last_still_input_order() -> None:
    # The hardest ordered case: the INPUT head (v=0) completes LAST while every later batch completes first.
    # An executor that emitted in completion order would put the head last; ordered mode pins the head first
    # and the rest in input order behind it.
    vals = [0, 1, 2, 3, 4]
    gates = {v: asyncio.Event() for v in vals}
    op = _GatedDouble(gates, ordered=True)

    async def control(emitted: list[int], progress: asyncio.Event) -> None:
        await _pump(None)
        for v in [4, 3, 2, 1, 0]:  # head released dead last
            gates[v].set()
            await asyncio.sleep(0)

    emitted = await _drive(op, vals, control)
    assert emitted == [0, 2, 4, 6, 8]  # input order; nothing emitted until the head drained


# --- (b) unordered mode: completion-order emission for the same injected completions -------------


async def test_unordered_emits_in_completion_order() -> None:
    # Same out-of-order completions as (a), but ordered()=False. Each gate is opened only after its result
    # has been observed downstream, so the completion order is pinned exactly — and the emitted sequence
    # equals that completion order, not the input order. Stateless-only, so this is sound.
    vals = [10, 11, 12, 13, 14]
    release_order = [12, 14, 10, 13, 11]
    gates = {v: asyncio.Event() for v in vals}
    op = _GatedDouble(gates, ordered=False)

    async def control(emitted: list[int], progress: asyncio.Event) -> None:
        await _pump(None)  # all fetches in flight before the first gate opens
        for v in release_order:
            gates[v].set()
            want = v * 2
            while want not in emitted:  # drain this release before opening the next gate
                progress.clear()
                if want not in emitted:
                    await progress.wait()

    emitted = await _drive(op, vals, control)
    assert emitted == [v * 2 for v in release_order]  # completion order [24, 28, 20, 26, 22]
    assert emitted != [v * 2 for v in vals]  # and demonstrably NOT the input order


async def test_unordered_slow_head_does_not_block_finished_followers() -> None:
    # The input head (v=0) is held shut while every follower completes; unordered mode emits the followers
    # immediately (in their completion order) rather than pinning them behind the slow head — the latency
    # win ordered mode forgoes. The head appears last, when its gate finally opens.
    vals = [0, 1, 2, 3]
    release_order = [3, 1, 2, 0]  # head (0) released last
    gates = {v: asyncio.Event() for v in vals}
    op = _GatedDouble(gates, ordered=False)

    async def control(emitted: list[int], progress: asyncio.Event) -> None:
        await _pump(None)
        for v in release_order:
            gates[v].set()
            want = v * 2
            while want not in emitted:
                progress.clear()
                if want not in emitted:
                    await progress.wait()

    emitted = await _drive(op, vals, control)
    assert emitted == [6, 2, 4, 0]  # completion order; the head's 0 is emitted last


# --- (c) the reorder buffer stays bounded by max_in_flight --------------------------------------


async def _bounded_run(cap: int, n: int) -> tuple[int, list[int]]:
    """Feed ``n`` batches through a ``_BoundTracker`` with ``max_in_flight=cap``, all fetches blocked, let
    the buffer fill and settle, capture the peak fetches-in-flight, then release everything and return
    ``(peak_at_settle, emitted)``. The head is released after the followers so ordered drain still starts
    from the input head."""
    head_gate, follower_gate = asyncio.Event(), asyncio.Event()
    op = _BoundTracker(head_gate, follower_gate, max_in_flight=cap)
    in_chan, out_chan = InProcChannel(256), InProcChannel(256)
    emitted: list[int] = []
    settled_peak = 0

    async def feed() -> None:
        for v in range(n):
            await in_chan.send(Batch(pa.record_batch({"v": [v]})))
        await in_chan.send(EOS_FRAME)

    async def collect() -> None:
        while True:
            fr = await out_chan.recv()
            if isinstance(fr, Batch):
                emitted.append(fr.data.column("v")[0].as_py())
            elif isinstance(fr, EOS):
                return

    async def opener() -> None:
        nonlocal settled_peak
        await _pump(None)  # every launchable fetch blocks on its gate; the buffer fills to cap
        settled_peak = op.peak  # the bound, sampled while all fetches are stalled
        follower_gate.set()
        await _pump(None)  # followers drain; the head still gates the ordered emission
        head_gate.set()

    async def run() -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                run_async_transform(
                    op, OperatorContext("op0"), Mailbox([in_chan]), [Output([out_chan], Forward())]
                )
            )
            tg.create_task(feed())
            tg.create_task(collect())
            tg.create_task(opener())

    await asyncio.wait_for(run(), timeout=5)
    return settled_peak, emitted


async def test_max_in_flight_caps_concurrent_fetches_and_results_reassemble() -> None:
    # A slow head plus many blocked followers: with 40 batches offered but max_in_flight=4, at most 4
    # fetches are ever inside fetch() at once (never all 40), and once released the whole stream still
    # reassembles in input order. The bound is what stops an unbounded reorder buffer.
    peak, emitted = await _bounded_run(cap=4, n=40)
    assert peak == 4  # exactly the bound while stalled — never more outstanding at once
    assert emitted == [v * 2 for v in range(40)]  # and still input-order-correct


async def test_max_in_flight_bound_holds_across_caps() -> None:
    # The peak-in-flight tracks the configured bound exactly (2, 4, 8), each time reassembling correctly —
    # the buffer is bounded by max_in_flight, not by the number of offered batches.
    for cap in (2, 4, 8):
        peak, emitted = await _bounded_run(cap=cap, n=40)
        assert peak == cap
        assert emitted == [v * 2 for v in range(40)]
