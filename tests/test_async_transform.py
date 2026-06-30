"""The async transform (Stage 6.3): an intermediate :class:`AsyncOneInputOperator` whose I/O is awaited
in ``fetch`` (run as bounded, overlapping concurrent tasks) and whose result is folded into state and
emitted in a synchronous ``integrate``, in input order.

The "external I/O" here is an ``asyncio.sleep`` plus an in-process transform, so these run hermetically:
what is exercised is the engine's driver — ordered emission under out-of-order completion, the watermark/
EOS barriers, the in-flight bound, fail-fast, the per-request timeout, the state guard, and keyed
co-partitioning — not any real network.
"""

from __future__ import annotations

import asyncio
import json
import sys

import cloudpickle
import pyarrow as pa
import pytest

from nautilus import AsyncMapBatch, AsyncOneInputOperator, OperatorContext, source
from nautilus.api import LogicalEdge, LogicalGraph, LogicalVertex
from nautilus.api.graph import async_one_input, one_input
from nautilus.api.graph import source as source_vertex
from nautilus.compile import compile_graph
from nautilus.compile.lower import SINK_ID
from nautilus.core.operator import Collector, StateAccessError
from nautilus.core.records import EOS, EOS_FRAME, WATERMARK_MAX, Batch, Watermark
from nautilus.driver.run import run_compiled
from nautilus.operators import MapBatch, _add, from_batches
from nautilus.runtime.actor import Output, run_async_transform
from nautilus.runtime.channel import InProcChannel
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.partition import Forward
from nautilus.state import KeyContext

# Ship this module's operator classes/functions to spawned workers by value, so a cross-process run
# reconstructs them without importing the test module (which the worker has no path to).
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def _batches(values: list[int], per_batch: int = 1) -> list[pa.RecordBatch]:
    return [
        pa.record_batch({"v": values[i : i + per_batch]}) for i in range(0, len(values), per_batch)
    ]


def _counter_total(report: object, name: str) -> int:
    d = json.loads(report.to_json())  # type: ignore[attr-defined]
    return sum(c["value"] for op in d["operators"] for c in op["counters"] if c["name"] == name)


def _gauge_max(report: object, name: str) -> int:
    d = json.loads(report.to_json())  # type: ignore[attr-defined]
    maxes = [g["max"] for op in d["operators"] for g in op["gauges"] if g["name"] == name]
    return max(maxes) if maxes else 0


# --- async fns / operators (module-level so they cloudpickle / deep-copy cleanly) ---------------


async def _double(batch: pa.RecordBatch) -> pa.RecordBatch:
    # Reverse latency: smaller (earlier) values sleep LONGER, so fetches complete out of input order and
    # the reorder buffer must put them back. The output must still be in input order.
    v = batch.column("v")[0].as_py()
    await asyncio.sleep((20 - v % 20) * 0.001)
    return pa.record_batch({"v": [x * 2 for x in batch.column("v").to_pylist()]})


def _double_sync(batch: pa.RecordBatch) -> pa.RecordBatch:
    return pa.record_batch({"v": [x * 2 for x in batch.column("v").to_pylist()]})


class AsyncKeyedCount(AsyncOneInputOperator):
    """A keyed async enrich: ``fetch`` does the (awaited) I/O — here just reads the key column — and
    ``integrate`` folds a per-key count into keyed state, flushed at EOS. The point is that the keyed
    state stays single-writer while many fetches overlap."""

    _STATE = "count"

    def __init__(
        self, key_col: str = "k", *, max_in_flight: int = 8, delay_s: float = 0.001
    ) -> None:
        self._key_col = key_col
        self._cap = max_in_flight
        self._delay = delay_s

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx

    def key_columns(self) -> tuple[str, ...]:
        return (self._key_col,)

    def max_in_flight(self) -> int:
        return self._cap

    async def fetch(self, batch: pa.RecordBatch) -> object:
        await asyncio.sleep(self._delay)
        return batch.column(self._key_col).to_pylist()

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        for k in result:  # type: ignore[attr-defined]
            ctx.reducing_state(self._STATE, KeyContext((k,)), _add).add(1)

    def on_watermark(self, t: int, ctx: OperatorContext, out: Collector) -> None:
        if t < WATERMARK_MAX:
            return
        keys: list[object] = []
        totals: list[int] = []
        fired: list[KeyContext] = []
        for kctx, value in ctx.entries(self._STATE):
            keys.append(kctx.key[0])
            totals.append(value)  # type: ignore[arg-type]
            fired.append(kctx)
        for kctx in fired:
            ctx.clear_state(self._STATE, kctx)
        if keys:
            out.emit(pa.record_batch({self._key_col: keys, "count": pa.array(totals, pa.int64())}))


class StampSubtask(AsyncOneInputOperator):
    """A keyed async enrich that stamps which subtask integrated each row, so a parallel run can prove a
    key never split across instances."""

    def __init__(self, key_col: str = "k") -> None:
        self._key_col = key_col

    def open(self, ctx: OperatorContext) -> None:
        self._sub = ctx.subtask_index

    def key_columns(self) -> tuple[str, ...]:
        return (self._key_col,)

    async def fetch(self, batch: pa.RecordBatch) -> object:
        await asyncio.sleep(0)
        return batch.column(self._key_col).to_pylist()

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        ks = list(result)  # type: ignore[call-overload]
        out.emit(pa.record_batch({"k": ks, "sub": pa.array([self._sub] * len(ks), pa.int64())}))


# --- ordering: equals the sync map byte-for-byte, digest stable --------------------------------


def test_ordered_map_async_equals_sync_map() -> None:
    vals = list(range(24))
    got = source(_batches(vals, per_batch=2)).map_async(_double, max_in_flight=8).collect()
    want = source(_batches(vals, per_batch=2)).map(_double_sync).collect()
    assert got == want  # ordered output is identical despite out-of-order fetch completion
    assert [r["v"] for r in got] == [v * 2 for v in vals]


def test_digest_is_stable_across_latency_trials() -> None:
    vals = list(range(30))

    def digest_async() -> str:
        return (
            source(_batches(vals, per_batch=3))
            .map_async(_double, max_in_flight=8)
            .run()
            .telemetry.structural_digest()
        )

    # Latency-driven nondeterminism (out-of-order fetch completion) must not reach the structural digest:
    # ordered emission keeps rows/batches/watermark counts reproducible across trials. (The digest differs
    # from the sync map's by design — it includes op_class/kind — so output equality is asserted
    # separately in test_ordered_map_async_equals_sync_map.)
    assert digest_async() == digest_async()


# --- concurrency / backpressure ----------------------------------------------------------------


def test_in_flight_overlaps_and_is_bounded() -> None:
    cap = 4
    result = source(_batches(list(range(20)))).map_async(_double, max_in_flight=cap).run()
    peak = _gauge_max(result.telemetry, "async.in_flight")
    assert 2 <= peak <= cap  # genuinely overlapping, never above the bound
    assert _counter_total(result.telemetry, "async.requests") == 20
    assert _gauge_max(result.telemetry, "async.capacity") == cap


def test_max_in_flight_one_is_serial_and_ordered() -> None:
    result = source(_batches(list(range(8)))).map_async(_double, max_in_flight=1).run()
    assert _gauge_max(result.telemetry, "async.in_flight") == 1
    assert [r["v"] for r in result.to_pylist()] == [v * 2 for v in range(8)]


def test_slow_head_keeps_reorder_buffer_bounded() -> None:
    # The head sleeps far longer than the tail, so many tails finish first; the buffer (== in_flight) must
    # still never exceed the bound, because inflight is decremented only on pop.
    cap = 4
    result = source(_batches(list(range(40)))).map_async(_double, max_in_flight=cap).run()
    assert _gauge_max(result.telemetry, "async.in_flight") <= cap
    assert [r["v"] for r in result.to_pylist()] == [v * 2 for v in range(40)]


# --- keyed enrich folding into state -----------------------------------------------------------


def test_keyed_async_enrich_matches_sync_count() -> None:
    keys = ["a", "b", "a", "c", "b", "a", "d", "a"]
    src = [pa.record_batch({"k": [k]}) for k in keys]
    got = source(src).apply_async(AsyncKeyedCount(), key_columns="k").collect()
    want = source(src).count_by("k").collect()
    assert sorted((r["k"], r["count"]) for r in got) == sorted((r["k"], r["count"]) for r in want)


def test_keyed_async_co_partitions_across_instances() -> None:
    keys = ["a", "b", "c", "a", "b", "c", "a", "d", "b", "c"]
    src = from_batches(*[pa.record_batch({"k": [k]}) for k in keys])
    rows = source(src).apply_async(StampSubtask(), key_columns="k", parallelism=2).collect()
    owner: dict[object, int] = {}
    for r in rows:
        assert owner.setdefault(r["k"], r["sub"]) == r["sub"]  # a key never split across instances
    assert len(rows) == len(keys)
    assert len(set(owner.values())) > 1  # both instances were actually used


# --- EOS / watermark ordering (direct drive, asserting order not just counts) -------------------


async def _drive_and_capture(op: AsyncOneInputOperator, frames: list[object]) -> list[object]:
    """Drive a transform over one input and return the exact output FRAME sequence (data + control), so a
    test can assert a watermark never overtakes the data before it and EOS comes last."""
    in_chan = InProcChannel(256)
    out_chan = InProcChannel(256)
    captured: list[object] = []

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
            run_async_transform(
                op, OperatorContext("op0"), Mailbox([in_chan]), [Output([out_chan], Forward())]
            )
        )
        tg.create_task(feed())
        tg.create_task(collect())
    return captured


async def test_eos_and_watermarks_drain_in_order() -> None:
    # data1's fetch is slower than data2's, so data2 completes first — yet ordered emission + the marker
    # barrier must yield Batch(1*2), Watermark(5), Batch(2*2), Watermark(10), then EOS. A watermark never
    # overtakes the data before it, and the terminal close emits no spurious Watermark(WATERMARK_MAX): it
    # fires on_watermark locally and signals completion with EOS, exactly like the synchronous loop.
    frames = [
        Batch(pa.record_batch({"v": [1]})),
        Watermark(5),
        Batch(pa.record_batch({"v": [2]})),
        Watermark(10),
        EOS_FRAME,
    ]
    out = await _drive_and_capture(AsyncMapBatch(_double), frames)
    kinds = [
        (
            ("B", o.data.column("v")[0].as_py())
            if isinstance(o, Batch)
            else ("W", o.t) if isinstance(o, Watermark) else ("EOS", None)
        )
        for o in out
    ]
    assert kinds == [("B", 2), ("W", 5), ("B", 4), ("W", 10), ("EOS", None)]
    assert WATERMARK_MAX not in [o.t for o in out if isinstance(o, Watermark)]


def _frame_seq(out: list[object]) -> list[object]:
    return [
        (
            ("B", tuple(o.data.column("v").to_pylist()))
            if isinstance(o, Batch)
            else ("W", o.t) if isinstance(o, Watermark) else ("EOS", None)
        )
        for o in out
    ]


async def test_async_loop_matches_sync_loop_frame_for_frame() -> None:
    # The strong oracle: the async transform's full OUTPUT frame sequence (data + watermarks + EOS) is
    # identical to the synchronous MapBatch loop's over the same watermark-rich input — so the async loop
    # forwards watermarks/EOS exactly as the proven loop, not merely the same row count.
    from nautilus.operators import MapBatch
    from nautilus.runtime.actor import run_transform

    frames = [
        Batch(pa.record_batch({"v": [3]})),
        Batch(pa.record_batch({"v": [1]})),
        Watermark(7),
        Batch(pa.record_batch({"v": [2]})),
        Watermark(9),
        Batch(pa.record_batch({"v": [5]})),
        EOS_FRAME,
    ]

    async def drive_sync() -> list[object]:
        in_chan, out_chan = InProcChannel(64), InProcChannel(64)
        captured: list[object] = []

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
                    MapBatch(_double_sync),
                    OperatorContext("op0"),
                    Mailbox([in_chan]),
                    [Output([out_chan], Forward())],
                )
            )
            tg.create_task(feed())
            tg.create_task(collect())
        return captured

    async_out = await _drive_and_capture(AsyncMapBatch(_double), frames)
    sync_out = await drive_sync()
    assert _frame_seq(async_out) == _frame_seq(sync_out)


async def test_eos_drains_many_slow_fetches_then_terminates() -> None:
    n = 25
    frames = [Batch(pa.record_batch({"v": [i]})) for i in range(n)] + [EOS_FRAME]
    out = await _drive_and_capture(AsyncMapBatch(_double, max_in_flight=16), frames)
    data = [o.data.column("v")[0].as_py() for o in out if isinstance(o, Batch)]
    assert data == [i * 2 for i in range(n)]  # all drained, in input order
    assert isinstance(out[-1], EOS)


# --- fail-fast + timeout -----------------------------------------------------------------------


class _FailFetch(AsyncOneInputOperator):
    """Fetch raises on the sentinel value; every other fetch blocks forever (so it must be cancelled)."""

    completed: list[int] = []

    def max_in_flight(self) -> int:
        return 8

    async def fetch(self, batch: pa.RecordBatch) -> object:
        v = batch.column("v")[0].as_py()
        if v == 99:
            raise RuntimeError("boom")
        await asyncio.Event().wait()  # never set — only a cancel ends this
        _FailFetch.completed.append(v)
        return batch

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        out.emit(batch)


def test_failed_fetch_is_fail_fast_and_cancels_siblings() -> None:
    _FailFetch.completed = []
    handle = source(_batches([10, 11, 99, 12])).apply_async(_FailFetch())
    with pytest.raises((RuntimeError, ExceptionGroup)):
        asyncio.run(asyncio.wait_for(handle.run_async(), timeout=10))
    assert _FailFetch.completed == []  # the blocked siblings were cancelled, never completed


class _TimeoutFetch(AsyncOneInputOperator):
    def max_in_flight(self) -> int:
        return 2

    def timeout_micros(self) -> int:
        return 5_000  # 5 ms

    async def fetch(self, batch: pa.RecordBatch) -> object:
        await asyncio.sleep(0.1)  # 100 ms — always exceeds the timeout
        return batch

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        out.emit(batch)


async def test_fetch_timeout_fails_job_and_counts() -> None:
    # Driven directly so the timeout counter can be read off the recorder before the abort.
    from nautilus.telemetry import Owner, RecorderRegistry, TelemetryConfig, make_recorder

    reg = RecorderRegistry()
    rec = reg.register(
        make_recorder(
            operator_id="op0",
            op_class="_TimeoutFetch",
            kind="async_one_input",
            subtask_index=0,
            node="local",
            config=TelemetryConfig(),
            owner=Owner.ENGINE,
        )
    )
    ch = InProcChannel(8)
    await ch.send(Batch(pa.record_batch({"v": [1]})))
    await ch.send(EOS_FRAME)
    with pytest.raises(TimeoutError):
        await run_async_transform(
            _TimeoutFetch(), OperatorContext("op0"), Mailbox([ch]), [], recorder=rec
        )
    assert rec.counter("async.timeouts", operator_id="op0", subtask_index=0).value == 1


# --- the state guard ---------------------------------------------------------------------------


class _StateFromFetch(AsyncOneInputOperator):
    """Illegally reaches keyed state from the awaiting half — the engine must turn this into a loud raise."""

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx  # stashing ctx is the trap the guard is built to catch

    async def fetch(self, batch: pa.RecordBatch) -> object:
        self._ctx.reducing_state("x", KeyContext(("k",)), _add).add(1)  # raises StateAccessError
        return batch

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        out.emit(batch)


def test_state_access_from_fetch_raises() -> None:
    handle = source(_batches([1, 2, 3])).apply_async(_StateFromFetch())
    with pytest.raises((StateAccessError, ExceptionGroup)):
        asyncio.run(handle.run_async())


def test_ordered_false_is_rejected() -> None:
    class _Unordered(AsyncMapBatch):
        def ordered(self) -> bool:
            return False

    handle = source(_batches([1, 2])).apply_async(_Unordered(_double))
    with pytest.raises((NotImplementedError, ExceptionGroup)):
        asyncio.run(handle.run_async())


def test_max_in_flight_must_be_positive() -> None:
    handle = source(_batches([1])).map_async(_double, max_in_flight=0)
    with pytest.raises((ValueError, ExceptionGroup)):
        asyncio.run(handle.run_async())


# --- compile / lowering ------------------------------------------------------------------------


def test_async_transform_leaf_still_synthesizes_collectsink() -> None:
    # An async_one_input as the leaf is NOT a sink, so the collecting sink is still synthesized and its
    # output is collected — unlike an async_sink leaf.
    plan = compile_graph(source(_batches([1, 2])).map_async(_double).to_graph())
    collect = [op for op in plan.operators if op.operator_id == SINK_ID]
    assert len(collect) == 1 and collect[0].op_class == "CollectSink"
    leaf = next(op for op in plan.operators if op.kind == "async_one_input")
    assert leaf.factory is not None


def test_adding_async_transform_does_not_change_existing_digest() -> None:
    g = source(_batches([1, 2, 3])).map(_double_sync).count_by("v")
    d1 = g.run().telemetry.structural_digest()
    d2 = g.run().telemetry.structural_digest()
    assert d1 == d2  # reproducible, and unaffected by the async-transform machinery


def test_async_transform_in_linear_graph_is_rejected() -> None:
    vs = (
        LogicalVertex("s", lambda: None, "source", 1, None),
        LogicalVertex("a", lambda: AsyncMapBatch(_double), "async_one_input", 1, None),
    )
    with pytest.raises(ValueError, match="needs explicit edges"):
        LogicalGraph(vs)  # no edges -> linear shape, which has no place for an awaiting transform


# --- cross-process -----------------------------------------------------------------------------


def test_async_transform_plan_cloudpickles_and_runs() -> None:
    plan = compile_graph(source(_batches([1, 2, 3])).map_async(_double).to_graph())
    revived = cloudpickle.loads(cloudpickle.dumps(plan))  # the factory must survive the round-trip
    res = asyncio.run(run_compiled(revived))
    assert sorted(r["v"] for r in res.to_pylist()) == [2, 4, 6]


def test_async_transform_runs_across_worker_processes() -> None:
    # A keyed async enrich at parallelism 2 across two workers: separate memory, so conservation is
    # asserted via the coordinator-aggregated telemetry.
    from nautilus.cluster import deploy

    keys = ["a", "b", "c", "a", "b", "c", "a", "d"] * 2
    graph = (
        source(from_batches(*[pa.record_batch({"k": [k]}) for k in keys]))
        .apply_async(AsyncKeyedCount(), key_columns="k", parallelism=2)
        .to_graph()
    )
    res = deploy(graph, num_workers=2)
    assert _counter_total(res.telemetry, "operator.rows_in") >= len(keys)
    # every key's total count is conserved across the two writers' emitted rows
    counts: dict[object, int] = {}
    for r in res.to_pylist():
        counts[r["k"]] = counts.get(r["k"], 0) + r["count"]
    assert counts == {"a": 6, "b": 4, "c": 4, "d": 2}


def test_async_one_input_builder_in_explicit_edge_graph() -> None:
    # The api.graph builder for an async transform composes in a DAG with a downstream sync transform.
    g = LogicalGraph(
        (
            source_vertex("s", lambda: from_batches(pa.record_batch({"v": [1, 2]}))),
            async_one_input("a", lambda: AsyncMapBatch(_double)),
            one_input("m", lambda: MapBatch(_double_sync)),
        ),
        (LogicalEdge("s", "a", 0, None), LogicalEdge("a", "m", 0, None)),
    )
    out = asyncio.run(run_compiled(compile_graph(g)))
    assert sorted(r["v"] for r in out.to_pylist()) == [4, 8]  # *2 (async) then *2 (sync)
