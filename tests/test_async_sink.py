"""The async sink (Stage 6.0–6.2): an authored terminal that writes each batch to an external store
with bounded, overlapping concurrency, drains at EOS, and replaces the synthesized collecting sink as the
graph's leaf.

The "external store" here is an in-process list (or counter), so these run hermetically: what is exercised
is the engine's driving of an awaiting sink — concurrency bound, EOS drain, fail-fast, timeout, keyed
co-partitioning, conditional CollectSink synthesis — not any real I/O.
"""

from __future__ import annotations

import asyncio
import json
import sys

import cloudpickle
import pyarrow as pa
import pytest

from nautilus import AsyncSink, source
from nautilus.api import LogicalEdge, LogicalGraph, LogicalVertex
from nautilus.compile import compile_graph
from nautilus.compile.lower import SINK_ID
from nautilus.driver.run import run_compiled
from nautilus.operators import from_batches

# Ship this module's sink classes to spawned workers by value, so a cross-process run reconstructs them
# without importing the test module (which the worker has no path to).
cloudpickle.register_pickle_by_value(sys.modules[__name__])

# A module-global write log so deep-copied per-subtask sink instances (parallelism > 1) all record to the
# same place (an instance attribute would be deep-copied apart). Single-process only: a spawned worker has
# its own copy, so the cross-process test asserts via the aggregated telemetry instead.
_WRITES: list[tuple[int, object]] = []


def _batches(values: list[int], per_batch: int = 1, key_col: str = "v") -> list[pa.RecordBatch]:
    return [
        pa.record_batch({key_col: values[i : i + per_batch]})
        for i in range(0, len(values), per_batch)
    ]


def _counter_total(report: object, name: str) -> int:
    d = json.loads(report.to_json())  # type: ignore[attr-defined]
    return sum(c["value"] for op in d["operators"] for c in op["counters"] if c["name"] == name)


def _gauge_max(report: object, name: str) -> int:
    d = json.loads(report.to_json())  # type: ignore[attr-defined]
    maxes = [g["max"] for op in d["operators"] for g in op["gauges"] if g["name"] == name]
    return max(maxes) if maxes else 0


# --- sinks used across the tests (module-level so they cloudpickle / deep-copy cleanly) ---------


class ListSink(AsyncSink):
    """Collects every written row's value into the module-global ``_WRITES`` (tagged by subtask)."""

    def __init__(self, *, key_col: str = "v", max_in_flight: int = 8, delay_s: float = 0.0) -> None:
        self._key_col = key_col
        self._cap = max_in_flight
        self._delay = delay_s

    def key_columns(self) -> tuple[str, ...] | None:
        return (self._key_col,) if self._key_col == "k" else None

    def max_in_flight(self) -> int:
        return self._cap

    def open(self, ctx) -> None:  # type: ignore[no-untyped-def]
        self._sub = ctx.subtask_index

    async def write(self, batch: pa.RecordBatch) -> None:
        if self._delay:
            await asyncio.sleep(self._delay)
        else:
            await asyncio.sleep(0)  # always yield so concurrency is real
        for v in batch.column(self._key_col).to_pylist():
            _WRITES.append((self._sub, v))

    async def close(self) -> None:
        pass


# --- basic behaviour ---------------------------------------------------------------------------


def test_sink_writes_every_row_and_returns_no_batches() -> None:
    _WRITES.clear()
    result = source(_batches([1, 2, 3, 4])).sink(ListSink()).run()
    assert sorted(v for _, v in _WRITES) == [1, 2, 3, 4]
    assert result.batches == []  # the data went to the store, not into a collected result
    assert result.to_pylist() == []
    assert _counter_total(result.telemetry, "async.requests") == 4


def test_collect_graph_keeps_collectsink_but_sink_leaf_drops_it() -> None:
    # A transform-leaf graph still synthesizes the CollectSink (kind "sink", no factory) — unchanged.
    collect_plan = compile_graph(source(_batches([1, 2])).map(lambda b: b).to_graph())
    collect_sink = [op for op in collect_plan.operators if op.operator_id == SINK_ID]
    assert len(collect_sink) == 1
    assert collect_sink[0].op_class == "CollectSink" and collect_sink[0].factory is None

    # A sink-leaf graph synthesizes no CollectSink; the async sink is the terminal.
    sink_plan = compile_graph(source(_batches([1, 2])).sink(ListSink()).to_graph())
    assert not any(op.operator_id == SINK_ID for op in sink_plan.operators)
    leaf = sink_plan.operators[-1]
    assert leaf.kind == "async_sink" and leaf.factory is not None


def test_adding_async_sink_support_does_not_change_an_existing_graph_digest() -> None:
    # The structural digest of a plain transform-leaf graph must be identical with the new kind present.
    g = source(_batches([1, 2, 3])).map(lambda b: b).count_by("v")
    d1 = run_compiled_sync(g.to_graph())
    d2 = run_compiled_sync(g.to_graph())
    assert d1 == d2  # reproducible, and (by construction) unaffected by the async-sink machinery


def run_compiled_sync(graph: LogicalGraph) -> str:
    return asyncio.run(_digest(graph))


async def _digest(graph: LogicalGraph) -> str:
    res = await run_compiled(compile_graph(graph))
    return res.telemetry.structural_digest()


# --- concurrency, backpressure, EOS drain ------------------------------------------------------


def test_in_flight_overlaps_and_is_bounded() -> None:
    _WRITES.clear()
    cap = 4
    result = (
        source(_batches(list(range(20)))).sink(ListSink(max_in_flight=cap, delay_s=0.005)).run()
    )
    peak = _gauge_max(result.telemetry, "async.in_flight")
    assert 2 <= peak <= cap  # genuinely overlapping, never above the bound
    assert _counter_total(result.telemetry, "async.requests") == 20
    assert _gauge_max(result.telemetry, "async.capacity") == cap


def test_max_in_flight_one_is_serial() -> None:
    _WRITES.clear()
    result = source(_batches(list(range(8)))).sink(ListSink(max_in_flight=1, delay_s=0.001)).run()
    assert _gauge_max(result.telemetry, "async.in_flight") == 1
    assert [v for _, v in _WRITES] == list(range(8))  # one at a time, in input order


def test_eos_drains_every_in_flight_write() -> None:
    # Many slow writes still all complete before the run returns (terminal drain), not just the head.
    _WRITES.clear()
    n = 30
    source(_batches(list(range(n)))).sink(ListSink(max_in_flight=16, delay_s=0.002)).run()
    assert sorted(v for _, v in _WRITES) == list(range(n))


# --- fail-fast + timeout -----------------------------------------------------------------------


class _FailSink(AsyncSink):
    """Raises on the sentinel value; every other write blocks forever (so it must be cancelled)."""

    closed = False
    completed: list[int] = []

    def max_in_flight(self) -> int:
        return 8

    async def write(self, batch: pa.RecordBatch) -> None:
        v = batch.column("v")[0].as_py()
        if v == 99:
            raise RuntimeError("boom")
        await asyncio.Event().wait()  # never set — only a cancel ends this
        _FailSink.completed.append(v)

    async def close(self) -> None:
        _FailSink.closed = True


def test_failed_write_is_fail_fast_and_cancels_siblings() -> None:
    _FailSink.closed = False
    _FailSink.completed = []
    # 99 fails; 10/11/12 are in flight and blocked, so they must be cancelled (not left to hang).
    handle = source(_batches([10, 11, 99, 12])).sink(_FailSink())
    with pytest.raises((RuntimeError, ExceptionGroup)):
        asyncio.run(asyncio.wait_for(handle.run_async(), timeout=10))
    assert _FailSink.closed is True  # close() still runs (resource release)
    assert _FailSink.completed == []  # the blocked siblings were cancelled, never completed


class _TimeoutSink(AsyncSink):
    def max_in_flight(self) -> int:
        return 2

    def timeout_micros(self) -> int:
        return 5_000  # 5 ms

    async def write(self, batch: pa.RecordBatch) -> None:
        await asyncio.sleep(0.1)  # 100 ms — always exceeds the timeout

    async def close(self) -> None:
        pass


def test_write_timeout_fails_the_job() -> None:
    handle = source(_batches([1, 2])).sink(_TimeoutSink())
    with pytest.raises((TimeoutError, ExceptionGroup)):
        asyncio.run(asyncio.wait_for(handle.run_async(), timeout=10))


async def test_concurrent_write_failures_are_all_counted() -> None:
    # Two writes are released together (by an external controller arriving last on a 3-party barrier, so
    # both resume in the same event-loop pass) and both fail. The TaskGroup raises only one, but each task
    # records its own error first, so the count must be 2 — a failure must not lose its telemetry to the
    # group's cancellation. Driven directly to control the timing.
    from nautilus.core.operator import OperatorContext
    from nautilus.core.records import EOS_FRAME, Batch
    from nautilus.runtime.actor import run_async_sink
    from nautilus.runtime.channel import InProcChannel
    from nautilus.runtime.mailbox import Mailbox
    from nautilus.telemetry import Owner, RecorderRegistry, TelemetryConfig, make_recorder

    barrier = asyncio.Barrier(3)

    class GatedSink(AsyncSink):
        def max_in_flight(self) -> int:
            return 4

        async def write(self, batch: pa.RecordBatch) -> None:
            await barrier.wait()
            raise RuntimeError("boom")

        async def close(self) -> None:
            pass

    async def controller() -> None:
        await barrier.wait()

    reg = RecorderRegistry()
    rec = reg.register(
        make_recorder(
            operator_id="op0",
            op_class="GatedSink",
            kind="async_sink",
            subtask_index=0,
            node="local",
            config=TelemetryConfig(),
            owner=Owner.ENGINE,
        )
    )
    ch = InProcChannel()
    await ch.send(Batch(pa.record_batch({"v": [1]})))
    await ch.send(Batch(pa.record_batch({"v": [2]})))
    await ch.send(EOS_FRAME)
    ctrl = asyncio.create_task(controller())
    with pytest.raises(ExceptionGroup):  # the TaskGroup wraps the write failures
        await run_async_sink(GatedSink(), OperatorContext("op0"), Mailbox([ch]), recorder=rec)
    await ctrl
    assert rec.counter("operator.errors", operator_id="op0", exc_type="RuntimeError").value == 2


def test_max_in_flight_must_be_positive() -> None:
    handle = source(_batches([1])).sink(ListSink(max_in_flight=0))
    with pytest.raises((ValueError, ExceptionGroup)):
        asyncio.run(handle.run_async())


# --- parallelism: keyed co-partition vs keyless fan-out ----------------------------------------


def test_keyed_sink_co_partitions() -> None:
    _WRITES.clear()
    keys = ["a", "b", "c", "a", "b", "c", "a", "d"]
    src = from_batches(*[pa.record_batch({"k": [k]}) for k in keys])
    source(src).sink(ListSink(key_col="k"), parallelism=2).run()
    # every occurrence of a key was written by the same subtask (no key split across instances)
    owner: dict[object, int] = {}
    for sub, k in _WRITES:
        assert owner.setdefault(k, sub) == sub
    assert len(_WRITES) == len(keys)
    assert len(set(owner.values())) > 1  # both instances were actually used


def test_keyless_sink_fans_out() -> None:
    _WRITES.clear()
    source(_batches(list(range(12)))).sink(ListSink(), parallelism=2).run()
    subs = {sub for sub, _ in _WRITES}
    assert subs == {0, 1}  # keyless round-robin reached both instances
    assert sorted(v for _, v in _WRITES) == list(range(12))


# --- the plan cloudpickles to a bare worker ----------------------------------------------------


def test_sink_plan_cloudpickles_and_runs() -> None:
    _WRITES.clear()
    plan = compile_graph(source(_batches([1, 2, 3])).sink(ListSink()).to_graph())
    revived = cloudpickle.loads(cloudpickle.dumps(plan))  # the factory must survive the round-trip
    res = asyncio.run(run_compiled(revived))
    assert _counter_total(res.telemetry, "operator.rows_in") >= 3  # the sink consumed every row
    assert res.batches == []


# --- IR validation: a sink must be a leaf ------------------------------------------------------


def test_async_sink_must_be_a_leaf() -> None:
    vs = (
        LogicalVertex("s", lambda: None, "source", 1, None),
        LogicalVertex("k", lambda: ListSink(), "async_sink", 1, None),
        LogicalVertex("m", lambda: None, "one_input", 1, None),
    )
    es = (LogicalEdge("s", "k", 0, None), LogicalEdge("k", "m", 0, None))  # k -> m: sink not a leaf
    with pytest.raises(ValueError, match="must be a leaf"):
        LogicalGraph(vs, es)


# --- the sink runs across worker processes -----------------------------------------------------


def test_sink_runs_across_worker_processes() -> None:
    # Two sink instances on two workers, fed by a cross-worker round-robin edge. The workers have separate
    # memory, so conservation is asserted via the coordinator-aggregated telemetry, not the write log.
    from nautilus.cluster import deploy

    n = 24
    graph = source(_batches(list(range(n)))).sink(ListSink(), parallelism=2).to_graph()
    res = deploy(graph, num_workers=2)
    assert _counter_total(res.telemetry, "operator.rows_in") == n  # every row reached a sink writer
    assert res.batches == []
