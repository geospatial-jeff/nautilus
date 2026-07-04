"""Telemetry that performance analysis needs: occupancy is countable (on_eos time is in
step_micros), state growth is visible (state.entries/keys), and keyed-shuffle cost is attributed
(partition.route_micros). These close gaps that previously left the engine's hottest costs invisible.
"""

from nautilus.core.records import EOS_FRAME
from nautilus.core.time import TestClock
from nautilus.driver.local import run_local_chain
from nautilus.driver.pipeline import graph_from_pipeline
from nautilus.driver.run import run_plan
from nautilus.operators import InMemorySource, KeyedCount
from nautilus.state import InMemoryStateBackend, StateScope
from nautilus.telemetry.catalog import Tier
from nautilus.telemetry.recorder import TelemetryConfig
from nautilus.testing import data


def _keyed():
    # Two distinct keys, each present in both batches. KeyedCount holds one count entry per key and
    # flushes them all at end of stream — the on_eos work the timing/state assertions below observe.
    frames = [
        data(key=["a", "a", "b"]),
        data(key=["a", "b"]),
        EOS_FRAME,
    ]
    return InMemorySource(frames), [KeyedCount("key")]


def _gauge_max(op, name):
    return max((g.max for g in op.gauges if g.name == name), default=None)


def _counter(op, name):
    return sum(p.value for p in op.counters if p.name == name)


async def test_step_micros_includes_on_eos_time():
    src, ops = _keyed()
    rep = (await run_local_chain(src, ops, clock=TestClock())).telemetry
    op0 = rep.operator("op0")
    step = _counter(op0, "runtime.step_micros")
    proc = next(h for h in op0.histograms if h.name == "operator.process_micros")
    eos = next(h for h in op0.histograms if h.name == "operator.on_eos_micros")
    # busy/self-time covers both synchronous critical sections (catalog meaning). step accumulates in
    # nanoseconds and carries the sub-µs remainder, so it is at least the per-call histograms' truncated
    # sums and exceeds them by at most one microsecond per call.
    assert proc.sum + eos.sum <= step <= proc.sum + eos.sum + proc.count + eos.count
    assert eos.sum > 0  # this operator does real work in on_eos (the end-of-stream flush)


async def test_state_entries_and_keys_track_keyed_state():
    src, ops = _keyed()
    rep = (await run_local_chain(src, ops, clock=TestClock())).telemetry
    op0 = rep.operator("op0")
    # KeyedCount holds one count entry per distinct key; keys a and b -> 2 entries across 2 distinct
    # keys, sampled at end of stream before on_eos clears them.
    assert _gauge_max(op0, "state.entries") == 2
    assert _gauge_max(op0, "state.keys") == 2


async def test_state_gauges_absent_when_telemetry_off():
    src, ops = _keyed()
    rep = (await run_local_chain(src, ops, telemetry=TelemetryConfig(tier=Tier.OFF))).telemetry
    assert rep.operator("op0") is None  # OFF records nothing at all


async def test_partition_route_micros_attributes_the_keyed_shuffle():
    src, ops = _keyed()
    # parallelism 3 -> source->op0 is a keyed shuffle that routes
    graph = graph_from_pipeline(src, ops, 3)
    rep = (await run_plan(graph, telemetry=TelemetryConfig(tier=Tier.COUNTERS))).telemetry
    src_op = next(o for o in rep.operators if o.operator_id == "source")
    routed = [h for h in src_op.histograms if h.name == "partition.route_micros"]
    assert routed, "the shuffling source must record partition.route_micros"
    assert all(h.labels for h in routed)  # labeled by (operator_id, edge_dst)
    assert sum(h.count for h in routed) > 0


def test_micros_accumulator_carries_sub_microsecond_remainders():
    from nautilus.runtime.actor import _MicrosAccumulator
    from nautilus.telemetry.model import Counter

    c = Counter()
    acc = _MicrosAccumulator(c)
    for _ in range(2000):
        acc.add_ns(600)  # 0.6µs each — truncating per add would floor every one to 0 (lose it all)
    assert c.value == 1200  # 2000 * 600ns = 1,200,000ns = 1200µs, recovered exactly by carrying


async def test_source_generation_time_is_recorded_as_self_time():
    from nautilus.benchmarks import SyntheticKeyedSource, passthrough
    from nautilus.operators import MapBatch

    # A source doing real (numpy) generation work must show non-zero self-time, not read as idle.
    src = SyntheticKeyedSource(num_batches=50, batch_rows=2048, key_cardinality=100)
    rep = (await run_local_chain(src, [MapBatch(passthrough)])).telemetry
    assert _counter(rep.operator("source"), "runtime.step_micros") > 0


async def test_queue_depth_histogram_records_a_distribution():
    src, ops = _keyed()
    rep = (await run_local_chain(src, ops, clock=TestClock())).telemetry
    # Every in-process edge with sends has a depth distribution (counts how often at each fill level),
    # not just the high-water gauge.
    hists = [h for o in rep.operators for h in o.histograms if h.name == "edge.queue_depth_hist"]
    assert hists and sum(h.count for h in hists) > 0


def test_backend_sizes_count_entries_and_distinct_keys():
    be = InMemoryStateBackend()
    be.put(StateScope("op", "acc", ("a",), "w0"), 1)
    be.put(
        StateScope("op", "acc", ("a",), "w1"), 1
    )  # same key, new namespace -> +1 entry, same key
    be.put(StateScope("op", "acc", ("b",), "w0"), 1)
    assert be.sizes() == {("op", "acc"): (3, 2)}  # 3 entries, 2 distinct keys
    be.put(StateScope("op", "acc", ("a",), "w0"), 99)  # update existing slot -> counts unchanged
    assert be.sizes() == {("op", "acc"): (3, 2)}
    be.clear(StateScope("op", "acc", ("a",), "w0"))
    be.clear(StateScope("op", "acc", ("a",), "w1"))  # key 'a' now fully gone
    assert be.sizes() == {("op", "acc"): (1, 1)}


def test_reduce_all_matches_per_key_fold_and_tracks_sizes():
    # KeyedCount's hot path is the bulk fold; it must land exactly the state the per-key
    # reducing_state().add() loop it replaced does, and keep sizes() (state.entries/keys) correct.
    from nautilus.core.operator import OperatorContext
    from nautilus.state import KeyContext

    def add(a: object, b: object) -> object:
        return a + b  # type: ignore[operator]

    per_key = InMemoryStateBackend()
    ctx = OperatorContext("op", state_backend=per_key)
    for key, count in [("a", 2), ("b", 1), ("a", 3), ("b", 4)]:  # the loop reduce_all replaces
        ctx.reducing_state("count", KeyContext((key,)), add).add(count)

    bulk = InMemoryStateBackend()
    bulk_ctx = OperatorContext("op", state_backend=bulk)
    bulk_ctx.reduce_all("count", [(("a",), 2), (("b",), 1)], add)  # first fold: writes each key
    bulk_ctx.reduce_all("count", [(("a",), 3), (("b",), 4)], add)  # second fold: reduces existing

    folded = {kctx.key: value for kctx, value in bulk_ctx.entries("count")}
    assert folded == {("a",): 5, ("b",): 5}
    assert folded == {
        kctx.key: value for kctx, value in ctx.entries("count")
    }  # identical to per-key
    assert bulk.sizes() == per_key.sizes() == {("op", "count"): (2, 2)}  # sizes bookkeeping intact
