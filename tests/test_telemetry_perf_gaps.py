"""Telemetry that performance analysis needs: occupancy is countable (on_watermark time is in
step_micros), state growth is visible (state.entries/keys), and keyed-shuffle cost is attributed
(partition.route_micros). These close gaps that previously left the engine's hottest costs invisible.
"""

from nautilus.core.records import EOS_FRAME
from nautilus.core.time import TestClock
from nautilus.operators import InMemorySource, KeyedTumblingSum
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.parallel import graph_from_pipeline
from nautilus.runtime.run import run_plan
from nautilus.state import InMemoryStateBackend, StateScope
from nautilus.telemetry.catalog import Tier
from nautilus.telemetry.recorder import TelemetryConfig
from nautilus.testing import data, wm
from nautilus.windows import TumblingEventTimeWindows


def _windowed():
    # keys a,a,b in window [0,10); wm(10) fires it; a,b in [10,20); wm(20) fires it.
    frames = [
        data(key=["a", "a", "b"], val=[1, 2, 5], ts=[1, 5, 7]),
        wm(10),
        data(key=["a", "b"], val=[10, 3], ts=[12, 14]),
        wm(20),
        EOS_FRAME,
    ]
    op = KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))
    return InMemorySource(frames), [op]


def _gauge_max(op, name):
    return max((g.max for g in op.gauges if g.name == name), default=None)


def _counter(op, name):
    return sum(p.value for p in op.counters if p.name == name)


def _hist_sum(op, name):
    return sum(h.sum for h in op.histograms if h.name == name)


async def test_step_micros_includes_on_watermark_time():
    src, ops = _windowed()
    rep = (await run_local_chain(src, ops, clock=TestClock())).telemetry
    op0 = rep.operator("op0")
    step = _counter(op0, "runtime.step_micros")
    proc = _hist_sum(op0, "operator.process_micros")
    wmk = _hist_sum(op0, "operator.on_watermark_micros")
    # busy/self-time must cover both synchronous critical sections, as the catalog meaning states.
    assert step == proc + wmk
    # this operator does real work in on_watermark (window flush), so the two differ.
    assert wmk > 0


async def test_state_entries_and_keys_track_open_windows():
    src, ops = _windowed()
    rep = (await run_local_chain(src, ops, clock=TestClock())).telemetry
    op0 = rep.operator("op0")
    # At the wm(10) fire, both a and b have an open window in [0,10): 2 entries across 2 distinct keys.
    assert _gauge_max(op0, "state.entries") == 2
    assert _gauge_max(op0, "state.keys") == 2


async def test_state_gauges_absent_when_telemetry_off():
    src, ops = _windowed()
    rep = (await run_local_chain(src, ops, telemetry=TelemetryConfig(tier=Tier.OFF))).telemetry
    assert rep.operator("op0") is None  # OFF records nothing at all


async def test_partition_route_micros_attributes_the_keyed_shuffle():
    src, ops = _windowed()
    graph = graph_from_pipeline(src, ops, 3)  # P=3 -> source->op0 is a keyed shuffle that routes
    rep = (await run_plan(graph, telemetry=TelemetryConfig(tier=Tier.COUNTERS))).telemetry
    src_op = next(o for o in rep.operators if o.operator_id == "source")
    routed = [h for h in src_op.histograms if h.name == "partition.route_micros"]
    assert routed, "the shuffling source must record partition.route_micros"
    assert all(h.labels for h in routed)  # labeled by (operator_id, edge_dst)
    assert sum(h.count for h in routed) > 0


def test_backend_sizes_count_entries_and_distinct_keys():
    be = InMemoryStateBackend()
    be.put(StateScope("op", "acc", ("a",), "w0"), 1)
    be.put(StateScope("op", "acc", ("a",), "w1"), 1)  # same key, new namespace -> +1 entry, same key
    be.put(StateScope("op", "acc", ("b",), "w0"), 1)
    assert be.sizes() == {("op", "acc"): (3, 2)}  # 3 entries, 2 distinct keys
    be.put(StateScope("op", "acc", ("a",), "w0"), 99)  # update existing slot -> counts unchanged
    assert be.sizes() == {("op", "acc"): (3, 2)}
    be.clear(StateScope("op", "acc", ("a",), "w0"))
    be.clear(StateScope("op", "acc", ("a",), "w1"))  # key 'a' now fully gone
    assert be.sizes() == {("op", "acc"): (1, 1)}
