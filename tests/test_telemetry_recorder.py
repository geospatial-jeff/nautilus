"""Recorder + model: instruments record correctly, tiers gate cost, snapshots are picklable."""

import pickle

from nautilus.core.time import TestClock
from nautilus.telemetry.catalog import Tier
from nautilus.telemetry.model import Histogram
from nautilus.telemetry.recorder import (
    NOOP_COUNTER,
    NULL_RECORDER,
    InstanceRecorder,
    TelemetryConfig,
    make_recorder,
)


def _recorder(tier=Tier.COUNTERS):
    return InstanceRecorder(
        operator_id="op0",
        op_class="Tokenize",
        kind="one_input",
        config=TelemetryConfig(tier=tier, clock=TestClock(0)),
    )


def test_counters_and_histograms_record():
    r = _recorder()
    r.incr("operator.rows_in", 5, operator_id="op0", subtask_index=0)
    r.incr("operator.rows_in", 3, operator_id="op0", subtask_index=0)
    r.observe("operator.batch_rows", 4, operator_id="op0", subtask_index=0)
    snap = r.snapshot()
    assert (
        snap.counters[("operator.rows_in", (("operator_id", "op0"), ("subtask_index", "0")))] == 8
    )
    ((_, hd),) = [(k, v) for k, v in snap.histograms.items()]
    assert hd.count == 1 and hd.sum == 4


def test_histogram_bucketing_is_exact():
    h = Histogram((1, 2, 4, 8))
    for v in (0, 1, 2, 3, 100):
        h.observe(v)
    # boundaries are upper-inclusive: 0->b0, 1->b0(<=1), 2->b1, 3->b2, 100->overflow
    assert h.buckets == [2, 1, 1, 0, 1]
    assert h.count == 5 and h.sum == 106 and h.min == 0 and h.max == 100


def test_tier_gating_makes_disabled_metrics_noops():
    r = _recorder(tier=Tier.COUNTERS)
    # bytes_* is FULL-only; at COUNTERS it resolves to the shared no-op and records nothing.
    c = r.counter("operator.bytes_in", operator_id="op0", subtask_index=0)
    assert c is NOOP_COUNTER
    c.add(123)
    assert r.snapshot().counters == {}


def test_full_tier_enables_byte_metrics():
    r = _recorder(tier=Tier.FULL)
    r.incr("operator.bytes_in", 999, operator_id="op0", subtask_index=0)
    assert any(n == "operator.bytes_in" for (n, _l) in r.snapshot().counters)


def test_null_recorder_is_zero_cost_and_make_recorder_off():
    assert NULL_RECORDER.counter("anything") is NOOP_COUNTER
    NULL_RECORDER.incr("operator.rows_in", 10)  # no error, no effect
    assert NULL_RECORDER.snapshot().counters == {}
    off = make_recorder(
        operator_id="x", op_class="X", kind="one_input", config=TelemetryConfig(tier=Tier.OFF)
    )
    assert off is NULL_RECORDER


def test_events_record_and_validate_fields():
    r = _recorder(tier=Tier.COUNTERS)
    r.event(
        "operator.lifecycle.open",
        operator_id="op0",
        op_class="Tokenize",
        source_location="nautilus.operators:Tokenize",
        num_inputs=1,
    )
    snap = r.snapshot()
    assert len(snap.events) == 1 and snap.events[0].name == "operator.lifecycle.open"


def test_snapshot_pickle_round_trips():
    r = _recorder()
    r.incr("operator.rows_in", 7, operator_id="op0", subtask_index=0)
    r.set_gauge("eos.expected", 1, operator_id="op0")
    r.observe(
        "operator.process_micros", 42, operator_id="op0", op_class="Tokenize", subtask_index=0
    )
    r.event("eos.forwarded", operator_id="op0", wall_micros=5)  # gated off at COUNTERS -> absent
    snap = r.snapshot()
    restored = pickle.loads(pickle.dumps(snap))
    assert restored == snap


def test_empty_histogram_has_null_min_max():
    r = _recorder()
    h = r.histogram("operator.process_micros", operator_id="op0", op_class="X", subtask_index=0)
    data = h.data()
    assert data.count == 0 and data.min is None and data.max is None
