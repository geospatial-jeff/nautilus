"""Tier-1 characterization tests pinning the telemetry report/recorder surface for the Rust port.

These assert the *observable* contract a faithful reimplementation must reproduce: label
normalization, zero-division-safe derived ratios, error/traceback placement, cross-snapshot merge
arithmetic (gauge and histogram), byte-stable JSON, and event-field validation. Every golden here was
read off the current Python implementation, not hand-computed — a rewrite that passes this file
reproduces the same numbers and strings the current engine emits.
"""

import json

import pytest

import nautilus
from nautilus.core.time import TestClock
from nautilus.telemetry.model import (
    EventRecord,
    HistogramData,
    InstanceSnapshot,
    make_labels,
)
from nautilus.telemetry.recorder import InstanceRecorder, TelemetryConfig
from nautilus.telemetry.report import RunMeta, build_report


def _meta(run_id="run-0", wall=100):
    return RunMeta(
        run_id=run_id,
        started_at_micros=0,
        ended_at_micros=wall,
        wall_micros=wall,
        clock_kind="TestClock",
        nautilus_version=nautilus.__version__,
        python_version="3.12",
        config_digest="deadbeef",
        capacity=16,
    )


def _recorder(**kw):
    cfg = kw.pop("config", TelemetryConfig(clock=TestClock(0)))
    return InstanceRecorder(
        operator_id=kw.pop("operator_id", "op0"),
        op_class=kw.pop("op_class", "Tokenize"),
        kind=kw.pop("kind", "one_input"),
        config=cfg,
        **kw,
    )


def _two_op_snapshots():
    src = _recorder(operator_id="op0", op_class="Tokenize")
    src.incr("operator.rows_in", 10, operator_id="op0", subtask_index=0)
    src.incr("operator.rows_out", 25, operator_id="op0", subtask_index=0)
    src.incr("runtime.step_micros", 999, operator_id="op0", subtask_index=0)
    snk = _recorder(operator_id="op1", op_class="KeyedCount")
    snk.incr("operator.rows_in", 25, operator_id="op1", subtask_index=0)
    snk.incr("operator.rows_out", 5, operator_id="op1", subtask_index=0)
    return [src.snapshot(), snk.snapshot()]


# --- (a) make_labels normalization -------------------------------------------------------------


def test_make_labels_sorts_by_key_and_stringifies_values():
    # Values are str()-ified (int 2 -> '2', True -> 'True', None -> 'None') and pairs sort by key.
    assert make_labels({"b": 2, "a": True, "c": None}) == (
        ("a", "True"),
        ("b", "2"),
        ("c", "None"),
    )


def test_make_labels_sort_is_by_key_not_insertion_order():
    assert make_labels({"z": 1, "a": "x", "m": False}) == (
        ("a", "x"),
        ("m", "False"),
        ("z", "1"),
    )


# --- (b) derived ratios never divide by zero ---------------------------------------------------


def test_zero_wall_and_zero_busy_ratios_are_zero_not_nan():
    # wall_micros == 0 and busy == 0: every derived ratio returns a plain 0.0 (never NaN/inf/raise).
    src = _recorder(operator_id="op0")
    src.incr("operator.rows_in", 10, operator_id="op0", subtask_index=0)
    src.incr("operator.rows_out", 0, operator_id="op0", subtask_index=0)
    report = build_report([src.snapshot()], meta=_meta(wall=0))

    assert report.throughput_rows_per_sec() == 0.0
    assert report.by_occupancy() == [("op0", 0.0)]
    assert report.by_rows_per_sec() == [("op0", 0.0)]


# --- (c) errors: sorted, traceback stored once, stripped from events, optionals -> None --------


def _error_event(seq, at_micros, **fields):
    return EventRecord(seq, at_micros, "op0", "operator.error", tuple(sorted(fields.items())))


def _errors_report():
    # Emit "late" (at 500) before "early" (at 100) so build_report must re-sort by at_micros.
    late = _error_event(
        0,
        500,
        operator_id="op0",
        op_class="Tokenize",
        phase="process",
        exc_type="ValueError",
        message="boom late",
        traceback="TB-LATE",
        frame_kind="data",
        input_index=0,
        batch_rows=7,
        source_location="mod:Op",
    )
    early = _error_event(
        1,
        100,
        operator_id="op0",
        op_class="Tokenize",
        phase="open",
        exc_type="RuntimeError",
        message="boom early",
        traceback="TB-EARLY",
    )
    snap = InstanceSnapshot("op0", "Tokenize", "one_input", 0, "local", events=(late, early))
    return build_report([snap], meta=_meta())


def test_errors_are_sorted_by_at_micros():
    report = _errors_report()
    assert [(e.at_micros, e.exc_type) for e in report.errors] == [
        (100, "RuntimeError"),
        (500, "ValueError"),
    ]


def test_error_missing_optionals_become_none():
    report = _errors_report()
    early = report.errors[0]  # the RuntimeError, which omitted the optional fields
    assert early.frame_kind is None
    assert early.input_index is None
    assert early.batch_rows is None
    assert early.source_location is None
    # A present optional is carried through with its typed value.
    late = report.errors[1]
    assert late.frame_kind == "data"
    assert late.input_index == 0
    assert late.batch_rows == 7
    assert late.source_location == "mod:Op"


def test_traceback_kept_in_errors_but_stripped_from_events():
    doc = _errors_report().to_dict()
    # Full traceback lives once, in errors[].
    assert [e["traceback"] for e in doc["errors"]] == ["TB-EARLY", "TB-LATE"]
    # ...and is stripped from every matching events[] entry so it is not serialized twice.
    for ev in doc["events"]:
        assert ev["name"] == "operator.error"
        assert "traceback" not in ev["fields"]


# --- (d) histogram cross-snapshot merge --------------------------------------------------------


def test_histogram_merge_sums_buckets_extends_min_max():
    lbls = make_labels({"operator_id": "op0", "op_class": "X", "subtask_index": 0})
    key = ("operator.process_micros", lbls)
    # Same key in two snapshots. Second has min=None on purpose to pin the None-side merge.
    h1 = HistogramData(boundaries=(1, 2, 4), buckets=(1, 2, 0, 1), count=4, sum=10, min=0, max=9)
    h2 = HistogramData(
        boundaries=(1, 2, 4), buckets=(0, 1, 3, 2), count=6, sum=20, min=None, max=15
    )
    a = InstanceSnapshot("op0", "X", "one_input", 0, "local", histograms={key: h1})
    b = InstanceSnapshot("op0", "X", "one_input", 0, "local", histograms={key: h2})

    hp = build_report([a, b], meta=_meta()).operators[0].histograms[0]
    assert hp.buckets == (1, 3, 3, 3)  # elementwise sum
    assert hp.count == 10  # 4 + 6
    assert hp.sum == 30  # 10 + 20
    assert hp.min == 0  # min(0, None) collapses to the non-None side
    assert hp.max == 15  # max(9, 15)


# --- (e) gauge merge across snapshots ----------------------------------------------------------


def test_gauge_merge_takes_second_last_widest_min_max():
    # (last=5,min=1,max=9) merged with (last=3,min=2,max=7) -> (last=3,min=1,max=9):
    # last comes from the second snapshot; min/max widen across both.
    key = ("eos.expected", make_labels({"operator_id": "op0"}))
    a = InstanceSnapshot("op0", "X", "one_input", 0, "local", gauges={key: (5.0, 1.0, 9.0)})
    b = InstanceSnapshot("op0", "X", "one_input", 0, "local", gauges={key: (3.0, 2.0, 7.0)})

    gp = build_report([a, b], meta=_meta()).operators[0].gauges[0]
    assert (gp.last, gp.min, gp.max) == (3.0, 1.0, 9.0)


# --- (f) JSON is byte-stable and self-sorting --------------------------------------------------


def test_report_json_is_byte_identical_across_identical_builds():
    s1 = build_report(_two_op_snapshots(), meta=_meta()).to_json()
    s2 = build_report(_two_op_snapshots(), meta=_meta()).to_json()
    assert s1 == s2


def test_report_json_keys_are_already_sorted():
    s = build_report(_two_op_snapshots(), meta=_meta()).to_json()
    # to_json already emits sort_keys=True, so a re-dump with sort_keys is a no-op.
    assert json.dumps(json.loads(s), sort_keys=True) == s


# --- (g) event field validation ----------------------------------------------------------------


def test_event_unknown_field_raises_keyerror_naming_it():
    rec = _recorder()
    with pytest.raises(KeyError, match=r"got fields not in its EventSpec: \['bogus_field'\]"):
        rec.event(
            "operator.lifecycle.open",
            operator_id="op0",
            op_class="T",
            source_location="m:C",
            num_inputs=1,
            bogus_field=1,
        )


def test_event_validate_false_accepts_unknown_field():
    rec = _recorder(config=TelemetryConfig(clock=TestClock(0), validate=False))
    rec.event(
        "operator.lifecycle.open",
        operator_id="op0",
        op_class="T",
        source_location="m:C",
        num_inputs=1,
        bogus_field=1,
    )
    snap = rec.snapshot()
    assert len(snap.events) == 1
    assert ("bogus_field", 1) in snap.events[0].fields


# --- (h) deepest_queue -------------------------------------------------------------------------


def test_deepest_queue_picks_max_depth_edge():
    rec = _recorder(operator_id="op0", op_class="Src")
    rec.set_gauge(
        "edge.queue_depth", 3, operator_id="op0", edge_src="op0", edge_dst="op1", channel_index=0
    )
    rec.set_gauge(
        "edge.queue_depth", 7, operator_id="op0", edge_src="op0", edge_dst="op2", channel_index=0
    )
    report = build_report([rec.snapshot()], meta=_meta())
    assert report.summary.deepest_queue == ("op0->op2", 7)


def test_deepest_queue_is_none_without_edges():
    rec = _recorder(operator_id="op0", op_class="Src")
    rec.incr("operator.rows_in", 1, operator_id="op0", subtask_index=0)
    report = build_report([rec.snapshot()], meta=_meta())
    assert report.summary.deepest_queue is None
