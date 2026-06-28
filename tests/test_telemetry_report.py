"""build_report aggregates snapshots into a versioned report; JSON is well-formed; the structural
digest is stable and ignores timing."""

import json

import pytest

import nautilus
from nautilus.core.time import TestClock
from nautilus.telemetry.recorder import InstanceRecorder, TelemetryConfig
from nautilus.telemetry.report import REPORT_SCHEMA_VERSION, RunMeta, build_report


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


def _two_op_snapshots(wall_in_step=999):
    src = InstanceRecorder(
        operator_id="op0",
        op_class="Tokenize",
        kind="one_input",
        config=TelemetryConfig(clock=TestClock(0)),
    )
    src.incr("operator.rows_in", 10, operator_id="op0", subtask_index=0)
    src.incr("operator.rows_out", 25, operator_id="op0", subtask_index=0)
    src.incr("runtime.step_micros", wall_in_step, operator_id="op0", subtask_index=0)

    snk = InstanceRecorder(
        operator_id="op1",
        op_class="KeyedCount",
        kind="one_input",
        config=TelemetryConfig(clock=TestClock(0)),
    )
    snk.incr("operator.rows_in", 25, operator_id="op1", subtask_index=0)
    snk.incr("operator.rows_out", 5, operator_id="op1", subtask_index=0)
    snk.set_gauge("watermark.final_micros", 2**62, operator_id="op1")
    return [src.snapshot(), snk.snapshot()]


def test_build_report_and_json():
    report = build_report(_two_op_snapshots(), meta=_meta())
    assert report.schema_version == REPORT_SCHEMA_VERSION == 3
    assert report.summary.total_rows_in == 35
    assert report.summary.total_rows_out == 30
    doc = json.loads(report.to_json())
    assert doc["schema_version"] == 3
    assert doc["catalog_version"] == 1
    assert doc["events_dropped"] == 0
    assert {o["operator_id"] for o in doc["operators"]} == {"op0", "op1"}
    # the report embeds the catalog slice for metrics that appear
    metric_names = {m["name"] for m in doc["catalog"]["metrics"]}
    assert "operator.rows_in" in metric_names


def test_structural_digest_ignores_timing():
    # Same structural facts, different wall-clock step time -> identical digest.
    a = build_report(_two_op_snapshots(wall_in_step=10), meta=_meta(run_id="a", wall=10))
    b = build_report(_two_op_snapshots(wall_in_step=99999), meta=_meta(run_id="b", wall=99999))
    assert a.structural_digest() == b.structural_digest()


def test_structural_digest_changes_with_structure():
    base = build_report(_two_op_snapshots(), meta=_meta())
    other = InstanceRecorder(
        operator_id="op0",
        op_class="Tokenize",
        kind="one_input",
        config=TelemetryConfig(clock=TestClock(0)),
    )
    other.incr("operator.rows_in", 11, operator_id="op0", subtask_index=0)  # different count
    changed = build_report([other.snapshot()], meta=_meta())
    assert base.structural_digest() != changed.structural_digest()


def test_events_dropped_is_surfaced_in_report():
    rec = InstanceRecorder(
        operator_id="op0",
        op_class="Tokenize",
        kind="one_input",
        config=TelemetryConfig(clock=TestClock(0), event_log_capacity=2),
    )
    for _ in range(5):  # 5 events into a 2-slot log -> 3 dropped
        rec.event(
            "operator.lifecycle.open",
            operator_id="op0",
            op_class="Tokenize",
            source_location="m:C",
            num_inputs=1,
        )
    snap = rec.snapshot()
    assert snap.events_dropped == 3
    report = build_report([snap], meta=_meta())
    assert report.events_dropped == 3
    assert json.loads(report.to_json())["events_dropped"] == 3


def test_ranked_by_rejects_unknown_axis():
    report = build_report(_two_op_snapshots(), meta=_meta())
    assert report.by_self_time()  # a valid axis works
    with pytest.raises(KeyError):
        report.ranked_by("busy_micros")  # not a real OperatorSummary field
