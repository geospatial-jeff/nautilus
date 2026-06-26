"""Deterministic serialization of a :class:`RunReport` to dict/JSON.

The JSON is the complete, machine-readable surface (sorted keys, stable shapes). The report embeds the
catalog *slice* for the metrics/events actually present in the run plus a ``catalog_version``, so an
agent reading one report in isolation is self-contained while the payload stays lean. The token-budgeted
markdown digest is added in S3.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nautilus.telemetry.catalog import (
    EVENT_SPECS,
    METRIC_SPECS,
    EventSpec,
    MetricSpec,
)
from nautilus.telemetry.model import Labels

if TYPE_CHECKING:
    from nautilus.telemetry.report.report import RunReport

CATALOG_VERSION = 1


def _labels(labels: Labels) -> dict[str, str]:
    return {k: v for k, v in labels}


def _metric_spec_dict(spec: MetricSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "kind": str(spec.kind),
        "unit": spec.unit,
        "labels": list(spec.labels),
        "reduction": str(spec.reduction),
        "meaning": spec.meaning,
        "relates_to": list(spec.relates_to),
        "derivation": spec.derivation,
        "since_stage": spec.since_stage,
        "stability": str(spec.stability),
        "deterministic": spec.deterministic,
        "min_tier": int(spec.min_tier),
        "boundaries": list(spec.boundaries),
    }


def _event_spec_dict(spec: EventSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "fields": list(spec.fields),
        "meaning": spec.meaning,
        "since_stage": spec.since_stage,
        "stability": str(spec.stability),
        "min_tier": int(spec.min_tier),
    }


def report_to_dict(report: RunReport) -> dict[str, object]:
    present_metrics: set[str] = set()
    present_events: set[str] = set()

    operators = []
    for o in report.operators:
        for p in o.counters:
            present_metrics.add(p.name)
        for g in o.gauges:
            present_metrics.add(g.name)
        for h in o.histograms:
            present_metrics.add(h.name)
        operators.append(
            {
                "operator_id": o.operator_id,
                "op_class": o.op_class,
                "kind": o.kind,
                "subtask_index": o.subtask_index,
                "node": o.node,
                "error_count": o.error_count,
                "counters": [
                    {"name": p.name, "labels": _labels(p.labels), "value": p.value}
                    for p in o.counters
                ],
                "gauges": [
                    {
                        "name": p.name,
                        "labels": _labels(p.labels),
                        "last": p.last,
                        "min": p.min,
                        "max": p.max,
                    }
                    for p in o.gauges
                ],
                "histograms": [
                    {
                        "name": p.name,
                        "labels": _labels(p.labels),
                        "boundaries": list(p.boundaries),
                        "buckets": list(p.buckets),
                        "count": p.count,
                        "sum": p.sum,
                        "min": p.min,
                        "max": p.max,
                    }
                    for p in o.histograms
                ],
            }
        )

    events = []
    for ev in report.events:
        present_events.add(ev.name)
        events.append(
            {
                "seq": ev.seq,
                "at_micros": ev.at_micros,
                "operator_id": ev.operator_id,
                "name": ev.name,
                "fields": {k: v for k, v in ev.fields},
            }
        )

    meta = report.meta
    out: dict[str, object] = {
        "schema_version": report.schema_version,
        "nautilus_version": report.nautilus_version,
        "run_id": report.run_id,
        "catalog_version": CATALOG_VERSION,
        "meta": {
            "run_id": meta.run_id,
            "started_at_micros": meta.started_at_micros,
            "ended_at_micros": meta.ended_at_micros,
            "wall_micros": meta.wall_micros,
            "clock_kind": meta.clock_kind,
            "nautilus_version": meta.nautilus_version,
            "python_version": meta.python_version,
            "config_digest": meta.config_digest,
            "capacity": meta.capacity,
        },
        "topology": _topology_dict(report),
        "operators": operators,
        "edges": [
            {
                "src_operator_id": e.src_operator_id,
                "dst_operator_id": e.dst_operator_id,
                "channel_index": e.channel_index,
                "capacity": e.capacity,
                "frames_sent_total": e.frames_sent_total,
                "batches_sent_total": e.batches_sent_total,
                "rows_sent_total": e.rows_sent_total,
                "send_wait_micros_total": e.send_wait_micros_total,
                "max_queue_depth": e.max_queue_depth,
            }
            for e in report.edges
        ],
        "events": events,
        "events_dropped": report.events_dropped,
        "errors": [
            {
                "operator_id": e.operator_id,
                "op_class": e.op_class,
                "phase": e.phase,
                "exc_type": e.exc_type,
                "message": e.message,
                "traceback": e.traceback,
                "at_micros": e.at_micros,
                "frame_kind": e.frame_kind,
                "input_index": e.input_index,
                "batch_rows": e.batch_rows,
            }
            for e in report.errors
        ],
        "summary": {
            "wall_micros": report.summary.wall_micros,
            "total_rows_in": report.summary.total_rows_in,
            "total_rows_out": report.summary.total_rows_out,
            "total_errors": report.summary.total_errors,
            "per_operator": [
                {
                    "operator_id": s.operator_id,
                    "busy_micros_total": s.busy_micros_total,
                    "send_wait_micros_total": s.send_wait_micros_total,
                    "rows_out_total": s.rows_out_total,
                    "error_count": s.error_count,
                }
                for s in report.summary.per_operator
            ],
            "deepest_queue": (
                list(report.summary.deepest_queue) if report.summary.deepest_queue else None
            ),
        },
        "catalog": {
            "metrics": [
                _metric_spec_dict(METRIC_SPECS[n])
                for n in sorted(present_metrics)
                if n in METRIC_SPECS
            ],
            "events": [
                _event_spec_dict(EVENT_SPECS[n]) for n in sorted(present_events) if n in EVENT_SPECS
            ],
        },
    }
    return out


def _topology_dict(report: RunReport) -> object:
    if report.topology is None:
        return None
    return {
        "nodes": [
            {
                "operator_id": n.operator_id,
                "op_class": n.op_class,
                "kind": n.kind,
                "subtask_index": n.subtask_index,
                "num_subtasks": n.num_subtasks,
                "source_file": n.source_file,
                "source_line": n.source_line,
            }
            for n in report.topology.nodes
        ],
        "edges": [
            {
                "src_operator_id": e.src_operator_id,
                "dst_operator_id": e.dst_operator_id,
                "channel_index": e.channel_index,
                "partitioner": e.partitioner,
                "capacity": e.capacity,
            }
            for e in report.topology.edges
        ],
    }


def report_to_json(report: RunReport, *, indent: int | None = None) -> str:
    return json.dumps(report_to_dict(report), sort_keys=True, indent=indent)


def report_to_markdown(report: RunReport, *, token_budget: int = 4000) -> str:
    """A compact, token-bounded digest for an agent to read. It surfaces raw facts (every number here
    also appears in ``to_json()``) under axis-explicit rankings; it draws no conclusions. RunSummary
    and errors are always shown in full; the per-operator table truncates to fit the budget."""
    max_chars = token_budget * 4
    m, s = report.meta, report.summary
    op_class = {o.operator_id: o.op_class for o in report.operators}

    head = [
        f"# nautilus run {report.run_id}",
        f"schema {report.schema_version} · nautilus {report.nautilus_version} · "
        f"wall {m.wall_micros}us · clock {m.clock_kind}",
        "",
        "## summary",
        f"rows_in={s.total_rows_in} rows_out={s.total_rows_out} errors={s.total_errors}",
    ]
    if s.deepest_queue:
        edge, depth = s.deepest_queue
        head.append(f"deepest_queue {edge} depth={depth}")
    if report.events_dropped:
        head.append(f"events_dropped={report.events_dropped} (event log truncated)")

    errors: list[str] = []
    if report.errors:
        errors.append("")
        errors.append("## errors")
        for e in report.errors:  # errors are never dropped
            errors.append(f"- {e.operator_id} {e.phase} {e.exc_type}: {e.message}")

    table: list[str] = [
        "",
        "## operators — by self-time (runtime.step_micros)",
        "| operator | class | rows_out | busy_us | send_wait_us | errors |",
        "|---|---|--:|--:|--:|--:|",
    ]
    fixed_chars = len("\n".join(head + errors)) + len("\n".join(table)) + 80
    truncated = False
    for stat in report.by_self_time():
        row = (
            f"| {stat.operator_id} | {op_class.get(stat.operator_id, '')} | "
            f"{stat.rows_out_total} | {stat.busy_micros_total} | "
            f"{stat.send_wait_micros_total} | {stat.error_count} |"
        )
        if fixed_chars + len("\n".join(table)) + len(row) > max_chars:
            truncated = True
            break
        table.append(row)
    if truncated:
        table.append("… more operators omitted; see to_json()")

    order = ", ".join(stat.operator_id for stat in report.by_send_wait())
    tail = ["", f"rankings · by send-wait: {order}", "", "full data: result.telemetry.to_json()"]
    return "\n".join(head + errors + table + tail)
