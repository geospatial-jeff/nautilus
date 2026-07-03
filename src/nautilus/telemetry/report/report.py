"""The RunReport tree and ``build_report`` — aggregate per-instance snapshots into one versioned report.

The report is a tree of frozen dataclasses. The schema is treated as a public API from day one
(:data:`REPORT_SCHEMA_VERSION`) so a future benchmark/diff tool can consume it. Only *raw* facts are
stored — ratios like rows/sec and busy % are computed on demand by the query helpers, never persisted,
so warmup differences cannot create false regressions in a diff. (Selectivity is not computed here at
all; it is emitted only as a derivation hint for consumers — see the markdown digest.)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from nautilus.telemetry.catalog import STRUCTURAL_METRICS
from nautilus.telemetry.model import EventRecord, HistogramData, InstanceSnapshot, Labels

REPORT_SCHEMA_VERSION = (
    3  # v3: summary.per_operator carries subtask_index/node (v2: events_dropped)
)

#: The numeric :class:`OperatorSummary` fields that :meth:`RunReport.ranked_by` may sort on.
RANKABLE_FIELDS: frozenset[str] = frozenset(
    {"busy_micros_total", "send_wait_micros_total", "rows_out_total", "error_count"}
)


# --- Topology ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperatorNode:
    operator_id: str
    op_class: str
    kind: str
    subtask_index: int = 0
    num_subtasks: int = 1
    source_file: str | None = None
    source_line: int | None = None


@dataclass(frozen=True, slots=True)
class Edge:
    src_operator_id: str
    dst_operator_id: str
    channel_index: int
    partitioner: str
    capacity: int


@dataclass(frozen=True, slots=True)
class Topology:
    nodes: tuple[OperatorNode, ...]
    edges: tuple[Edge, ...]


# --- Per-operator series -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CounterPoint:
    name: str
    labels: Labels
    value: int


@dataclass(frozen=True, slots=True)
class GaugePoint:
    name: str
    labels: Labels
    last: float
    min: float
    max: float


@dataclass(frozen=True, slots=True)
class HistogramPoint:
    name: str
    labels: Labels
    boundaries: tuple[int, ...]
    buckets: tuple[int, ...]
    count: int
    sum: int
    min: int | None
    max: int | None


@dataclass(frozen=True, slots=True)
class OperatorStats:
    operator_id: str
    op_class: str
    kind: str
    subtask_index: int
    node: str
    counters: tuple[CounterPoint, ...]
    gauges: tuple[GaugePoint, ...]
    histograms: tuple[HistogramPoint, ...]
    error_count: int


@dataclass(frozen=True, slots=True)
class EdgeStats:
    src_operator_id: str
    dst_operator_id: str
    channel_index: int
    capacity: int
    frames_sent_total: int
    batches_sent_total: int
    rows_sent_total: int
    send_wait_micros_total: int
    max_queue_depth: int


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    operator_id: str
    op_class: str
    phase: str
    exc_type: str
    message: str
    traceback: str
    at_micros: int
    frame_kind: str | None
    input_index: int | None
    batch_rows: int | None
    source_location: str | None


@dataclass(frozen=True, slots=True)
class RunMeta:
    run_id: str
    started_at_micros: int
    ended_at_micros: int
    wall_micros: int
    clock_kind: str
    nautilus_version: str
    python_version: str
    config_digest: str
    capacity: int


@dataclass(frozen=True, slots=True)
class OperatorSummary:
    operator_id: str
    subtask_index: int
    node: str
    busy_micros_total: int
    send_wait_micros_total: int
    rows_out_total: int
    error_count: int


@dataclass(frozen=True, slots=True)
class RunSummary:
    wall_micros: int
    total_rows_in: int
    total_rows_out: int
    total_errors: int
    per_operator: tuple[OperatorSummary, ...]
    deepest_queue: tuple[str, int] | None


@dataclass(frozen=True, slots=True)
class RunReport:
    schema_version: int
    nautilus_version: str
    run_id: str
    meta: RunMeta
    topology: Topology | None
    operators: tuple[OperatorStats, ...]
    edges: tuple[EdgeStats, ...]
    events: tuple[EventRecord, ...]
    #: Events discarded because a recorder's bounded event log overflowed (0 = the log is complete).
    events_dropped: int
    errors: tuple[ErrorRecord, ...]
    summary: RunSummary

    def to_dict(self) -> dict[str, object]:
        from nautilus.telemetry.report.serialize import report_to_dict

        return report_to_dict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        from nautilus.telemetry.report.serialize import report_to_json

        return report_to_json(self, indent=indent)

    def structural_digest(self) -> str:
        return structural_digest(self)

    def to_markdown(self, *, token_budget: int = 4000) -> str:
        from nautilus.telemetry.report.serialize import report_to_markdown

        return report_to_markdown(self, token_budget=token_budget)

    # --- Query helpers: sort / filter / project. They surface facts; they never diagnose. --------

    def operator(self, operator_id: str) -> OperatorStats | None:
        return next((o for o in self.operators if o.operator_id == operator_id), None)

    def edge(self, src: str, dst: str, channel_index: int = 0) -> EdgeStats | None:
        return next(
            (
                e
                for e in self.edges
                if e.src_operator_id == src
                and e.dst_operator_id == dst
                and e.channel_index == channel_index
            ),
            None,
        )

    def ranked_by(self, field: str) -> list[OperatorSummary]:
        """Operators sorted descending by one named raw field of :class:`OperatorSummary`."""
        if field not in RANKABLE_FIELDS:
            raise KeyError(f"unknown rank axis {field!r}; choose one of {sorted(RANKABLE_FIELDS)}")
        return sorted(self.summary.per_operator, key=lambda s: getattr(s, field), reverse=True)

    def by_self_time(self) -> list[OperatorSummary]:
        return self.ranked_by("busy_micros_total")

    def by_send_wait(self) -> list[OperatorSummary]:
        return self.ranked_by("send_wait_micros_total")

    def throughput_rows_per_sec(self) -> float:
        """End-to-end throughput: total rows out per wall-clock second (derived on demand, never
        persisted). The headline number for comparing two runs of the same pipeline."""
        return (
            self.summary.total_rows_out / (self.meta.wall_micros / 1_000_000)
            if self.meta.wall_micros
            else 0.0
        )

    def by_occupancy(self) -> list[tuple[str, float]]:
        """Per-instance occupancy — self-time (``runtime.step_micros``) as a fraction of wall — highest
        first (derived on demand, never persisted). Returns ``(operator_id, busy_fraction)``. The wall
        not captured here appears in ``partition.route_micros``, ``edge.input_wait_micros``, or
        cross-process I/O."""
        wall = self.meta.wall_micros
        ranked = [
            (s.operator_id, (s.busy_micros_total / wall if wall else 0.0))
            for s in self.summary.per_operator
        ]
        return sorted(ranked, key=lambda kv: kv[1], reverse=True)

    def by_rows_per_sec(self) -> list[tuple[str, float]]:
        """Operators ranked by rows_out per second of self-time (a derived ratio, computed on demand
        and never persisted). Returns ``(operator_id, rows_per_sec)``."""
        ranked = [
            (
                s.operator_id,
                (
                    s.rows_out_total / (s.busy_micros_total / 1_000_000)
                    if s.busy_micros_total
                    else 0.0
                ),
            )
            for s in self.summary.per_operator
        ]
        return sorted(ranked, key=lambda kv: kv[1], reverse=True)


# --- Aggregation helpers -----------------------------------------------------------------------


def _counter_total(snapshot: InstanceSnapshot, name: str) -> int:
    return sum(v for (n, _labels), v in snapshot.counters.items() if n == name)


def _labels_get(labels: Labels, key: str) -> str | None:
    for k, v in labels:
        if k == key:
            return v
    return None


def _merge_min(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else min(a, b)


def _merge_max(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else max(a, b)


def _operator_stats(snaps: list[InstanceSnapshot]) -> OperatorStats:
    """Merge every snapshot for one (operator_id, subtask) — typically the actor's built-in recorder
    plus the operator-author ``ctx.metrics`` recorder — into one OperatorStats."""
    counters: dict[tuple[str, Labels], int] = {}
    gauges: dict[tuple[str, Labels], tuple[float, float, float]] = {}
    histograms: dict[tuple[str, Labels], HistogramData] = {}
    for s in snaps:
        for k, v in s.counters.items():
            counters[k] = counters.get(k, 0) + v
        for k, (last, mn, mx) in s.gauges.items():
            if k in gauges:
                _last, _mn, _mx = gauges[k]
                gauges[k] = (last, min(_mn, mn), max(_mx, mx))
            else:
                gauges[k] = (last, mn, mx)
        for k, h in s.histograms.items():
            if k in histograms:
                p = histograms[k]
                buckets = tuple(a + b for a, b in zip(p.buckets, h.buckets, strict=True))
                histograms[k] = HistogramData(
                    p.boundaries,
                    buckets,
                    p.count + h.count,
                    p.sum + h.sum,
                    _merge_min(p.min, h.min),
                    _merge_max(p.max, h.max),
                )
            else:
                histograms[k] = h

    head = snaps[0]
    counter_pts = tuple(
        sorted(
            (CounterPoint(n, lbls, v) for (n, lbls), v in counters.items()),
            key=lambda p: (p.name, p.labels),
        )
    )
    gauge_pts = tuple(
        sorted(
            (GaugePoint(n, lbls, last, mn, mx) for (n, lbls), (last, mn, mx) in gauges.items()),
            key=lambda p: (p.name, p.labels),
        )
    )
    hist_pts = tuple(
        sorted(
            (
                HistogramPoint(n, lbls, h.boundaries, h.buckets, h.count, h.sum, h.min, h.max)
                for (n, lbls), h in histograms.items()
            ),
            key=lambda p: (p.name, p.labels),
        )
    )
    error_count = sum(v for (n, _l), v in counters.items() if n == "operator.errors")
    return OperatorStats(
        operator_id=head.operator_id,
        op_class=head.op_class,
        kind=head.kind,
        subtask_index=head.subtask_index,
        node=head.node,
        counters=counter_pts,
        gauges=gauge_pts,
        histograms=hist_pts,
        error_count=error_count,
    )


def _build_edges(snapshots: list[InstanceSnapshot]) -> tuple[EdgeStats, ...]:
    # Aggregate producer-owned edge.* series keyed by (edge_src, edge_dst, channel_index).
    acc: dict[tuple[str, str, int], dict[str, int]] = {}
    for snap in snapshots:
        for (name, labels), value in snap.counters.items():
            if not name.startswith("edge."):
                continue
            src, dst, ch = _edge_key(labels)
            if src is None or dst is None:
                continue
            slot = acc.setdefault((src, dst, ch), {})
            slot[name] = slot.get(name, 0) + value
        for (name, labels), (last, _mn, mx) in snap.gauges.items():
            if name not in ("edge.queue_depth", "edge.queue_capacity"):
                continue
            src, dst, ch = _edge_key(labels)
            if src is None or dst is None:
                continue
            slot = acc.setdefault((src, dst, ch), {})
            if name == "edge.queue_depth":
                slot["__depth"] = max(slot.get("__depth", 0), int(mx))
            else:
                slot["__capacity"] = int(last)
    edges = [
        EdgeStats(
            src_operator_id=src,
            dst_operator_id=dst,
            channel_index=ch,
            capacity=slot.get("__capacity", 0),
            frames_sent_total=slot.get("edge.frames_sent", 0),
            batches_sent_total=slot.get("edge.batches_sent", 0),
            rows_sent_total=slot.get("edge.rows_sent", 0),
            send_wait_micros_total=slot.get("edge.send_wait_micros", 0),
            max_queue_depth=slot.get("__depth", 0),
        )
        for (src, dst, ch), slot in sorted(acc.items())
    ]
    return tuple(edges)


def _edge_key(labels: Labels) -> tuple[str | None, str | None, int]:
    src = _labels_get(labels, "edge_src")
    dst = _labels_get(labels, "edge_dst")
    ch = _labels_get(labels, "channel_index")
    return src, dst, int(ch) if ch is not None else 0


def _build_errors(snapshots: list[InstanceSnapshot]) -> tuple[ErrorRecord, ...]:
    errors: list[ErrorRecord] = []
    for snap in snapshots:
        for ev in snap.events:
            if ev.name != "operator.error":
                continue
            f = ev.as_dict()
            errors.append(
                ErrorRecord(
                    operator_id=str(f.get("operator_id", snap.operator_id)),
                    op_class=str(f.get("op_class", snap.op_class)),
                    phase=str(f.get("phase", "")),
                    exc_type=str(f.get("exc_type", "")),
                    message=str(f.get("message", "")),
                    traceback=str(f.get("traceback", "")),
                    at_micros=ev.at_micros,
                    frame_kind=_opt_str(f.get("frame_kind")),
                    input_index=_opt_int(f.get("input_index")),
                    batch_rows=_opt_int(f.get("batch_rows")),
                    source_location=_opt_str(f.get("source_location")),
                )
            )
    errors.sort(key=lambda e: e.at_micros)
    return tuple(errors)


def _opt_str(v: object) -> str | None:
    return None if v is None else str(v)


def _opt_int(v: object) -> int | None:
    return None if v is None else cast(int, v)


def build_report(
    snapshots: list[InstanceSnapshot],
    *,
    meta: RunMeta,
    topology: Topology | None = None,
) -> RunReport:
    """Aggregate per-instance snapshots into one immutable :class:`RunReport`."""
    # node joins the grouping key so each worker's hardware ("process") row stays distinct — there is
    # exactly one per worker. For a dataflow row this is a no-op: an operator instance lives
    # on exactly one node, so adding node never splits it.
    groups: dict[tuple[str, int, str], list[InstanceSnapshot]] = {}
    for s in snapshots:
        if s.operator_id:
            groups.setdefault((s.operator_id, s.subtask_index, s.node), []).append(s)
    operators = tuple(
        sorted(
            (_operator_stats(g) for g in groups.values()),
            key=lambda o: (o.operator_id, o.subtask_index, o.node),
        )
    )
    edges = _build_edges(snapshots)
    errors = _build_errors(snapshots)
    events = tuple(
        sorted(
            (ev for s in snapshots for ev in s.events),
            key=lambda e: (e.at_micros, e.operator_id, e.seq),
        )
    )
    events_dropped = sum(s.events_dropped for s in snapshots)

    total_rows_in = sum(_counter_total(s, "operator.rows_in") for s in snapshots)
    total_rows_out = sum(_counter_total(s, "operator.rows_out") for s in snapshots)
    total_errors = sum(o.error_count for o in operators)
    per_operator = tuple(
        OperatorSummary(
            operator_id=o.operator_id,
            subtask_index=o.subtask_index,
            node=o.node,
            busy_micros_total=_point_total(o.counters, "runtime.step_micros"),
            send_wait_micros_total=_point_total(o.counters, "edge.send_wait_micros"),
            rows_out_total=_point_total(o.counters, "operator.rows_out"),
            error_count=o.error_count,
        )
        # The process/hardware row is not a dataflow operator; it is surfaced separately (hardware
        # panel / report.operator("process")), not ranked among the dataflow stages.
        for o in operators
        if o.kind != "process"
    )
    deepest = max(
        ((f"{e.src_operator_id}->{e.dst_operator_id}", e.max_queue_depth) for e in edges),
        key=lambda kv: kv[1],
        default=None,
    )
    summary = RunSummary(
        wall_micros=meta.wall_micros,
        total_rows_in=total_rows_in,
        total_rows_out=total_rows_out,
        total_errors=total_errors,
        per_operator=per_operator,
        deepest_queue=deepest,
    )
    return RunReport(
        schema_version=REPORT_SCHEMA_VERSION,
        nautilus_version=meta.nautilus_version,
        run_id=meta.run_id,
        meta=meta,
        topology=topology,
        operators=operators,
        edges=edges,
        events=events,
        events_dropped=events_dropped,
        errors=errors,
        summary=summary,
    )


def _point_total(points: tuple[CounterPoint, ...], name: str) -> int:
    return sum(p.value for p in points if p.name == name)


# --- Structural digest (provably-deterministic facts only) -------------------------------------


def structural_digest(report: RunReport) -> str:
    """SHA-256 over only reproducible facts (topology + structural counts). Excludes all timing,
    queue depths, idle/active counts, event order, run_id and timestamps — so the same logical run
    yields the same digest regardless of scheduling or wall clock."""
    canonical: dict[str, object] = {
        "schema_version": report.schema_version,
        "operators": [
            {
                "operator_id": o.operator_id,
                "op_class": o.op_class,
                "kind": o.kind,
                "subtask_index": o.subtask_index,
                "counters": sorted(
                    [p.name, list(p.labels), p.value]
                    for p in o.counters
                    if p.name in STRUCTURAL_METRICS
                ),
                "gauges": sorted(
                    [p.name, list(p.labels), p.last]
                    for p in o.gauges
                    if p.name in STRUCTURAL_METRICS
                ),
            }
            # The process/hardware sampler is a non-deterministic, machine-dependent row; exclude it
            # entirely so enabling sampling never changes a run's structural identity.
            for o in report.operators
            if o.kind != "process"
        ],
    }
    if report.topology is not None:
        canonical["topology"] = {
            "nodes": sorted([n.operator_id, n.op_class, n.kind] for n in report.topology.nodes),
            "edges": sorted(
                [e.src_operator_id, e.dst_operator_id, e.channel_index]
                for e in report.topology.edges
            ),
        }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
