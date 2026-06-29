"""The telemetry BOUNDARY layer: aggregate per-instance snapshots into a versioned run report,
serialize it for agents, and generate the self-describing reference.

This package reads across instances and is forbidden from the per-record data path by an import-linter
contract — the data path only ever writes to a recorder; assembling a report is a job-boundary concern.
"""

from nautilus.telemetry.report.report import (
    REPORT_SCHEMA_VERSION,
    CounterPoint,
    Edge,
    EdgeStats,
    ErrorRecord,
    GaugePoint,
    HistogramPoint,
    OperatorNode,
    OperatorStats,
    OperatorSummary,
    RunMeta,
    RunReport,
    RunSummary,
    Topology,
    build_report,
)
from nautilus.telemetry.report.sink import BufferSink, NullSink, Sink

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "CounterPoint",
    "Edge",
    "EdgeStats",
    "ErrorRecord",
    "GaugePoint",
    "HistogramPoint",
    "OperatorNode",
    "OperatorStats",
    "OperatorSummary",
    "RunMeta",
    "RunReport",
    "RunSummary",
    "Topology",
    "build_report",
    "Sink",
    "NullSink",
    "BufferSink",
]
