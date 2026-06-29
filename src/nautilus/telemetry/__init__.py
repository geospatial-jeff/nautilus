"""Nautilus telemetry — the data-path layer (model, catalog, recorder, registry).

Importable by ``nautilus.runtime``/``nautilus.core``. The boundary layer that aggregates across
instances and serializes reports lives in :mod:`nautilus.telemetry.report` and is forbidden from the
data path by an import-linter contract. This package deliberately does NOT import the report layer, so
pulling in telemetry on the hot path never drags in report assembly.
"""

from nautilus.telemetry.catalog import (
    EVENT_SPECS,
    METRIC_SPECS,
    EventSpec,
    MetricKind,
    MetricSpec,
    Owner,
    Reduction,
    Stability,
    Tier,
)
from nautilus.telemetry.recorder import (
    DEFAULT_CONFIG,
    NULL_RECORDER,
    InstanceRecorder,
    NullRecorder,
    Recorder,
    TelemetryConfig,
    make_recorder,
)
from nautilus.telemetry.registry import RecorderRegistry

__all__ = [
    "METRIC_SPECS",
    "EVENT_SPECS",
    "MetricSpec",
    "EventSpec",
    "MetricKind",
    "Owner",
    "Reduction",
    "Stability",
    "Tier",
    "Recorder",
    "InstanceRecorder",
    "NullRecorder",
    "NULL_RECORDER",
    "TelemetryConfig",
    "DEFAULT_CONFIG",
    "make_recorder",
    "RecorderRegistry",
]
