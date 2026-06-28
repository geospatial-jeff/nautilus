"""Boundary helpers shared by every runner: the run metadata and its config digest.

Building a :class:`~nautilus.telemetry.report.RunReport` is a boundary-time job done in three places —
the single-process runner (:mod:`nautilus.runtime.run`), the live server (:mod:`nautilus.telemetry.live`),
and the Stage 2 coordinator (:mod:`nautilus.cluster.coordinator`). They share the :class:`RunMeta` these
functions assemble, so it lives here rather than in any one of them: the ``config_digest`` in particular
must be computed identically wherever a report is stamped, or two reports of the same run would not
compare. This is a boundary module, so it may import the report layer.
"""

from __future__ import annotations

import hashlib
import json
import platform

import nautilus
from nautilus.core.time import Clock
from nautilus.telemetry import TelemetryConfig
from nautilus.telemetry.report import RunMeta, Topology


def config_digest(topology: Topology, config: TelemetryConfig, capacity: int) -> str:
    """A short, stable hash of the run's *shape* (capacity, tier, topology nodes/edges) — so two runs of
    the same configured pipeline carry the same digest, and a shape change is visible at a glance.
    """
    canonical = {
        "capacity": capacity,
        "tier": int(config.tier),
        "nodes": [[n.operator_id, n.op_class, n.kind] for n in topology.nodes],
        "edges": [[e.src_operator_id, e.dst_operator_id, e.channel_index] for e in topology.edges],
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def make_run_meta(
    *,
    run_id: str,
    started_at: int,
    ended_at: int,
    wall_micros: int,
    clk: Clock,
    topology: Topology,
    config: TelemetryConfig,
    capacity: int,
) -> RunMeta:
    """Assemble a :class:`RunMeta`. Shared so the runner, the coordinator, and the live server (which
    builds a point-in-time meta mid-run) all stamp identical metadata."""
    return RunMeta(
        run_id=run_id,
        started_at_micros=started_at,
        ended_at_micros=ended_at,
        wall_micros=wall_micros,
        clock_kind=type(clk).__name__,
        nautilus_version=nautilus.__version__,
        python_version=platform.python_version(),
        config_digest=config_digest(topology, config, capacity),
        capacity=capacity,
    )
