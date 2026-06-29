"""``deploy`` — run a graph across W spawned workers and aggregate one report.

The coordinator is the control plane and nothing more. It compiles the graph once, computes placement,
spawns the workers, drives the two-phase bootstrap, and waits at the job boundary for one ``Done`` per
worker (it moves only control messages — see ``DESIGN.md`` for why this keeps the data path
scheduler-free).

As the telemetry boundary it does the report assembly the workers don't: it translates the plan into a
:class:`Topology` and aggregates every worker's raw snapshots into the single :class:`RunReport`. It is
fail-fast: a worker's :class:`Failed` (its child traceback) or a hard crash aborts the run and reaps
every worker immediately, so a failure never hangs or orphans a process.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter_ns

import cloudpickle
import pyarrow as pa

from nautilus.api import LogicalGraph
from nautilus.cluster.cohort import LocalCohort, RemoteCohort, WorkerCohort
from nautilus.cluster.launcher import spawn_workers
from nautilus.cluster.placement import max_parallelism, place
from nautilus.cluster.protocol import Done, Failed, decode_batches
from nautilus.cluster.rendezvous import WorkerCrashed, WorkerError, bind_barrier
from nautilus.compile import compile_graph
from nautilus.core.time import Clock, SystemClock
from nautilus.driver.meta import make_run_meta
from nautilus.driver.result import RunResult
from nautilus.driver.run import plan_to_topology
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.telemetry import TelemetryConfig
from nautilus.telemetry.model import InstanceSnapshot
from nautilus.telemetry.report import NullSink, Sink, build_report
from nautilus.transport.connector import DEFAULT_CONNECT_TIMEOUT

_log = logging.getLogger(__name__)
_DEFAULT_BOOTSTRAP_TIMEOUT = 60.0  # max seconds of silence between worker registrations during bind


def deploy(
    graph: LogicalGraph,
    *,
    num_workers: int = 2,
    capacity: int = DEFAULT_CAPACITY,
    key_groups: int | None = None,
    host: str = "127.0.0.1",
    advertise_host: str | None = None,
    daemons: list[tuple[str, int]] | None = None,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    clock: Clock | None = None,
    telemetry: TelemetryConfig | None = None,
    sink: Sink | None = None,
    bootstrap_timeout: float = _DEFAULT_BOOTSTRAP_TIMEOUT,
) -> RunResult:
    """Compile ``graph`` and run it across workers, returning the sink's batches plus one aggregated
    telemetry report. Two backends, same body: with ``daemons`` ``None`` (the default) it spawns
    ``num_workers`` local worker processes; with ``daemons`` a roster of ``(host, port)`` worker-daemon
    control addresses it dials them instead (the multi-node path), assigning ``worker_id = roster index``
    and inferring the worker count from the roster. Either count is capped at the plan's maximum
    parallelism (a wider W would only idle a worker).

    For the local path, ``host`` is the interface every worker's listener binds (loopback by default;
    ``0.0.0.0`` to accept on a container's bridge) and ``advertise_host`` is the routable host peers dial
    (defaults to ``host``, so a single-machine run is unchanged); the daemon supplies its own bind/advertise
    in the remote path, so these are ignored there. ``connect_timeout`` bounds dialing each daemon's
    control port.

    ``bootstrap_timeout`` bounds only the bind/register phase, where a silent worker means a hang; once
    the job is running the wait is unbounded, because a healthy job runs as long as its data does. The
    full ``telemetry`` config reaches every worker; a custom ``clock``, however, cannot cross to a worker,
    so it affects only the coordinator's run-meta timestamps — worker operators always use a ``SystemClock``.
    Always reaps every worker. Raises :class:`WorkerError` (with the failing worker's traceback) on a
    caught operator error, :class:`WorkerCrashed` on a hard crash (or a daemon's control connection
    closing before it reports), or ``TimeoutError`` if a worker never registers."""
    if daemons is not None:
        if not daemons:
            raise ValueError("daemons roster is empty")
        num_workers = len(daemons)  # the roster is the worker count in the remote path
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    plan = compile_graph(graph, key_groups=key_groups)
    effective = min(num_workers, max_parallelism(plan))
    if effective < num_workers:
        _log.info(
            "capping workers from %d to %d (the plan's maximum parallelism)", num_workers, effective
        )
    worker_ids = list(range(effective))
    placement = place(plan, worker_ids)

    clk = clock or SystemClock()
    config = telemetry or TelemetryConfig(clock=clk)
    out_sink = sink or NullSink()
    topology = plan_to_topology(plan, capacity)

    plan_bytes = cloudpickle.dumps(plan)
    # Workers get the full telemetry config (so every tier/interval/capacity/validate setting matches a
    # single-process run), except the clock: a user clock generally cannot cross a spawn, and the data
    # plane times itself with a SystemClock regardless. A custom clock therefore affects only the
    # coordinator's run-meta timestamps, not worker operators.
    worker_config = replace(config, clock=SystemClock())
    # Stamp the start time BEFORE spawning, so a custom clock that raises cannot do so between spawn and
    # the reap try-block — which would orphan the live workers and break the "always reaps" guarantee.
    started_at = clk.now_micros()
    wall0 = perf_counter_ns()
    if daemons is not None:
        cohort: WorkerCohort = RemoteCohort.launch(
            daemons, plan_bytes, placement, capacity, worker_config, effective, connect_timeout
        )
    else:
        advertise = advertise_host if advertise_host is not None else host
        cohort = LocalCohort(
            *spawn_workers(
                plan_bytes, placement, host, advertise, capacity, worker_config, effective
            )
        )
    try:
        bind_barrier(cohort, effective, bootstrap_timeout)

        # Job-boundary completion: one Done per worker. No wall-clock timeout here — a busy worker is
        # silent until it finishes, so the wait is bounded only by the job's own length (crash detection
        # still fires). The sink's batches come from whichever worker hosts it; every worker's snapshots
        # are aggregated into the one report.
        snapshots: list[InstanceSnapshot] = []
        batches: list[pa.RecordBatch] = []
        remaining = set(worker_ids)
        while remaining:
            # Crash-detect only still-outstanding workers: a worker's whole contribution is in its Done
            # message, after which it tears down and may exit non-zero — that must not fail a run whose
            # data is already complete. The cohort narrows liveness to the watched set.
            message = cohort.next_event(None, remaining)
            if isinstance(message, Failed):
                raise WorkerError(message.worker_id, message.traceback)
            if not isinstance(message, Done):
                raise RuntimeError(f"unexpected control message awaiting completion: {message!r}")
            snapshots.extend(message.snapshots)
            batches.extend(decode_batches(message.sink_batches))
            remaining.discard(message.worker_id)

        wall_micros = (perf_counter_ns() - wall0) // 1000
        ended_at = clk.now_micros()
        meta = make_run_meta(
            run_id=config.run_id or f"run-{started_at}",
            started_at=started_at,
            ended_at=ended_at,
            wall_micros=wall_micros,
            clk=clk,
            topology=topology,
            config=config,
            capacity=capacity,
        )
        report = build_report(snapshots, meta=meta, topology=topology)
        out_sink.emit_report(report)
        return RunResult(batches, report)
    finally:
        cohort.reap()


__all__ = ["deploy", "WorkerError", "WorkerCrashed"]
