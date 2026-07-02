"""Serve the live dashboard for a distributed run.

The single-process dashboard (:func:`nautilus.telemetry.live.serve_graph`) runs the pipeline itself and
reads one registry. A distributed run has no single registry — each worker holds its own — so this drives
:func:`~nautilus.cluster.coordinator.deploy` with a live-report hook and serves what the coordinator
aggregates from every worker's heartbeats. It is the cluster counterpart to ``serve_graph``: the same
``LiveServer`` and dashboard HTML, fed by the coordinator instead of an in-process run.

``deploy`` blocks until the (bounded) run completes; the ``LiveServer`` runs on its own daemon thread the
whole time, serving the report the coordinator updates on each heartbeat. When the run finishes the
dashboard lingers on the final aggregated report until interrupted — the batch-first analog of a saved
report, captured live. This lives in ``cluster`` rather than ``telemetry.live`` because it imports
``deploy``, and an import-linter contract forbids ``telemetry`` from importing the control plane.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from nautilus.api import LogicalGraph
from nautilus.cluster.coordinator import deploy
from nautilus.driver.result import RunResult
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.telemetry import TelemetryConfig
from nautilus.telemetry.live import (
    LiveAggregator,
    LiveServer,
    load_dashboard_html,
    open_in_browser,
)


def serve_cluster(
    graph: LogicalGraph,
    *,
    num_workers: int = 2,
    daemons: list[tuple[str, int]] | None = None,
    key_groups: int | None = None,
    capacity: int = DEFAULT_CAPACITY,
    telemetry: TelemetryConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8787,
    heartbeat_interval_micros: int = 500_000,
    linger: bool = True,
    max_seconds: float | None = None,
    open_browser: bool = False,
    on_ready: Callable[[str], None] | None = None,
) -> RunResult:
    """Run ``graph`` across workers and serve a live dashboard of the aggregated telemetry, returning the
    run's :class:`RunResult`. Serves at ``http://host:port``; ``num_workers`` / ``daemons`` choose the
    backend exactly as :func:`~nautilus.cluster.coordinator.deploy` does. A bounded run completes, then
    (with ``linger``) the dashboard keeps serving the final aggregated report until interrupted;
    ``max_seconds`` caps that linger so a demo or CI exits on its own. ``on_ready`` is called with the URL
    once the server is up. Unbounded pipelines are not yet supported here — ``deploy`` waits for the run to
    finish before lingering."""
    aggregator = LiveAggregator()
    server = LiveServer(aggregator, load_dashboard_html(), host=host, port=port)
    server.start()
    if on_ready is not None:
        on_ready(server.url)
    if open_browser:
        open_in_browser(server.url)
    try:
        result = deploy(
            graph,
            num_workers=num_workers,
            daemons=daemons,
            key_groups=key_groups,
            capacity=capacity,
            telemetry=telemetry,
            on_report=aggregator.update,
            heartbeat_interval_micros=heartbeat_interval_micros,
        )
        # The final Done-built report is authoritative — serve it, then hold it as "completed".
        aggregator.update(result.telemetry)
        aggregator.mark_completed()
        if linger:
            _linger(max_seconds)
        return result
    finally:
        server.stop()


def _linger(max_seconds: float | None) -> None:
    """Block, serving the final report, until interrupted — or ``max_seconds`` elapses. Waits in short
    ticks so a ``KeyboardInterrupt`` lands promptly (a bare ``Event().wait()`` with no timeout can swallow
    it on the main thread)."""
    idle = threading.Event()
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    while not idle.wait(0.5):
        if deadline is not None and time.monotonic() >= deadline:
            return
