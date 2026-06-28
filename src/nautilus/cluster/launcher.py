"""Spawn the worker processes and reap them.

The two-process harness generalized to W workers under one ``spawn`` context. This module only starts
and stops processes and owns the control queues' creation; the coordinator drives the bootstrap over
them. Reaping is unconditional (terminate, join, then kill as a last resort) so a worker never lingers
as a zombie even when the job fails.

The plan crosses to each worker as cloudpickled bytes (a spawn argument), because stdlib pickle — which
the queues use — cannot carry the plan's lambda operator factories. The worker entrypoint and any
operator classes the factories name must be importable in the spawned child, so demo and test pipelines
use module-level classes.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

from nautilus.cluster.worker_main import worker_main
from nautilus.telemetry import TelemetryConfig


def spawn_workers(
    plan_bytes: bytes,
    placement: dict[tuple[str, int], int],
    host: str,
    capacity: int,
    config: TelemetryConfig,
    num_workers: int,
) -> tuple[list[Any], Any, dict[int, Any]]:
    """Start ``num_workers`` worker processes. Returns the processes, the shared events queue
    (workers → coordinator) and a per-worker command queue (coordinator → worker)."""
    ctx = mp.get_context("spawn")
    events: Any = ctx.Queue()
    commands: dict[int, Any] = {wid: ctx.Queue() for wid in range(num_workers)}
    procs: list[Any] = []
    try:
        for wid in range(num_workers):
            proc = ctx.Process(
                target=worker_main,
                args=(
                    wid,
                    plan_bytes,
                    placement,
                    host,
                    capacity,
                    config,
                    events,
                    commands[wid],
                ),
            )
            proc.start()
            procs.append(proc)
    except BaseException:
        reap(procs)  # a later start() failed — never orphan the workers already started
        raise
    return procs, events, commands


def reap(procs: list[Any]) -> None:
    """Stop every worker and join it, so none is left running or becomes a zombie."""
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    for proc in procs:
        proc.join(timeout=10)
        if proc.is_alive():  # last resort
            proc.kill()
            proc.join(timeout=5)
