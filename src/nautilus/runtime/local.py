"""The single-process in-memory entry points: ``run_local_chain`` and the synchronous ``run``.

``run_local_chain`` lowers a ``(source, transforms)`` chain to a :class:`~nautilus.compile.plan.PhysicalPlan`
and runs it through the compiled executor (:func:`~nautilus.runtime.run.run_plan`) over in-process
channels — the *same* engine the parallel and multi-worker paths use, so the default run exercises one
execution path, not a second hand-wired copy. ``run`` is the synchronous one-liner around it. The
report-assembly boundary lives in :mod:`nautilus.runtime.run`; the shared run metadata in
:mod:`nautilus.runtime.meta`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from nautilus.core.operator import OneInputOperator, SourceOperator
from nautilus.core.time import Clock
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.runtime.parallel import graph_from_ops
from nautilus.runtime.result import RunResult
from nautilus.runtime.run import run_plan
from nautilus.telemetry import RecorderRegistry, TelemetryConfig
from nautilus.telemetry.report import Sink


async def run_local_chain(
    source: SourceOperator,
    transforms: list[OneInputOperator],
    *,
    capacity: int = DEFAULT_CAPACITY,
    clock: Clock | None = None,
    telemetry: TelemetryConfig | None = None,
    sink: Sink | None = None,
    registry: RecorderRegistry | None = None,
) -> RunResult:
    """Run a linear pipeline single-process and return the final-stage batches plus a telemetry report.

    Lowers the ``(source, transforms)`` chain to a plan and runs it through the compiled executor, so a
    single-process run uses the same engine as ``--parallelism``/``--workers``. Pass an external
    ``registry`` to read snapshots live while the run is in flight (the live server does this); the
    default creates its own.
    """
    return await run_plan(
        graph_from_ops(source, transforms),
        capacity=capacity,
        clock=clock,
        telemetry=telemetry,
        sink=sink,
        registry=registry,
    )


def run(source: SourceOperator, transforms: list[OneInputOperator], **kwargs: Any) -> RunResult:
    """Synchronous one-liner: run a bounded pipeline to completion and return its :class:`RunResult`.

    A convenience for top-level / boundary use (scripts, a REPL) that wraps :func:`run_local_chain` in
    :func:`asyncio.run`. It cannot be called from inside a running event loop — ``await
    run_local_chain(...)`` directly there (as the CLI and live server do). Accepts the same keyword
    arguments as :func:`run_local_chain` (``capacity``, ``clock``, ``telemetry``, ``sink``,
    ``registry``)."""
    return asyncio.run(run_local_chain(source, transforms, **kwargs))
