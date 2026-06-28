"""The ``Stage``/pipeline → ``LogicalGraph`` bridges, plus ``run_parallel_chain`` over the compiler.

Stage 1.5 ran a parallel channel mesh here directly. Stage 2's compiler and executor subsumed it: a
graph is now lowered to a :class:`~nautilus.compile.plan.PhysicalPlan` and run through
:func:`~nautilus.runtime.run.run_plan` (single process) or :func:`~nautilus.cluster.deploy` (workers),
and the in-process-vs-socket choice the old ``ChannelFactory`` made is the
:class:`~nautilus.runtime.connector.Connector`'s job.

What stays here is the adapter from the ephemeral wiring types — :class:`Stage`, or bare operator
instances — to a :class:`~nautilus.api.LogicalGraph` the compiler can lower. ``api`` itself stays pure
(``runtime`` may import ``api``, never the reverse), so these bridges, which know about ``Stage`` and the
operator types, live here. :func:`run_parallel_chain` remains as a thin wrapper (compile a ``Stage``
chain, run it single-process) so existing callers keep working.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from nautilus.api import LogicalGraph, LogicalVertex, linear_graph
from nautilus.core.operator import OneInputOperator, SourceOperator
from nautilus.core.time import Clock
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.runtime.result import RunResult
from nautilus.runtime.run import run_plan
from nautilus.telemetry import RecorderRegistry, TelemetryConfig
from nautilus.telemetry.report import Sink


@dataclass(frozen=True)
class Stage:
    """One logical operator in a parallel chain.

    ``factory`` builds a *fresh* operator (and, through its ``OperatorContext``, a fresh state backend)
    for each of the ``parallelism`` instances; it must be a side-effect-free constructor (resource
    acquisition belongs in ``open()``), because the compiler also calls it to read the operator class
    name. ``key_columns`` is the source of truth for the edge *feeding this stage*: set, it selects a
    keyed shuffle on those columns; unset with ``parallelism > 1``, a round-robin rebalance;
    ``parallelism == 1`` forwards.
    """

    factory: Callable[[], OneInputOperator]
    parallelism: int = 1
    key_columns: Sequence[str] | None = None


def _const_factory(instance: object) -> Callable[[], object]:
    """A factory that returns one fixed instance. Safe only at parallelism 1 — there the operator has a
    single actor, so the one instance is never replicated and its ``open()`` context is never overwritten.
    """
    return lambda: instance


def _replicating_factory(instance: object) -> Callable[[], object]:
    """A factory that builds a fresh deep copy of ``instance`` each call, for replicating an operator
    across subtasks. Each subtask gets its own unopened operator (its ``open()`` builds per-instance
    state); the operator must be deep-copyable."""
    return lambda: copy.deepcopy(instance)


def _resolve_stage_keys(stage: Stage) -> tuple[str, ...] | None:
    """The key columns for the edge feeding ``stage``, with the operator's own declaration as the source
    of truth. An explicit ``Stage.key_columns`` may only *match* what the operator declares — if they
    disagree, that is a wiring mistake, raised here rather than silently round-robining a keyed operator
    (which would split a key's state across instances). The factory build is side-effect-free per the
    Stage contract (the compiler builds it too)."""
    declared = stage.factory().key_columns()
    explicit = tuple(stage.key_columns) if stage.key_columns else None
    if explicit is not None and declared is not None and explicit != declared:
        raise ValueError(
            f"Stage.key_columns {explicit!r} disagrees with the operator's declared key_columns() "
            f"{declared!r}; drop Stage.key_columns and let the operator declare its key"
        )
    return explicit if explicit is not None else declared


def graph_from_stages(source: SourceOperator, stages: Sequence[Stage]) -> LogicalGraph:
    """Bridge the ``(source instance, [Stage])`` shape to a :class:`~nautilus.api.LogicalGraph`. The
    source is pinned to one instance, so wrapping it in a factory is safe; each :class:`Stage` already
    supplies a fresh-operator factory for its parallelism. The edge key is the operator's own
    ``key_columns()`` (see :func:`_resolve_stage_keys`), so a keyed operator is never silently
    round-robined."""
    vertices = [
        LogicalVertex(
            id=f"op{j}",
            factory=stage.factory,
            kind="one_input",
            parallelism=stage.parallelism,
            key_columns=_resolve_stage_keys(stage),
        )
        for j, stage in enumerate(stages)
    ]
    return linear_graph(_const_factory(source), vertices)


def graph_from_pipeline(
    source: SourceOperator, transforms: Sequence[OneInputOperator], parallelism: int
) -> LogicalGraph:
    """Bridge a ``(source, instances)`` pipeline to a graph that runs every transform at ``parallelism``,
    keyed by each operator's self-declared :meth:`~nautilus.core.operator.OneInputOperator.key_columns`.

    This is the CLI's builder→IR bridge. At ``parallelism > 1`` each instance is replicated per subtask
    by :func:`copy.deepcopy` (a fresh, unopened operator — the operator must be deep-copyable), since one
    shared instance cannot be replicated. A keyed operator's declared columns select the keyed shuffle; a
    keyless operator (``None``) rebalances round-robin, so a key is never silently split."""
    if parallelism < 1:
        raise ValueError(f"parallelism must be >= 1, got {parallelism}")
    vertices = [
        LogicalVertex(
            id=f"op{k}",
            factory=_const_factory(op) if parallelism == 1 else _replicating_factory(op),
            kind="one_input",
            parallelism=parallelism,
            key_columns=op.key_columns(),
        )
        for k, op in enumerate(transforms)
    ]
    return linear_graph(_const_factory(source), vertices)


async def run_parallel_chain(
    source: SourceOperator,
    stages: Sequence[Stage],
    *,
    capacity: int = DEFAULT_CAPACITY,
    clock: Clock | None = None,
    telemetry: TelemetryConfig | None = None,
    sink: Sink | None = None,
    registry: RecorderRegistry | None = None,
) -> RunResult:
    """Run a ``source → stages → sink`` chain single-process by compiling it and executing the plan. A
    thin wrapper over :func:`~nautilus.runtime.run.run_plan` kept for existing callers; for multiple
    processes use :func:`nautilus.cluster.deploy`."""
    return await run_plan(
        graph_from_stages(source, stages),
        capacity=capacity,
        clock=clock,
        telemetry=telemetry,
        sink=sink,
        registry=registry,
    )
