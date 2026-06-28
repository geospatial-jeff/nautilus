"""The ``(source, instances)`` pipeline → ``LogicalGraph`` bridge.

The CLI and :func:`~nautilus.driver.local.run_local_chain` take the convenience ``(source, [operator])``
shape and run every transform at one uniform parallelism; this lowers that shape to a
:class:`~nautilus.api.LogicalGraph` the compiler can lower. ``api`` stays pure (it imports nothing else in
nautilus), so this bridge — which knows the operator types — lives here at the boundary, not in the IR.
The fluent :class:`nautilus.dsl.Stream` is the richer builder; this is the thin adapter the simple shape
needs.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

from nautilus.api import LogicalGraph, LogicalVertex, linear_graph
from nautilus.core.operator import OneInputOperator, SourceOperator


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


def graph_from_pipeline(
    source: SourceOperator, transforms: Sequence[OneInputOperator], parallelism: int
) -> LogicalGraph:
    """Bridge a ``(source, instances)`` pipeline to a graph that runs every transform at ``parallelism``,
    keyed by each operator's self-declared :meth:`~nautilus.core.operator.OneInputOperator.key_columns`.

    This is the CLI's builder→IR bridge. At ``parallelism > 1`` each instance is replicated per subtask
    by :func:`copy.deepcopy` (a fresh, unopened operator — the operator must be deep-copyable), since one
    shared instance cannot be replicated. Each vertex carries the operator's own ``key_columns()``, so a
    keyed operator is never silently round-robined (the compiler turns that into the partitioner).
    """
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
