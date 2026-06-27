"""``compile_graph`` — lower a :class:`~nautilus.api.LogicalGraph` to a :class:`PhysicalPlan`.

This is where logical intent becomes a physical layout. It does three jobs the IR deliberately left
open:

* **Names the physical operators by position**, not by the vertices' logical ids, so two graphs that
  differ only in naming compile to the same plan: the source is ``"source"``, the ``j``-th transform
  is ``"op{j}"``, and the synthesized collecting sink is ``"sink"``.
* **Selects a partitioner spec for every edge** from the *downstream* operator's shape — a single
  instance takes :class:`ForwardSpec`, a keyed operator takes a :class:`KeyGroupSpec` keyed shuffle on
  its key columns, and a keyless fan-out takes :class:`RoundRobinSpec`.
* **Synthesizes the sink.** The user describes only the work; the collecting sink that gathers the
  final stage is the compiler's, pinned to one instance.

A keyed shuffle routes through key groups: each keyed edge gets a ``group → instance`` table of length
``G`` (the ``key_groups`` argument, defaulting to the operator's parallelism ``Q`` — the identity table,
byte-identical to a direct hash). A ``key_groups`` above ``Q`` makes a later rescale a table swap, not a
reshuffle; ``G < Q`` is rejected here, because then some instance would own no group.

It also rejects a parallel vertex whose factory hands back one shared instance, because the executor
must build a *fresh* operator per subtask — at parallelism > 1 a shared instance's ``open()`` would
overwrite its own per-instance context.
"""

from __future__ import annotations

from nautilus.api import LogicalGraph, LogicalVertex
from nautilus.compile.plan import (
    ForwardSpec,
    KeyGroupSpec,
    PartitionerSpec,
    PhysicalEdge,
    PhysicalOperator,
    PhysicalPlan,
    RoundRobinSpec,
)

#: The synthesized sink: a single instance that collects the final stage. There is no sink operator
#: class — the executor runs it as a collecting loop — so this is a label, matching the legacy mesh.
SINK_ID = "sink"
SINK_CLASS = "CollectSink"


def _group_table(num_groups: int, parallelism: int) -> tuple[int, ...]:
    """Map ``num_groups`` key groups round-robin onto ``parallelism`` instances. At ``G == Q`` this is
    the identity ``(0, 1, …, Q-1)``; for ``G > Q`` each instance owns several groups and every instance
    owns at least one, so no instance is left without a key range."""
    return tuple(g % parallelism for g in range(num_groups))


def _spec_for(
    parallelism: int, key_columns: tuple[str, ...] | None, key_groups: int | None
) -> PartitionerSpec:
    """Select the routing spec for an edge feeding a stage of the given width: a single owner forwards
    (even when keyed), a keyed fan-out is a key-group shuffle, a keyless fan-out rebalances round-robin.
    ``key_groups`` is the chosen group count ``G`` (``None`` defaults to ``Q``); it must be ``>= Q``.
    """
    if parallelism == 1:
        return ForwardSpec()
    if key_columns:
        num_groups = parallelism if key_groups is None else key_groups
        if num_groups < parallelism:
            raise ValueError(
                f"key groups G={num_groups} is below the operator parallelism Q={parallelism}; "
                "G must be >= Q so every instance owns at least one key group"
            )
        return KeyGroupSpec(key_columns, _group_table(num_groups, parallelism))
    return RoundRobinSpec()


def _op_class(vertex: LogicalVertex) -> str:
    """Read the operator's class name from its factory, checking that a parallel vertex builds a fresh
    instance each call (a shared instance cannot be replicated across subtasks)."""
    built = vertex.factory()
    if vertex.parallelism > 1 and vertex.factory() is built:
        raise ValueError(
            f"vertex {vertex.id!r} has parallelism {vertex.parallelism} but its factory returns one "
            "shared instance; parallelism > 1 needs a factory that builds a fresh operator each call"
        )
    return type(built).__name__


def compile_graph(graph: LogicalGraph, *, key_groups: int | None = None) -> PhysicalPlan:
    """Lower ``graph`` to a runnable, cloudpickle-able :class:`PhysicalPlan`.

    ``key_groups`` (``G``) is the number of key groups every keyed shuffle routes through; ``None``
    defaults each keyed edge to its operator's parallelism (the identity table). A given ``G`` must be
    ``>= Q`` for every keyed operator, or lowering raises."""
    operators: list[PhysicalOperator] = []

    # The source is operator 0; each later vertex is op{j}. Physical ids come from position, so the
    # plan is independent of the vertices' logical ids.
    for index, vertex in enumerate(graph.vertices):
        operator_id = "source" if index == 0 else f"op{index - 1}"
        operators.append(
            PhysicalOperator(
                operator_id=operator_id,
                op_class=_op_class(vertex),
                kind=vertex.kind,
                parallelism=vertex.parallelism,
                factory=vertex.factory,
            )
        )

    # The compiler-synthesized collecting sink (one instance, no operator class).
    operators.append(
        PhysicalOperator(
            operator_id=SINK_ID, op_class=SINK_CLASS, kind="sink", parallelism=1, factory=None
        )
    )

    # One edge per adjacent pair. The spec is chosen from the *downstream* vertex (its width and key
    # columns); the edge into the sink is a single-owner forward.
    edges: list[PhysicalEdge] = []
    downstream = list(graph.vertices[1:]) + [None]  # None marks the sink (width 1, unkeyed)
    for src, dst_op, dst_vertex in zip(operators[:-1], operators[1:], downstream, strict=True):
        if dst_vertex is None:
            spec: PartitionerSpec = ForwardSpec()
        else:
            spec = _spec_for(dst_op.parallelism, dst_vertex.key_columns, key_groups)
        edges.append(PhysicalEdge(src.operator_id, dst_op.operator_id, spec))

    return PhysicalPlan(tuple(operators), tuple(edges))
