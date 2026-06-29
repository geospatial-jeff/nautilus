"""``compile_graph`` — lower a :class:`~nautilus.api.LogicalGraph` to a :class:`PhysicalPlan`.

This is where logical intent becomes a physical layout. It does these jobs the IR deliberately left open:

* **Orders the operators topologically** (source-first), tie-breaking by the vertex's insertion index so
  the order is reproducible across machines, then **names them by that position** — not by the vertices'
  logical ids, so two graphs that differ only in naming compile to the same plan: a lone source is
  ``"source"``, the ``j``-th transform/join is ``"op{j}"``, and the synthesized sink is ``"sink"``.
* **Selects a partitioner spec for every edge** from the *downstream* operator's shape and the *edge's*
  key columns — a single instance takes :class:`ForwardSpec`, a keyed fan-out a :class:`KeyGroupSpec`
  shuffle, a keyless fan-out :class:`RoundRobinSpec`. A join's two edges both read the join vertex's one
  parallelism ``Q`` and the run's one ``G``, so their group tables match and a key co-partitions to the
  same join instance from both sides.
* **Synthesizes the sink.** The user describes only the work; the collecting sink that gathers the graph's
  single leaf is the compiler's, pinned to one instance.

A graph with no explicit edges is the linear shape: the compiler synthesizes the positional, port-0
adjacency (``source -> op0 -> ... -> leaf``), so a linear graph compiles byte-for-byte as it always has.

A keyed shuffle routes through key groups: each keyed edge gets a ``group → instance`` table of length
``G`` (the ``key_groups`` argument, defaulting to the operator's parallelism ``Q`` — the identity table).
A ``key_groups`` above ``Q`` makes a later rescale a table swap, not a reshuffle; ``G < Q`` is rejected
here, because then some instance would own no group.

It also rejects a parallel vertex whose factory hands back one shared instance, because the executor must
build a *fresh* operator per subtask — at parallelism > 1 a shared instance's ``open()`` would overwrite
its own per-instance context.
"""

from __future__ import annotations

from nautilus.api import LogicalEdge, LogicalGraph, LogicalVertex
from nautilus.api.graph import _SOURCE, _topological_order
from nautilus.compile.plan import (
    ForwardSpec,
    KeyGroupSpec,
    PartitionerSpec,
    PhysicalEdge,
    PhysicalOperator,
    PhysicalPlan,
    RoundRobinSpec,
)

#: The synthesized sink: a single instance that collects the graph's leaf. There is no sink operator
#: class — the executor runs it as a collecting loop — so this is a label, not an operator class.
SINK_ID = "sink"
SINK_CLASS = "CollectSink"


#: Upper bound on key groups (G): the rescale ceiling, and the length of the table the plan ships. A
#: larger value is almost certainly a mistake and would materialize a huge table. (Matches Flink's
#: default max-parallelism cap.)
MAX_KEY_GROUPS = 32768


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
        if num_groups > MAX_KEY_GROUPS:
            raise ValueError(
                f"key groups G={num_groups} exceeds the maximum {MAX_KEY_GROUPS} (it sizes the routing "
                "table the plan ships); choose a smaller rescale ceiling"
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


def _logical_edges(graph: LogicalGraph) -> tuple[LogicalEdge, ...]:
    """The graph's edges: the explicit list when given, otherwise the synthesized positional, port-0
    adjacency of a linear graph (``v0 -> v1 -> ... -> vn``), each edge keyed by the downstream vertex's
    ``key_columns`` convenience. Either way the IR already validated the shape."""
    if graph.edges:
        return graph.edges
    vs = graph.vertices
    return tuple(
        LogicalEdge(vs[i].id, vs[i + 1].id, 0, vs[i + 1].key_columns) for i in range(len(vs) - 1)
    )


def compile_graph(graph: LogicalGraph, *, key_groups: int | None = None) -> PhysicalPlan:
    """Lower ``graph`` to a runnable, cloudpickle-able :class:`PhysicalPlan`.

    ``key_groups`` (``G``) is the number of key groups every keyed shuffle routes through; ``None``
    defaults each keyed edge to its operator's parallelism (the identity table). A given ``G`` must be
    ``>= Q`` for every keyed operator, and the graph must have at least one keyed edge for ``key_groups``
    to mean anything — lowering raises on a ``G < Q`` or a ``key_groups`` set on a graph with no keyed
    shuffle."""
    by_id = {v.id: v for v in graph.vertices}
    edges = _logical_edges(graph)
    order = _topological_order(graph.vertices, edges)

    # Physical ids by topological position: a lone source is "source", every transform/join is "op{j}".
    # When there is more than one source (a join over two sources) every source is indexed — "source0",
    # "source1", … — so only a single-source graph uses the bare "source", and a linear graph names
    # exactly as it always has.
    num_sources = sum(1 for v in graph.vertices if v.kind == _SOURCE)
    phys: dict[str, str] = {}
    operators: list[PhysicalOperator] = []
    op_index = source_index = 0
    for vid in order:
        vertex = by_id[vid]
        if vertex.kind == _SOURCE:
            pid = "source" if num_sources == 1 else f"source{source_index}"
            source_index += 1
        else:
            pid = f"op{op_index}"
            op_index += 1
        phys[vid] = pid
        operators.append(
            PhysicalOperator(
                pid, _op_class(vertex), vertex.kind, vertex.parallelism, vertex.factory
            )
        )

    # The collecting sink attaches to the graph's single leaf (the vertex with no outbound edge — the
    # last transform, or the join's output). Fan-out to several leaves/sinks is not built yet.
    has_outbound = {e.src for e in edges}
    leaves = [vid for vid in order if vid not in has_outbound]
    if len(leaves) != 1:
        raise ValueError(
            f"a graph must have exactly one leaf (output) vertex to attach the sink to, got "
            f"{len(leaves)}: {[phys[v] for v in leaves]}"
        )
    operators.append(PhysicalOperator(SINK_ID, SINK_CLASS, "sink", 1, None))

    # One physical edge per logical edge (spec from the downstream operator's width and the edge's keys),
    # plus the synthesized leaf -> sink edge (one instance, unkeyed -> Forward).
    physical_edges = [
        PhysicalEdge(
            phys[e.src],
            phys[e.dst],
            _spec_for(by_id[e.dst].parallelism, e.key_columns, key_groups),
            e.dst_input_port,
        )
        for e in edges
    ]
    physical_edges.append(PhysicalEdge(phys[leaves[0]], SINK_ID, _spec_for(1, None, key_groups), 0))

    if key_groups is not None and not any(isinstance(e.spec, KeyGroupSpec) for e in physical_edges):
        # key_groups only means anything for a keyed shuffle; if the graph has none, the argument was
        # silently ignored — surface that instead, so a mistaken --key-groups can't pass unnoticed.
        raise ValueError(
            f"key_groups={key_groups} was given but no edge is keyed (no keyed operator at "
            "parallelism > 1), so it would have no effect"
        )

    return PhysicalPlan(tuple(operators), tuple(physical_edges))
