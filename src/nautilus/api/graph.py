"""The logical graph and its vertices — the frozen description a compiler lowers.

A :class:`LogicalVertex` names one operator: the ``factory`` that builds it, how many parallel instances
it runs as, and (for a one-input keyed operator) the columns its input is shuffled on. A
:class:`LogicalEdge` wires one vertex's output into a downstream vertex's input *port*, carrying the
columns that edge is co-partitioned on. A :class:`LogicalGraph` is those vertices wired into a dataflow.

A graph built with **no explicit edges** is the common linear case: ``vertices[0]`` is the sole source
and each later vertex consumes the one before it — the compiler synthesizes that positional, port-0
adjacency. **Explicit edges** describe any DAG, which is what a join needs: a ``two_input`` vertex takes
two inbound edges on distinct ports (port 0 = left, port 1 = right), each keyed on its own columns, so a
key co-partitions to the same join instance from both sides.

This module imports nothing else in nautilus on purpose. The factory it stores returns an *operator* (a
:class:`~nautilus.core.operator.SourceOperator`, :class:`~nautilus.core.operator.OneInputOperator`, or
:class:`~nautilus.core.operator.TwoInputOperator`), but the IR never names those types — it treats a
factory as an opaque ``() -> object`` so the value layer cannot reach down into the runtime. The
compiler, which does know those types, calls the factory; the IR only carries it.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: A vertex's operator constructor: a side-effect-free, zero-argument callable returning a fresh
#: operator. It must build a *new* instance each call (resource acquisition belongs in ``open()``),
#: because the compiler calls it to read the operator class name — and again, for a parallel vertex, to
#: check it returns a fresh instance — and the executor calls it once per parallel instance. A factory
#: that returns one shared instance is only safe at parallelism 1.
VertexFactory = Callable[[], object]

#: The kinds of vertex the IR supports. The *collecting* sink is synthesized by the compiler and never
#: authored here; an ``async_sink`` is the one authored terminal — a user operator that writes to an
#: external store and so takes the place of the synthesized collector as the graph's leaf.
_SOURCE = "source"
_ONE_INPUT = "one_input"
_TWO_INPUT = "two_input"
_ASYNC_SINK = "async_sink"
_KINDS = frozenset({_SOURCE, _ONE_INPUT, _TWO_INPUT, _ASYNC_SINK})

#: How many inbound ports each kind consumes — the number of edges that must arrive at it, on the ports
#: ``0 .. n-1``. A source has none; a one-input transform and an async sink one (port 0); a two-input
#: join two (port 0 is the left input, port 1 the right).
_NUM_INPUTS = {_SOURCE: 0, _ONE_INPUT: 1, _TWO_INPUT: 2, _ASYNC_SINK: 1}


@dataclass(frozen=True, slots=True)
class LogicalVertex:
    """One operator in a :class:`LogicalGraph`.

    ``id`` is a stable logical handle the edges reference; the compiler derives the *physical* operator
    id from topological position, so two graphs that differ only in vertex ids compile to the same plan.
    ``key_columns`` is the keyed-input convenience for the **linear** shape — the columns this operator's
    input is co-partitioned on — which the compiler copies onto the synthesized port-0 edge. In an
    explicit-edge graph the edge carries the keying instead (a join keys its two inputs on different
    columns), so a ``two_input`` vertex leaves ``key_columns`` ``None``.
    """

    id: str
    factory: VertexFactory
    kind: str
    parallelism: int = 1
    key_columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a LogicalVertex needs a non-empty id")
        if self.kind not in _KINDS:
            raise ValueError(f"unknown vertex kind {self.kind!r}; expected one of {sorted(_KINDS)}")
        if self.parallelism < 1:
            raise ValueError(f"parallelism must be >= 1, got {self.parallelism} for {self.id!r}")
        if self.key_columns is not None and not self.key_columns:
            # None means keyless; a non-empty tuple means keyed. An empty tuple is neither — reject it
            # rather than let the compiler silently downgrade it to a keyless round-robin.
            raise ValueError(
                f"key_columns must be None (keyless) or a non-empty tuple, got () for {self.id!r}"
            )
        if self.key_columns is not None and any(not c or not c.strip() for c in self.key_columns):
            raise ValueError(
                f"key_columns must be non-empty column names, got {self.key_columns!r} for {self.id!r}"
            )
        if self.key_columns is not None and self.kind != _ONE_INPUT:
            # Only a one-input vertex has a single input the synthesized linear edge can key. A source
            # has no input; a join keys per edge (left_on/right_on) — reject a value that has no effect.
            raise ValueError(
                f"key_columns is only meaningful on a one_input vertex; {self.id!r} is {self.kind!r} "
                "(a join keys its inputs on its edges)"
            )


@dataclass(frozen=True, slots=True)
class LogicalEdge:
    """A directed edge from vertex ``src`` to a downstream vertex's input ``dst_input_port``.

    The port is what lets a :class:`two_input <LogicalVertex>` distinguish its left input (port 0) from
    its right (port 1); a one-input vertex and the source consume only port 0. ``key_columns`` are the
    columns *this edge* is co-partitioned on — keying lives on the edge, not the vertex, so a join's two
    inputs can shuffle on differently-named columns yet land equal keys on the same instance.
    """

    src: str
    dst: str
    dst_input_port: int = 0
    key_columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.src or not self.dst:
            raise ValueError("a LogicalEdge needs non-empty src and dst ids")
        if self.dst_input_port not in (0, 1):
            raise ValueError(
                f"dst_input_port must be 0 or 1, got {self.dst_input_port} for {self.src!r}->{self.dst!r}"
            )
        if self.key_columns is not None and not self.key_columns:
            raise ValueError(
                f"key_columns must be None or a non-empty tuple, got () for {self.src!r}->{self.dst!r}"
            )
        if self.key_columns is not None and any(not c or not c.strip() for c in self.key_columns):
            raise ValueError(
                f"key_columns must be non-empty column names, got {self.key_columns!r} for "
                f"{self.src!r}->{self.dst!r}"
            )


@dataclass(frozen=True, slots=True)
class LogicalGraph:
    """A dataflow as vertices wired by edges. ``edges`` empty is the linear shape — ``vertices[0]`` is the
    sole source and each later vertex consumes its predecessor (build it with :func:`linear_graph`).
    A non-empty ``edges`` is an explicit DAG (build a join with :func:`two_input` and :class:`LogicalEdge`).
    """

    vertices: tuple[LogicalVertex, ...]
    edges: tuple[LogicalEdge, ...] = ()

    def __post_init__(self) -> None:
        if not self.vertices:
            raise ValueError("a LogicalGraph needs at least one vertex (the source)")
        ids = [v.id for v in self.vertices]
        if len(set(ids)) != len(ids):
            raise ValueError(f"vertex ids must be unique, got {ids}")
        if not self.edges:
            self._validate_linear()
        else:
            self._validate_dag()

    def _validate_linear(self) -> None:
        """The legacy positional shape: exactly one leading source, one-input transforms after it. A join
        needs two inputs, which positional adjacency cannot express, so it must use explicit edges.
        """
        sources = [v for v in self.vertices if v.kind == _SOURCE]
        if len(sources) != 1 or self.vertices[0].kind != _SOURCE:
            raise ValueError("a linear graph must start with exactly one source vertex")
        if self.vertices[0].parallelism != 1:
            raise ValueError("the source vertex must have parallelism 1")
        if any(v.kind == _TWO_INPUT for v in self.vertices):
            raise ValueError(
                "a two_input (join) vertex needs explicit edges for its two inputs; it cannot appear "
                "in a linear graph"
            )
        if any(v.kind == _ASYNC_SINK for v in self.vertices):
            raise ValueError(
                "an async_sink vertex needs an explicit edge from its input; it cannot appear in a "
                "linear graph (build it with the DSL .sink())"
            )

    def _validate_dag(self) -> None:
        """An explicit-edge DAG: every endpoint exists, sources have no inbound, each non-source has its
        kind's full set of input ports, no self-join, and no cycle."""
        by_id = {v.id: v for v in self.vertices}
        inbound: dict[str, list[LogicalEdge]] = {v.id: [] for v in self.vertices}
        has_outbound: set[str] = set()
        for e in self.edges:
            if e.src not in by_id:
                raise ValueError(f"edge references unknown src vertex {e.src!r}")
            if e.dst not in by_id:
                raise ValueError(f"edge references unknown dst vertex {e.dst!r}")
            inbound[e.dst].append(e)
            has_outbound.add(e.src)
        if not any(v.kind == _SOURCE for v in self.vertices):
            raise ValueError("a graph needs at least one source vertex (a vertex with no input)")
        for v in self.vertices:
            if v.kind == _ASYNC_SINK and v.id in has_outbound:
                # A sink writes to an external store and has no output, so it must be the graph's leaf —
                # chaining a combinator off it would feed an edge from a vertex that produces no frames.
                raise ValueError(
                    f"async_sink vertex {v.id!r} has an outbound edge; a sink must be a leaf (no "
                    "downstream)"
                )
            ins = inbound[v.id]
            if v.kind == _SOURCE:
                if ins:
                    raise ValueError(f"source vertex {v.id!r} must have no inbound edge")
                if v.parallelism != 1:
                    raise ValueError("the source vertex must have parallelism 1")
                continue
            want = list(range(_NUM_INPUTS[v.kind]))
            ports = sorted(e.dst_input_port for e in ins)
            if ports != want:
                raise ValueError(
                    f"vertex {v.id!r} ({v.kind}) needs exactly one inbound edge per input port {want}, "
                    f"got ports {ports}"
                )
            if v.kind == _TWO_INPUT and len({e.src for e in ins}) == 1:
                # Both inputs from one upstream: the two edges would share a ChannelId (which has no port
                # field), colliding the channel and its edge-metric labels. Route the two sides through
                # distinct upstreams; threading a port into ChannelId is the deferred real fix.
                raise ValueError(
                    f"two_input vertex {v.id!r} has both inputs from the same vertex (a self-join); "
                    "feed its two ports from distinct upstreams (port-in-ChannelId is deferred)"
                )
            if (
                v.kind == _TWO_INPUT
                and v.parallelism > 1
                and any(e.key_columns is None for e in ins)
            ):
                # A keyless edge into a parallel stage fans out round-robin, which scatters a key across
                # instances — fine for a stateless one-input rebalance, but for a join it splits a key's
                # two sides onto different instances so they never meet and matches vanish silently. Both
                # join inputs must be keyed so equal keys co-partition. (Harmless at parallelism 1, where
                # a single instance owns everything.)
                raise ValueError(
                    f"two_input vertex {v.id!r} runs at parallelism {v.parallelism} but an input edge is "
                    "keyless; both inputs must carry key_columns so equal keys co-partition to the same "
                    "instance (in the DSL, give the join on=/left_on=/right_on=)"
                )
        _topological_order(self.vertices, self.edges)  # raises on a cycle


def _topological_order(
    vertices: Sequence[LogicalVertex], edges: Sequence[LogicalEdge]
) -> list[str]:
    """Vertex ids in a deterministic topological order, raising on a cycle.

    Ties (several vertices ready at once) break by **insertion index** — a vertex's position in
    ``vertices`` — never dict/set iteration order, so the order, and therefore the physical operator
    names and the structural digest the compiler derives from it, are reproducible across machines.
    """
    index = {v.id: i for i, v in enumerate(vertices)}
    indegree = {v.id: 0 for v in vertices}
    successors: dict[str, list[str]] = {v.id: [] for v in vertices}
    for e in edges:
        indegree[e.dst] += 1
        successors[e.src].append(e.dst)
    ready = [(index[vid], vid) for vid, d in indegree.items() if d == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        _, vid = heapq.heappop(ready)
        order.append(vid)
        for nxt in successors[vid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, (index[nxt], nxt))
    if len(order) != len(vertices):
        raise ValueError("the logical graph has a cycle; a dataflow graph must be acyclic")
    return order


def source(id: str, factory: VertexFactory) -> LogicalVertex:
    """Build a source vertex (no input, parallelism 1) for an explicit-edge graph."""
    return LogicalVertex(id=id, factory=factory, kind=_SOURCE)


def one_input(
    id: str,
    factory: VertexFactory,
    *,
    parallelism: int = 1,
    key_columns: tuple[str, ...] | None = None,
) -> LogicalVertex:
    """Build a one-input transform vertex — the common case — without restating ``kind="one_input"``."""
    return LogicalVertex(
        id=id, factory=factory, kind=_ONE_INPUT, parallelism=parallelism, key_columns=key_columns
    )


def two_input(id: str, factory: VertexFactory, *, parallelism: int = 1) -> LogicalVertex:
    """Build a two-input (join) vertex. Its keying lives on its two inbound :class:`LogicalEdge`\\ s (port
    0 = left, port 1 = right), each co-partitioned on its own columns, so it carries no ``key_columns``.

    When hand-building the IR, those edges' ``key_columns`` must name the same columns the join operator
    joins on (its ``left_on``/``right_on``): the edge decides where a row goes, the operator decides what
    matches, so a mismatch shuffles on different columns than it joins and silently mis-joins. The fluent
    :meth:`nautilus.dsl.Stream.join` derives both from one ``on=`` / ``left_on=`` / ``right_on=``, so they
    cannot drift.
    """
    return LogicalVertex(id=id, factory=factory, kind=_TWO_INPUT, parallelism=parallelism)


def async_sink(id: str, factory: VertexFactory, *, parallelism: int = 1) -> LogicalVertex:
    """Build an async-sink vertex — an authored terminal that writes its one input to an external store
    (an :class:`~nautilus.core.operator.AsyncSink`). It must be a leaf. Like a join, its keying lives on
    its inbound :class:`LogicalEdge` (so a parallel keyed sink co-partitions), not on the vertex, so it
    carries no ``key_columns``."""
    return LogicalVertex(id=id, factory=factory, kind=_ASYNC_SINK, parallelism=parallelism)


def linear_graph(source_factory: VertexFactory, vertices: Sequence[LogicalVertex]) -> LogicalGraph:
    """Build a linear ``source -> vertices[0] -> ... -> vertices[-1]`` graph.

    ``source_factory`` builds the source operator; it becomes the single ``"source"`` vertex at
    parallelism 1. ``vertices`` are the one-input transforms in order. The sink is *not* a vertex — the
    compiler synthesizes the collecting sink — so a graph holds only the work the user described. The
    graph carries no explicit edges; the compiler reads the positional adjacency.
    """
    src = LogicalVertex(id=_SOURCE, factory=source_factory, kind=_SOURCE, parallelism=1)
    return LogicalGraph((src, *vertices))
