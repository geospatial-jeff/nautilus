"""The logical graph and its vertices — the frozen description a compiler lowers.

A :class:`LogicalVertex` names one operator: the ``factory`` that builds it, how many parallel
instances it runs as, and (for a keyed operator) the columns its input is shuffled on. A
:class:`LogicalGraph` is the vertices wired into a dataflow. Stage 2 is linear-only, so adjacency is
positional — ``vertices[0]`` is the source and each later vertex consumes the one before it — and
:func:`linear_graph` is the one constructor. The frozen ``vertices`` tuple is shaped to take an
explicit edge list later (joins, fan-out) without changing a vertex.

This module imports nothing else in nautilus on purpose. The factory it stores returns an *operator*
(a :class:`~nautilus.core.operator.SourceOperator` or
:class:`~nautilus.core.operator.OneInputOperator`), but the IR never names those types — it treats a
factory as an opaque ``() -> object`` so the value layer cannot reach down into the runtime. The
compiler, which does know those types, calls the factory; the IR only carries it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: A vertex's operator constructor: a side-effect-free, zero-argument callable returning a fresh
#: operator. It must build a *new* instance each call (resource acquisition belongs in ``open()``),
#: because the compiler calls it to read the operator class name — and again, for a parallel vertex, to
#: check it returns a fresh instance — and the executor calls it once per parallel instance. A factory
#: that returns one shared instance is only safe at parallelism 1.
VertexFactory = Callable[[], object]

#: The kinds of vertex Stage 2 supports. The sink is synthesized by the compiler, never authored here.
_SOURCE = "source"
_ONE_INPUT = "one_input"
_KINDS = frozenset({_SOURCE, _ONE_INPUT})


@dataclass(frozen=True, slots=True)
class LogicalVertex:
    """One operator in a :class:`LogicalGraph`.

    ``id`` is a stable logical handle (an explicit edge list will reference it later); the compiler
    derives the *physical* operator id from topological position, so two graphs that differ only in
    vertex ids compile to the same plan. ``key_columns`` is the source of truth for the keyed shuffle
    feeding this vertex — the columns its input is co-partitioned on, or ``None`` for keyless. How the
    compiler turns that and the parallelism into a partitioner spec is ``compile.lower._spec_for``'s job.
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
        if self.kind == _SOURCE and self.key_columns is not None:
            # A source has no input edge to shuffle, so the compiler would silently ignore its
            # key_columns; reject it rather than accept a value that has no effect.
            raise ValueError(
                f"a source vertex has no input to key; key_columns must be None for {self.id!r}"
            )


@dataclass(frozen=True, slots=True)
class LogicalGraph:
    """A dataflow as an ordered tuple of vertices. Linear for Stage 2: ``vertices[0]`` is the sole
    source and each later vertex consumes its predecessor. Build it with :func:`linear_graph`."""

    vertices: tuple[LogicalVertex, ...]

    def __post_init__(self) -> None:
        if not self.vertices:
            raise ValueError("a LogicalGraph needs at least one vertex (the source)")
        ids = [v.id for v in self.vertices]
        if len(set(ids)) != len(ids):
            raise ValueError(f"vertex ids must be unique, got {ids}")
        sources = [v for v in self.vertices if v.kind == _SOURCE]
        if len(sources) != 1 or self.vertices[0].kind != _SOURCE:
            raise ValueError("a linear graph must start with exactly one source vertex")
        if self.vertices[0].parallelism != 1:
            raise ValueError("the source vertex must have parallelism 1")


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


def linear_graph(source_factory: VertexFactory, vertices: Sequence[LogicalVertex]) -> LogicalGraph:
    """Build a linear ``source -> vertices[0] -> ... -> vertices[-1]`` graph.

    ``source_factory`` builds the source operator; it becomes the single ``"source"`` vertex at
    parallelism 1. ``vertices`` are the one-input transforms in order. The sink is *not* a vertex — the
    compiler synthesizes the collecting sink — so a graph holds only the work the user described.
    """
    source = LogicalVertex(id=_SOURCE, factory=source_factory, kind=_SOURCE, parallelism=1)
    return LogicalGraph((source, *vertices))
