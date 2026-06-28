"""The fluent ``Stream`` DSL — the readable way to build a :class:`~nautilus.api.LogicalGraph`.

A :class:`Stream` is an immutable handle on a dataflow under construction: every combinator returns a
*new* Stream that adds one operator, so a Stream value is reusable and side-effect-free. :func:`source`
starts one; ``.map`` / ``.filter`` / ``.tokenize`` / ``.count_by`` / ``.tumbling_sum`` / ``.apply`` extend
it; ``.join`` combines two; and ``.run`` / ``.collect`` execute it. The same graph runs in one process or
across workers — ``.run(workers=N)`` is the only thing that changes.

This is a value layer that sits *above* :mod:`nautilus.api`: it knows the concrete operators (so it can
build them for you) but it only ever produces a :class:`~nautilus.api.LogicalGraph`. The runners live at
the boundary, so the terminal imports them lazily — building a Stream never pulls the execution engine or
the telemetry-report layer onto the path.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from nautilus.api import LogicalEdge, LogicalGraph, LogicalVertex
from nautilus.core.operator import OneInputOperator, SourceOperator
from nautilus.operators import (
    FilterRows,
    HashJoin,
    KeyedCount,
    KeyedTumblingSum,
    MapBatch,
    Tokenize,
    from_batches,
)
from nautilus.windows import TumblingEventTimeWindows

if TYPE_CHECKING:
    from collections.abc import Callable

    from nautilus.runtime.result import RunResult

_Keys = str | Sequence[str]


def _norm(columns: _Keys) -> tuple[str, ...]:
    return (columns,) if isinstance(columns, str) else tuple(columns)


def source(src: SourceOperator | pa.RecordBatch | Sequence[Any]) -> Stream:
    """Start a stream from a source. ``src`` is a :class:`~nautilus.core.operator.SourceOperator`, a bare
    ``pyarrow.RecordBatch`` (wrapped with a terminal EOS via
    :func:`~nautilus.operators.from_batches`), or a sequence of batches/frames."""
    if isinstance(src, SourceOperator):
        op: SourceOperator = src
    elif isinstance(src, pa.RecordBatch):
        op = from_batches(src)
    elif isinstance(src, Sequence):
        op = from_batches(*src)
    else:
        raise TypeError(
            f"source() takes a SourceOperator, a pyarrow.RecordBatch, or a sequence of batches/frames, "
            f"got {type(src).__name__}"
        )
    vertex = LogicalVertex("v0", lambda: op, "source", 1, None)
    return Stream((vertex,), (), "v0", 1)


@dataclass(frozen=True, slots=True)
class Stream:
    """An immutable dataflow under construction: its vertices, the edges between them, the id of its
    current output (``_tail``), and the next vertex-id counter. Build one with :func:`source` and the
    combinator methods; do not construct it directly."""

    _vertices: tuple[LogicalVertex, ...]
    _edges: tuple[LogicalEdge, ...]
    _tail: str
    _next: int

    # --- one-input combinators ------------------------------------------------------------------

    def _extend(
        self,
        factory: Callable[[], object],
        *,
        key_columns: tuple[str, ...] | None,
        parallelism: int,
    ) -> Stream:
        """Append a one-input vertex fed from the current tail. Keying rides on the *edge* (this is an
        explicit-edge graph), so the vertex itself carries none."""
        vid = f"v{self._next}"
        vertex = LogicalVertex(vid, factory, "one_input", parallelism, None)
        edge = LogicalEdge(self._tail, vid, 0, key_columns)
        return Stream((*self._vertices, vertex), (*self._edges, edge), vid, self._next + 1)

    def map(
        self, fn: Callable[[pa.RecordBatch], pa.RecordBatch], *, parallelism: int = 1
    ) -> Stream:
        """Apply a pure batch -> batch function (:class:`~nautilus.operators.MapBatch`)."""
        return self._extend(lambda: MapBatch(fn), key_columns=None, parallelism=parallelism)

    def filter(
        self, mask_fn: Callable[[pa.RecordBatch], pa.Array], *, parallelism: int = 1
    ) -> Stream:
        """Keep rows where ``mask_fn(batch)`` is true (:class:`~nautilus.operators.FilterRows`)."""
        return self._extend(lambda: FilterRows(mask_fn), key_columns=None, parallelism=parallelism)

    def tokenize(
        self, in_col: str, out_col: str = "word", *, lowercase: bool = True, parallelism: int = 1
    ) -> Stream:
        """Split a string column into one row per whitespace token (:class:`~nautilus.operators.Tokenize`)."""
        return self._extend(
            lambda: Tokenize(in_col, out_col, lowercase), key_columns=None, parallelism=parallelism
        )

    def count_by(self, key_col: str, count_col: str = "count", *, parallelism: int = 1) -> Stream:
        """Count occurrences per key, emitted at end of stream (:class:`~nautilus.operators.KeyedCount`).
        The input is shuffled on ``key_col``, so a key's rows meet on one instance when parallel."""
        return self._extend(
            lambda: KeyedCount(key_col, count_col), key_columns=(key_col,), parallelism=parallelism
        )

    def tumbling_sum(
        self,
        key_col: str,
        value_col: str,
        ts_col: str,
        window: TumblingEventTimeWindows,
        *,
        parallelism: int = 1,
    ) -> Stream:
        """Sum a value column per key per tumbling event-time window
        (:class:`~nautilus.operators.KeyedTumblingSum`), shuffled on ``key_col``."""
        return self._extend(
            lambda: KeyedTumblingSum(key_col, value_col, ts_col, window),
            key_columns=(key_col,),
            parallelism=parallelism,
        )

    def apply(
        self,
        operator: OneInputOperator,
        *,
        key_columns: _Keys | None = None,
        parallelism: int = 1,
    ) -> Stream:
        """Append an arbitrary one-input operator — the escape hatch for operators without a verb. The
        edge is keyed by ``key_columns`` if given, else the operator's own
        :meth:`~nautilus.core.operator.OneInputOperator.key_columns`. At parallelism > 1 the instance is
        deep-copied per subtask, so it must be deep-copyable."""
        keys = _norm(key_columns) if key_columns is not None else operator.key_columns()
        factory: Callable[[], object] = (
            (lambda: operator) if parallelism == 1 else (lambda: copy.deepcopy(operator))
        )
        return self._extend(factory, key_columns=keys, parallelism=parallelism)

    # --- two-input combinator -------------------------------------------------------------------

    def join(
        self,
        other: Stream,
        *,
        on: _Keys | None = None,
        left_on: _Keys | None = None,
        right_on: _Keys | None = None,
        parallelism: int = 1,
    ) -> Stream:
        """Inner equi-join with ``other`` (:class:`~nautilus.operators.HashJoin`). Give ``on`` for a shared
        column name, or ``left_on``/``right_on`` for differently-named keys; both sides must name the same
        number of columns. The two inputs are shuffled on their join keys so equal keys meet on one
        instance. The output is this stream's columns followed by ``other``'s non-key columns."""
        if other is self:
            raise ValueError(
                "a stream cannot be joined to itself; build the two inputs as separate streams"
            )
        left_keys, right_keys = _join_keys(on, left_on, right_on)
        # other's vertex ids are v0..v{m-1}; shift them past this stream's so the union has unique ids.
        shift = len(self._vertices)
        remap = {f"v{i}": f"v{shift + i}" for i in range(len(other._vertices))}
        other_vertices = tuple(replace(v, id=remap[v.id]) for v in other._vertices)
        other_edges = tuple(replace(e, src=remap[e.src], dst=remap[e.dst]) for e in other._edges)
        jid = f"v{shift + len(other._vertices)}"
        jvertex = LogicalVertex(
            jid, lambda: HashJoin(left_keys, right_keys), "two_input", parallelism, None
        )
        vertices = (*self._vertices, *other_vertices, jvertex)
        edges = (
            *self._edges,
            *other_edges,
            LogicalEdge(self._tail, jid, 0, left_keys),
            LogicalEdge(remap[other._tail], jid, 1, right_keys),
        )
        return Stream(vertices, edges, jid, shift + len(other._vertices) + 1)

    # --- terminal -------------------------------------------------------------------------------

    def to_graph(self, *, parallelism: int | None = None) -> LogicalGraph:
        """The :class:`~nautilus.api.LogicalGraph` this stream describes. ``parallelism`` overrides every
        non-source vertex's parallelism uniformly (the simple scale-up knob); omit it to keep each
        vertex's own."""
        if parallelism is None:
            return LogicalGraph(self._vertices, self._edges)
        vertices = tuple(
            v if v.kind == "source" else replace(v, parallelism=parallelism) for v in self._vertices
        )
        return LogicalGraph(vertices, self._edges)

    async def run_async(
        self, *, parallelism: int | None = None, key_groups: int | None = None, **kwargs: Any
    ) -> RunResult:
        """Compile and run this stream in the current event loop, single-process. For a synchronous
        caller or for multiple worker processes use :meth:`run`."""
        from nautilus.runtime.run import run_plan

        return await run_plan(
            self.to_graph(parallelism=parallelism), key_groups=key_groups, **kwargs
        )

    def run(
        self,
        *,
        workers: int | None = None,
        parallelism: int | None = None,
        key_groups: int | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Compile and run this stream to completion, returning its :class:`RunResult`. ``workers`` > 1
        deploys it across that many worker processes (the *same* graph); ``parallelism`` sets every
        operator's instance count; ``key_groups`` sets the keyed-shuffle rescale ceiling. A synchronous
        one-liner — inside a running event loop use :meth:`run_async` (single-process) instead."""
        graph = self.to_graph(parallelism=parallelism)
        if workers is not None and workers > 1:
            from nautilus.cluster import deploy

            return deploy(graph, num_workers=workers, key_groups=key_groups, **kwargs)
        import asyncio

        from nautilus.runtime.run import run_plan

        return asyncio.run(run_plan(graph, key_groups=key_groups, **kwargs))

    def collect(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Run the stream and return its rows as ``{column: value}`` dicts (a convenience over
        ``run().to_pylist()``)."""
        return self.run(**kwargs).to_pylist()


def _join_keys(
    on: _Keys | None, left_on: _Keys | None, right_on: _Keys | None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if on is not None:
        if left_on is not None or right_on is not None:
            raise ValueError("give either on= or left_on=/right_on=, not both")
        left = right = on
    elif left_on is not None and right_on is not None:
        left, right = left_on, right_on
    else:
        raise ValueError("a join needs on= (shared name) or both left_on= and right_on=")
    left_keys, right_keys = _norm(left), _norm(right)
    if len(left_keys) != len(right_keys):
        raise ValueError(
            f"left_on {left_keys} and right_on {right_keys} must name the same number of columns"
        )
    return left_keys, right_keys
