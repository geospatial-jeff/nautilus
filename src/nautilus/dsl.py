"""The fluent ``Stream`` DSL — the readable way to build a :class:`~nautilus.api.LogicalGraph`.

A :class:`Stream` is an immutable handle on a dataflow under construction: every combinator returns a
*new* Stream that adds one operator, so a Stream value is reusable and side-effect-free. :func:`source`
starts one; per-batch verbs (``.map`` / ``.filter`` / ``.tokenize``), column reshaping (``.select`` /
``.drop`` / ``.rename`` / ``.with_column``), keyed aggregation (``.count_by`` / ``.agg_by``), and
``.apply`` extend it; ``.join`` and ``.union`` combine two; and ``.run`` / ``.collect`` execute it. The
same graph runs in one process or across workers — ``.run(workers=…)`` is the only thing that changes.

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
from nautilus.core.operator import (
    AsyncOneInputOperator,
    AsyncSink,
    OneInputOperator,
    SourceOperator,
)
from nautilus.operators import (
    AsyncMapBatch,
    FilterRows,
    HashJoin,
    KeyedAgg,
    KeyedCount,
    MapBatch,
    Tokenize,
    Union,
    from_batches,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nautilus.driver.result import RunResult

_Keys = str | Sequence[str]


def _norm(columns: _Keys) -> tuple[str, ...]:
    return (columns,) if isinstance(columns, str) else tuple(columns)


def _reshape_cols(verb: str, columns: tuple[str, ...]) -> list[str]:
    """Build-time validation for a ``select``/``drop`` column list: at least one name, none blank, and no
    duplicate (a duplicate would name two output columns the same). A name's *presence* is a run-time
    check, not this one (see the column-reshaping comment)."""
    if not columns:
        raise ValueError(f"{verb}() needs at least one column name")
    if any(not c or not c.strip() for c in columns):
        raise ValueError(f"{verb}(): column names must be non-empty")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{verb}(): duplicate column name in {list(columns)}")
    return list(columns)


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
        kind: str = "one_input",
    ) -> Stream:
        """Append a one-input vertex — synchronous ``one_input`` or awaiting ``async_one_input`` — fed
        from the current tail. Keying rides on the *edge* (this is an explicit-edge graph), so the vertex
        itself carries none."""
        vid = f"v{self._next}"
        vertex = LogicalVertex(vid, factory, kind, parallelism, None)
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

    def agg_by(self, key_cols: _Keys, *, parallelism: int = 1, **aggs: tuple[str, str]) -> Stream:
        """Grouped aggregation (:class:`~nautilus.operators.KeyedAgg`) — ``GROUP BY key_cols`` emitting one
        row per group at end of stream. Each keyword names an output column as ``(input_col, func)`` with
        ``func`` one of ``sum``, ``count``, ``mean``, ``min``, ``max``::

            source(rows).agg_by("lat", mean_c=("temp", "mean"), n=("temp", "count"), hi=("temp", "max"))

        The input is shuffled on ``key_cols`` (one or several), so every row of a key meets on one instance
        when parallel — the same co-partitioning ``count_by`` uses. ``parallelism`` is reserved, so an
        output column cannot be named ``parallelism``."""
        if not isinstance(parallelism, int):
            raise TypeError(
                "agg_by: parallelism= must be an int (did you name an aggregation 'parallelism'? "
                f"it is reserved); got {parallelism!r}"
            )
        if not aggs:
            raise ValueError("agg_by needs at least one aggregation, e.g. mean_c=('temp', 'mean')")
        keys = _norm(key_cols)
        specs = dict(aggs)
        KeyedAgg(keys, specs)  # validate the specs eagerly, at the call site, not deep in the run
        return self._extend(
            lambda: KeyedAgg(keys, specs), key_columns=keys, parallelism=parallelism
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
        deep-copied per subtask, so it must be deep-copyable.

        Because a parallelism-1 ``apply`` shares the single given instance, a later uniform
        ``run(parallelism=…)`` / ``to_graph(parallelism=…)`` cannot scale *this* vertex up (the shared
        instance cannot be replicated). To run an applied operator in parallel, pass its ``parallelism``
        here. The verb combinators build a fresh operator per subtask, so they have no such limit.
        """
        keys = _norm(key_columns) if key_columns is not None else operator.key_columns()
        factory: Callable[[], object] = (
            (lambda: operator) if parallelism == 1 else (lambda: copy.deepcopy(operator))
        )
        return self._extend(factory, key_columns=keys, parallelism=parallelism)

    # --- column reshaping -----------------------------------------------------------------------
    # Vectorized batch -> batch column ops, each lowered to a MapBatch. They carry no state, so a
    # parallelism>1 instance can share the one stateless function; column names resolve at run time
    # against each batch's schema (learned from the first batch), so a name absent from the actual data
    # raises then, not at build time — the same contract .map's opaque function already has.

    def select(self, *columns: str, parallelism: int = 1) -> Stream:
        """Keep only ``columns``, in the given order, dropping the rest (Arrow ``RecordBatch.select``)."""
        cols = _reshape_cols("select", columns)
        return self._extend(
            lambda: MapBatch(lambda b: b.select(cols)), key_columns=None, parallelism=parallelism
        )

    def drop(self, *columns: str, parallelism: int = 1) -> Stream:
        """Drop ``columns``, keeping the rest in their original order (Arrow ``RecordBatch.drop_columns``)."""
        cols = _reshape_cols("drop", columns)
        return self._extend(
            lambda: MapBatch(lambda b: b.drop_columns(cols)),
            key_columns=None,
            parallelism=parallelism,
        )

    def rename(self, mapping: dict[str, str], *, parallelism: int = 1) -> Stream:
        """Rename columns by an ``{old: new}`` mapping; unnamed columns and column order are unchanged. At
        run time every ``old`` must be present (a missing one raises), and the result must have no duplicate
        names (renaming onto an existing column raises)."""
        if not mapping:
            raise ValueError("rename() needs a non-empty {old: new} mapping")
        if any(not k or not k.strip() or not v or not v.strip() for k, v in mapping.items()):
            raise ValueError("rename(): column names must be non-empty")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"rename(): two columns renamed to the same name in {mapping}")
        table = dict(mapping)

        def _rename(b: pa.RecordBatch) -> pa.RecordBatch:
            missing = [c for c in table if c not in b.schema.names]
            if missing:
                raise KeyError(f"rename(): column(s) {missing} not in schema {b.schema.names}")
            names = [table.get(n, n) for n in b.schema.names]
            if len(set(names)) != len(names):
                raise ValueError(f"rename(): produces duplicate column names {names}")
            return b.rename_columns(names)

        return self._extend(lambda: MapBatch(_rename), key_columns=None, parallelism=parallelism)

    def with_column(
        self, name: str, fn: Callable[[pa.RecordBatch], pa.Array], *, parallelism: int = 1
    ) -> Stream:
        """Add a column ``name`` computed by ``fn(batch) -> Array`` (one value per row), replacing it if it
        already exists and leaving the other columns unchanged."""
        if not name or not name.strip():
            raise ValueError("with_column(): name must be a non-empty column name")

        def _with(b: pa.RecordBatch) -> pa.RecordBatch:
            col = fn(b)
            i = b.schema.get_field_index(name)
            return b.set_column(i, name, col) if i >= 0 else b.append_column(name, col)

        return self._extend(lambda: MapBatch(_with), key_columns=None, parallelism=parallelism)

    # --- async one-input combinators ------------------------------------------------------------

    def map_async(
        self,
        fn: Callable[[pa.RecordBatch], Awaitable[pa.RecordBatch]],
        *,
        max_in_flight: int = 8,
        ordered: bool = True,
        parallelism: int = 1,
    ) -> Stream:
        """Apply an async ``batch -> batch`` function with overlapping I/O
        (:class:`~nautilus.operators.AsyncMapBatch`): up to ``max_in_flight`` calls run at once. The
        stateless async map — enrich a batch from an awaited lookup without crowding the I/O into the
        source. Keyless, so a parallel run spreads batches across instances to fan the I/O out.

        ``ordered=False`` emits in completion order instead of input order (lower latency), sound here
        because the map is stateless (see
        :meth:`~nautilus.core.operator.AsyncOneInputOperator.ordered`)."""
        return self._extend(
            lambda: AsyncMapBatch(fn, max_in_flight=max_in_flight, ordered=ordered),
            key_columns=None,
            parallelism=parallelism,
            kind="async_one_input",
        )

    def apply_async(
        self,
        operator: AsyncOneInputOperator,
        *,
        key_columns: _Keys | None = None,
        parallelism: int = 1,
    ) -> Stream:
        """Append an arbitrary async one-input operator — the escape hatch mirroring :meth:`apply` for an
        :class:`~nautilus.core.operator.AsyncOneInputOperator` (e.g. a keyed async enrich that folds each
        lookup into state in ``integrate``). The edge is keyed by ``key_columns`` if given, else the
        operator's own :meth:`~nautilus.core.operator.AsyncOneInputOperator.key_columns`, so a keyed async
        stage co-partitions and its per-key state is never split. At parallelism > 1 the instance is
        deep-copied per subtask (acquire its client in ``open()``, not ``__init__``); a parallelism-1
        ``apply_async`` shares the one instance, so scale it by passing ``parallelism`` here.

        A keyed operator (one that declares ``key_columns``) must stay ordered: ``ordered()=False`` is
        stateless-only, so it is rejected here for a keyed operator (:meth:`map_async` is the unordered
        path; why: :meth:`~nautilus.core.operator.AsyncOneInputOperator.ordered`)."""
        if not operator.ordered() and operator.key_columns() is not None:
            raise ValueError(
                f"apply_async: {type(operator).__name__} declares key_columns()="
                f"{operator.key_columns()!r} (keyed state) but ordered()=False; unordered emission is "
                "stateless-only, so a keyed async stage must stay ordered for a reproducible digest"
            )
        keys = _norm(key_columns) if key_columns is not None else operator.key_columns()
        factory: Callable[[], object] = (
            (lambda: operator) if parallelism == 1 else (lambda: copy.deepcopy(operator))
        )
        return self._extend(
            factory, key_columns=keys, parallelism=parallelism, kind="async_one_input"
        )

    # --- two-input combinator -------------------------------------------------------------------

    def join(
        self,
        other: Stream,
        *,
        on: _Keys | None = None,
        left_on: _Keys | None = None,
        right_on: _Keys | None = None,
        how: str = "inner",
        parallelism: int = 1,
    ) -> Stream:
        """Equi-join with ``other`` (:class:`~nautilus.operators.HashJoin`). Give ``on`` for a shared column
        name, or ``left_on``/``right_on`` for differently-named keys; both sides must name the same number
        of columns. ``how`` is ``"inner"`` (default), ``"left"``, ``"right"``, or ``"outer"`` — an outer
        join also keeps the unmatched rows on that side, with the other side's columns null. The two inputs
        are shuffled on their join keys so equal keys meet on one instance. The output is this stream's
        columns followed by ``other``'s non-key columns.

        An inner join runs at any ``parallelism``; an outer join runs at ``parallelism=1`` only (see
        :class:`~nautilus.operators.HashJoin` for why), so a ``how`` other than ``"inner"`` with
        ``parallelism > 1`` is rejected here.

        Build each side from its own :func:`source`. If both inputs derive from one source (a diamond),
        that source is read once per branch, so it must be replayable — the built-in sources are."""
        if other is self:
            raise ValueError(
                "a stream cannot be joined to itself; build the two inputs as separate streams"
            )
        if how != "inner" and parallelism > 1:
            raise ValueError(
                f"an outer join (how={how!r}) runs at parallelism 1 only, got parallelism={parallelism}; "
                "use how='inner' to parallelize the join, or set parallelism=1"
            )
        left_keys, right_keys = _join_keys(on, left_on, right_on)
        return self._combine(
            other,
            lambda: HashJoin(left_keys, right_keys, how),
            left_keys=left_keys,
            right_keys=right_keys,
            parallelism=parallelism,
            copartitioned=True,
        )

    def union(self, other: Stream, *, parallelism: int = 1) -> Stream:
        """Concatenate ``other`` onto this stream (:class:`~nautilus.operators.Union`, SQL ``UNION ALL``):
        every row from both inputs, duplicates kept, no key. The two streams must share a schema, checked
        at run time (nautilus learns schemas from the data). Unlike ``.join`` the inputs are not shuffled —
        each side's batches flow straight through — so it holds no state and runs at any ``parallelism``.

        Build each side from its own :func:`source`. If both inputs derive from one source (a diamond),
        that source is read once per branch, so it must be replayable — the built-in sources are."""
        if other is self:
            raise ValueError(
                "a stream cannot be unioned with itself; build the two inputs as separate streams"
            )
        return self._combine(
            other,
            Union,
            left_keys=None,
            right_keys=None,
            parallelism=parallelism,
            copartitioned=False,
        )

    def _combine(
        self,
        other: Stream,
        factory: Callable[[], object],
        *,
        left_keys: tuple[str, ...] | None,
        right_keys: tuple[str, ...] | None,
        parallelism: int,
        copartitioned: bool,
    ) -> Stream:
        """Splice ``other``'s subgraph into this one and append a two-input vertex fed by this stream's
        tail (left, port 0) and ``other``'s tail (right, port 1). ``other``'s vertex ids are shifted past
        this stream's so all ids stay unique. The two edges carry ``left_keys``/``right_keys`` — the key
        columns a join co-partitions on, or ``None`` for a keyless merge like ``union`` (``copartitioned``
        then says which it is, so a parallel union is not mistaken for an unkeyed join)."""
        shift = len(self._vertices)
        remap = {f"v{i}": f"v{shift + i}" for i in range(len(other._vertices))}
        other_vertices = tuple(replace(v, id=remap[v.id]) for v in other._vertices)
        other_edges = tuple(replace(e, src=remap[e.src], dst=remap[e.dst]) for e in other._edges)
        vid = f"v{shift + len(other._vertices)}"
        vertex = LogicalVertex(vid, factory, "two_input", parallelism, None, copartitioned)
        vertices = (*self._vertices, *other_vertices, vertex)
        edges = (
            *self._edges,
            *other_edges,
            LogicalEdge(self._tail, vid, 0, left_keys),
            LogicalEdge(remap[other._tail], vid, 1, right_keys),
        )
        return Stream(vertices, edges, vid, shift + len(other._vertices) + 1)

    # --- sink terminal --------------------------------------------------------------------------

    def sink(
        self,
        sink: AsyncSink,
        *,
        key_columns: _Keys | None = None,
        parallelism: int = 1,
    ) -> SinkHandle:
        """Write this stream to an external store through an :class:`~nautilus.core.operator.AsyncSink`,
        returning a :class:`SinkHandle` to run it. A sink is a terminal — it has no output — so a
        ``SinkHandle`` exposes only the runners (:meth:`SinkHandle.run` / :meth:`SinkHandle.run_async`),
        never the combinators; the resulting ``RunResult`` carries the telemetry report and no batches
        (the data went to the store).

        The edge is keyed by ``key_columns`` if given, else the sink's own
        :meth:`~nautilus.core.operator.AsyncSink.key_columns`, so a keyed sink co-partitions for per-key
        writes. At ``parallelism > 1`` the sink is deep-copied per subtask (acquire its client in
        ``open()``, not ``__init__``, so the copy carries no live connection); a parallelism-1 ``sink``
        shares the one instance, so scale it by passing ``parallelism`` here rather than to ``run``.
        """
        keys = _norm(key_columns) if key_columns is not None else sink.key_columns()
        factory: Callable[[], object] = (
            (lambda: sink) if parallelism == 1 else (lambda: copy.deepcopy(sink))
        )
        vid = f"v{self._next}"
        vertex = LogicalVertex(vid, factory, "async_sink", parallelism, None)
        edge = LogicalEdge(self._tail, vid, 0, keys)
        return SinkHandle((*self._vertices, vertex), (*self._edges, edge))

    # --- terminal -------------------------------------------------------------------------------

    def to_graph(self, *, parallelism: int | None = None) -> LogicalGraph:
        """The :class:`~nautilus.api.LogicalGraph` this stream describes. ``parallelism`` overrides every
        non-source vertex's parallelism uniformly (the simple scale-up knob); omit it to keep each
        vertex's own. A vertex added via :meth:`apply` at parallelism 1 shares one instance and cannot be
        scaled this way — give that ``apply`` its own ``parallelism`` instead."""
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
        from nautilus.driver.run import run_plan

        return await run_plan(
            self.to_graph(parallelism=parallelism), key_groups=key_groups, **kwargs
        )

    def run(
        self,
        *,
        workers: int | None = None,
        parallelism: int | None = None,
        key_groups: int | None = None,
        daemons: list[tuple[str, int]] | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Compile and run this stream to completion, returning its :class:`RunResult`. ``workers`` > 1
        deploys it across that many spawned worker processes (the *same* graph), capped at the plan's
        maximum operator parallelism (a wider value would only spawn idle workers); ``daemons`` (a
        ``[(host, port), …]`` roster) deploys it across long-lived worker daemons instead — the multi-node
        path, with the worker count taken from the roster. ``parallelism`` sets every operator's instance
        count; ``key_groups`` sets the keyed-shuffle rescale ceiling. A synchronous one-liner — inside a
        running event loop use :meth:`run_async` (single-process) instead."""
        graph = self.to_graph(parallelism=parallelism)
        if daemons is not None:
            from nautilus.cluster import deploy

            return deploy(graph, daemons=daemons, key_groups=key_groups, **kwargs)
        if workers is not None and workers > 1:
            from nautilus.cluster import deploy

            return deploy(graph, num_workers=workers, key_groups=key_groups, **kwargs)
        import asyncio

        from nautilus.driver.run import run_plan

        return asyncio.run(run_plan(graph, key_groups=key_groups, **kwargs))

    def collect(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Run the stream and return its rows as ``{column: value}`` dicts (a convenience over
        ``run().to_pylist()``)."""
        return self.run(**kwargs).to_pylist()


@dataclass(frozen=True, slots=True)
class SinkHandle:
    """A stream that ends in an :class:`~nautilus.core.operator.AsyncSink` — a terminal. It exposes only
    the runners, because a sink has no output to chain off; build it with :meth:`Stream.sink`."""

    _vertices: tuple[LogicalVertex, ...]
    _edges: tuple[LogicalEdge, ...]

    def to_graph(self, *, parallelism: int | None = None) -> LogicalGraph:
        """The :class:`~nautilus.api.LogicalGraph` this sink pipeline describes. ``parallelism`` overrides
        every non-source vertex's parallelism uniformly; a parallelism-1 ``sink`` shares one instance and
        cannot be scaled this way (give that ``sink`` its own ``parallelism`` instead)."""
        if parallelism is None:
            return LogicalGraph(self._vertices, self._edges)
        vertices = tuple(
            v if v.kind == "source" else replace(v, parallelism=parallelism) for v in self._vertices
        )
        return LogicalGraph(vertices, self._edges)

    async def run_async(
        self, *, parallelism: int | None = None, key_groups: int | None = None, **kwargs: Any
    ) -> RunResult:
        """Compile and run this sink pipeline in the current event loop, single-process. The returned
        :class:`~nautilus.driver.result.RunResult` carries the telemetry report and no batches."""
        from nautilus.driver.run import run_plan

        return await run_plan(
            self.to_graph(parallelism=parallelism), key_groups=key_groups, **kwargs
        )

    def run(
        self,
        *,
        workers: int | None = None,
        parallelism: int | None = None,
        key_groups: int | None = None,
        daemons: list[tuple[str, int]] | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Compile and run this sink pipeline to completion, returning its :class:`RunResult` (telemetry
        only; no batches). ``workers``/``daemons`` deploy the same graph across worker processes / daemons
        exactly as :meth:`Stream.run`; a synchronous one-liner — inside a running event loop use
        :meth:`run_async`."""
        graph = self.to_graph(parallelism=parallelism)
        if daemons is not None:
            from nautilus.cluster import deploy

            return deploy(graph, daemons=daemons, key_groups=key_groups, **kwargs)
        if workers is not None and workers > 1:
            from nautilus.cluster import deploy

            return deploy(graph, num_workers=workers, key_groups=key_groups, **kwargs)
        import asyncio

        from nautilus.driver.run import run_plan

        return asyncio.run(run_plan(graph, key_groups=key_groups, **kwargs))


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
