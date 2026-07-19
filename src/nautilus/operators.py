"""The built-in operators — the implementations behind the fluent ``Stream`` combinators.

Concrete operators that exercise the streaming semantics. Most follow the synchronous
``process``/``on_eos`` contract (emit into the ``Collector``, never await; see
:mod:`nautilus.core.operator`): :class:`MapBatch`, :class:`FilterRows`, :class:`Tokenize`, and
:class:`KeyedCount` back the DSL's ``.map`` / ``.filter`` / ``.tokenize`` / ``.count_by``,
:class:`KeyedAgg` backs ``.agg_by`` (grouped aggregation), and :class:`HashJoin` backs ``.join``.
:class:`KeyedMean` is the specialized single-key mean, applied through ``.apply``. :class:`AsyncMapBatch`
is the one awaiting built-in — it
backs ``.map_async``, doing its I/O in ``fetch`` and emitting in ``integrate``. What each one does is on
its own class.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nautilus.core.operator import (
    AsyncOneInputOperator,
    Collector,
    OneInputOperator,
    OperatorContext,
    SourceOperator,
    TwoInputOperator,
)
from nautilus.core.records import EOS_FRAME, Batch, Frame
from nautilus.state import KeyContext


def _add(a: int, b: int) -> int:
    return a + b


#: Ceiling on a value-indexed fast-path array: ~268M entries, ~2 GB as int64. Sizing an accumulator to
#: ``max(key) + 1`` past this costs more memory than the fast path saves. Generous — dense category-id keys
#: sit far below it; only a sparse or hashed integer id reaches it.
_FAST_PATH_MAX = 1 << 28


def _int_fast_ok(keys: np.ndarray) -> bool:
    """Whether an integer key batch may use a value-indexed fast path that sizes an accumulator to the key
    value — the ``np.bincount`` sum/count arrays in :class:`KeyedMean` and :class:`KeyedAgg`, the dense
    value→id map in :class:`HashJoin`. The keys must be non-negative and dense enough that an array sized to
    ``max(key) + 1`` stays under :data:`_FAST_PATH_MAX`. If either fails, the operator demotes to the general
    per-distinct-key path, which handles any key — one gate closing both the negative-key crash and the
    sparse-key OOM for all three callers. (:class:`KeyedCount` guards itself differently — its docstring.)
    """
    if keys.size == 0:
        return True
    return bool(keys.min() >= 0) and int(keys.max()) < _FAST_PATH_MAX


class InMemorySource(SourceOperator):
    """Yields a fixed, pre-built sequence of frames; used by deterministic tests. A bounded
    source must end its frame list with ``EOS_FRAME``; :func:`from_batches` appends it for you."""

    def __init__(self, frames: list[Frame]) -> None:
        for frame in frames:  # fail loudly at construction, not by silently vanishing in the actor
            if not isinstance(frame, Frame):
                raise TypeError(
                    f"InMemorySource frames must be Frame objects (Batch/EOS/...), got "
                    f"{type(frame).__name__}"
                )
        self._frames = frames

    async def frames(self) -> AsyncIterator[Frame]:
        for frame in self._frames:
            yield frame


def from_batches(*frames: Frame | pa.RecordBatch) -> InMemorySource:
    """Build a bounded :class:`InMemorySource` from data, appending the terminal ``EOS_FRAME`` for you
    (omitting it yields a source that never signals completion). A bare ``pyarrow.RecordBatch`` is
    wrapped in a :class:`~nautilus.core.records.Batch` for you, so ``from_batches(pa.record_batch(...))``
    just works. Use ``InMemorySource([...])`` directly when a test needs exact frame control — to omit
    EOS, or to place an EOS non-terminally."""
    out: list[Frame] = []
    for frame in frames:
        if isinstance(frame, Frame):
            out.append(frame)
        elif isinstance(frame, pa.RecordBatch):
            out.append(Batch(frame))
        elif isinstance(frame, pa.Table):
            raise TypeError(
                "from_batches takes RecordBatches, not a Table; pass *table.to_batches()"
            )
        else:
            raise TypeError(
                f"from_batches accepts nautilus Frame objects or a pyarrow.RecordBatch, got "
                f"{type(frame).__name__}"
            )
    return InMemorySource([*out, EOS_FRAME])


class MapBatch(OneInputOperator):
    """Applies a pure batch -> batch function."""

    def __init__(self, fn: Callable[[pa.RecordBatch], pa.RecordBatch]) -> None:
        self._fn = fn

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        out.emit(self._fn(batch))


class AsyncMapBatch(AsyncOneInputOperator):
    """Applies an async batch -> batch function: :meth:`fetch` awaits ``fn(batch)`` (the I/O) and
    :meth:`integrate` emits its result. The stateless async enrich/lookup built-in — one batch out per
    batch in — behind the DSL's ``.map_async``. Being stateless, it may run ``ordered=False``
    (completion-order emission, lower latency); ``ordered`` defaults ``True``."""

    def __init__(
        self,
        fn: Callable[[pa.RecordBatch], Awaitable[pa.RecordBatch]],
        *,
        max_in_flight: int = 8,
        ordered: bool = True,
    ) -> None:
        self._fn = fn
        self._cap = max_in_flight
        self._ordered = ordered

    async def fetch(self, batch: pa.RecordBatch) -> object:
        return await self._fn(batch)

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        out.emit(cast(pa.RecordBatch, result))

    def max_in_flight(self) -> int:
        return self._cap

    def ordered(self) -> bool:
        return self._ordered


class FilterRows(OneInputOperator):
    """Keeps rows where ``mask_fn(batch)`` (a boolean Arrow array) is true."""

    def __init__(self, mask_fn: Callable[[pa.RecordBatch], pa.Array]) -> None:
        self._mask_fn = mask_fn

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        out.emit(batch.filter(self._mask_fn(batch)))


class Tokenize(OneInputOperator):
    """Splits a string column into one row per whitespace-separated token."""

    def __init__(self, in_col: str, out_col: str = "word", lowercase: bool = True) -> None:
        self.in_col = in_col
        self.out_col = out_col
        self.lowercase = lowercase

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        # Per-row str.split(). The columnar form (pc.utf8_split_whitespace -> pc.list_flatten, then a
        # filter dropping the empty-string tokens it emits at whitespace runs/ends so the result matches
        # str.split()) is correct in isolation but corrupted nondeterministically under full-suite load —
        # the raw split was right and a retry passed, so it was not root-caused (likely a flattened-view
        # buffer-lifetime issue). A streaming engine cannot ship a nondeterministic tokenizer, so this
        # stays per-row and exact; the keyed shuffle is the measured hot path anyway.
        words: list[str] = []
        for s in batch.column(self.in_col).to_pylist():
            if s:
                words.extend((s.lower() if self.lowercase else s).split())
        if words:
            arr = pa.array(words, pa.string())
            out.emit(pa.RecordBatch.from_arrays([arr], names=[self.out_col]))


class KeyedCount(OneInputOperator):
    """Counts occurrences per key, emitted at end of stream (:meth:`on_eos`) — a keyed global aggregation.

    A non-negative integer key is counted in a numpy array indexed by the key value (``np.bincount`` per
    batch, added into a running array). A negative integer key — and any non-integer key — folds through
    keyed state instead, which counts any key. A *sparse* non-negative integer key is the one unguarded
    case, a deliberate speed choice: ``np.bincount`` sizes its array to the
    largest key value, so
    counting by a high-valued id allocates a proportionally huge array and can exhaust memory. Guarding it
    needs a max scan on every batch, which every dense count would pay for a rare key — so instead, key such
    a column through a dense or hashed id, or aggregate it with :class:`KeyedAgg` (``.agg_by``), which
    demotes a sparse key. Either way a null key is counted as its own group."""

    _STATE = "count"  # state-backend name (distinct from the output column, which count_col names)

    def __init__(self, key_col: str, count_col: str = "count") -> None:
        self.key_col = key_col
        self.count_col = count_col

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx
        self._key_type: pa.DataType | None = None  # input type, kept on the output
        self._counts: np.ndarray | None = None  # integer fast-path count accumulator
        self._nulls = 0  # null-key count (fast path; dict path folds null normally)

    def key_columns(self) -> tuple[str, ...]:
        return (self.key_col,)

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        col = batch.column(self.key_col)
        if self._key_type is None:
            self._key_type = col.type
            if pa.types.is_integer(col.type):
                self._counts = np.zeros(0, dtype=np.int64)
        if self._counts is not None:
            nn = col.drop_null() if col.null_count else col
            keys = np.asarray(nn.to_numpy(zero_copy_only=False))
            # np.bincount raises ValueError on a negative key (numpy rejects a negative index); catching
            # that demotes with no pre-scan on the hot path. A sparse key it does not guard — the deliberate
            # limitation in the class docstring.
            try:
                if keys.size:
                    bc = np.bincount(keys)
                    if bc.size > self._counts.size:
                        grown = np.zeros(bc.size, dtype=np.int64)
                        grown[: self._counts.size] = self._counts
                        self._counts = grown
                    self._counts[: bc.size] += bc
                self._nulls += col.null_count
                return
            except ValueError:
                pass  # negative key
            self._demote()
        counts = pc.value_counts(col)
        # One bulk fold of the batch's per-key counts — no KeyContext or reducing-state handle per key.
        key_tuples = ((v,) for v in counts.field("values").to_pylist())
        self._ctx.reduce_all(
            self._STATE, zip(key_tuples, counts.field("counts").to_pylist(), strict=True), _add
        )

    def _demote(self) -> None:
        # Drain the running bincount + null tally into the general keyed-state path, which counts any key,
        # then process this and every later batch there (no fast-path array to overflow).
        if self._counts is not None:
            nz = np.nonzero(self._counts)[0]
            items: list[tuple[tuple[object, ...], int]] = [
                ((int(k),), int(self._counts[k])) for k in nz
            ]
            if self._nulls:
                items.append(((None,), self._nulls))
            if items:
                self._ctx.reduce_all(self._STATE, items, _add)
        self._counts = None
        self._nulls = 0

    def on_eos(self, out: Collector) -> None:
        if self._counts is not None:  # integer fast path: emit nonzero keys + counts, vectorized
            nz = np.nonzero(self._counts)[0]
            if nz.size or self._nulls:
                key_arr = pa.array(nz, self._key_type)
                cnt_arr = pa.array(self._counts[nz], pa.int64())
                if self._nulls:  # append the null-key group (rare, a small concat)
                    key_arr = pa.concat_arrays([key_arr, pa.array([None], self._key_type)])
                    cnt_arr = pa.concat_arrays([cnt_arr, pa.array([self._nulls], pa.int64())])
                out.emit(
                    pa.RecordBatch.from_arrays(
                        [key_arr, cnt_arr], names=[self.key_col, self.count_col]
                    )
                )
            return
        keys: list[object] = []
        totals: list[int] = []
        fired: list[KeyContext] = []
        for kctx, value in self._ctx.entries(self._STATE):  # collect first, then clear (no mutation
            keys.append(kctx.key[0])  # during iteration, so the backend can stream entries lazily)
            totals.append(cast(int, value))
            fired.append(kctx)
        for kctx in fired:
            self._ctx.clear_state(self._STATE, kctx)
        if keys:
            out.emit(
                pa.RecordBatch.from_arrays(
                    [pa.array(keys, self._key_type), pa.array(totals, pa.int64())],
                    names=[self.key_col, self.count_col],
                )
            )


class KeyedMean(OneInputOperator):
    """``AVG(value) GROUP BY key`` — the mean of a value column per key, with the contributing count ``n``,
    emitted at end of stream. The specialized single-key mean; it agrees with
    ``.agg_by(key, mean_col=(value, "mean"), n=(value, "count"))`` but keeps a tuned bincount loop.

    For non-negative integer keys it folds each batch into running per-key ``sum`` and ``count`` numpy
    arrays (``np.bincount``) with no per-key Python — the vectorized fast path. Semantics match SQL
    ``AVG(col)``: a null value is
    skipped, and a group whose values are *all* null keeps its row with a ``NULL`` mean and ``0`` count (a
    *NaN* value, unlike a null, propagates to ``NaN``, as DataFusion ``AVG`` also yields). A null key, or an
    integer key the value-indexed path can't take (:func:`_int_fast_ok`), drains the accumulators into an
    Arrow group-by that handles any key. Emits one row per key: ``key_col``, ``mean_col``, and ``n``.
    """

    def __init__(self, key_col: str, value_col: str, mean_col: str = "mean") -> None:
        self.key_col = key_col
        self.value_col = value_col
        self.mean_col = mean_col

    def open(self, ctx: OperatorContext) -> None:
        self._key_type: pa.DataType | None = None
        self._fast: bool | None = (
            None  # None until the first batch; False once demoted to the group-by
        )
        self._sum: np.ndarray | None = None  # per-key running sum (float64)
        self._cnt: np.ndarray | None = None  # per-key non-null value count (int64)
        self._present: np.ndarray | None = (
            None  # per-key all-rows count; created only when a null appears
        )
        self._dict: dict[object, list[Any]] = (
            {}
        )  # group-by state (non-int/demoted): key -> [sum, n]

    def key_columns(self) -> tuple[str, ...]:
        return (self.key_col,)

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        kcol = batch.column(self.key_col)
        if self._fast is None:
            self._key_type = kcol.type
            self._fast = pa.types.is_integer(kcol.type)
        if self._fast:
            if kcol.null_count == 0:
                keys = np.asarray(kcol.to_numpy(zero_copy_only=False))
                if _int_fast_ok(keys):
                    self._fast_add(keys, batch.column(self.value_col))
                    return
            self._demote()  # a null key, or one _int_fast_ok rejects: the group-by handles any key
        self._dict_add(kcol, batch.column(self.value_col))

    @staticmethod
    def _grow(arr: np.ndarray | None, top: int, dtype: type) -> np.ndarray:
        if arr is None or arr.size < top:
            grown: np.ndarray = np.zeros(top, dtype=dtype)
            if arr is not None:
                grown[: arr.size] = arr
            return grown
        return arr

    def _fast_add(self, keys: np.ndarray, vcol: pa.Array) -> None:
        # Non-negative integer keys: the two bincounts a plain mean needs. Presence (all rows per key) is
        # tracked only once a null value has appeared — a null can hide a group from the non-null count — so
        # a null-free stream stays on exactly the old fast path with no extra work.
        top = int(keys.max()) + 1
        self._sum = self._grow(self._sum, top, np.float64)
        self._cnt = self._grow(self._cnt, top, np.int64)
        if vcol.null_count:
            if (
                self._present is None
            ):  # seed from the count so far (no nulls yet, so count == presence)
                self._present = self._cnt.copy()
            self._present = self._grow(self._present, top, np.int64)
            self._present[:top] += np.bincount(keys, minlength=top)
            valid = vcol.is_valid()
            vkeys = keys[np.asarray(valid)]
            if vkeys.size:
                vals = np.asarray(vcol.filter(valid).to_numpy(zero_copy_only=False), np.float64)
                self._sum[:top] += np.bincount(vkeys, weights=vals, minlength=top)
                self._cnt[:top] += np.bincount(vkeys, minlength=top)
        else:
            vals = np.asarray(vcol.to_numpy(zero_copy_only=False), np.float64)
            c = np.bincount(keys, minlength=top)
            self._cnt[:top] += c
            self._sum[:top] += np.bincount(keys, weights=vals, minlength=top)
            if self._present is not None:  # nulls seen in an earlier batch — keep presence current
                self._present = self._grow(self._present, top, np.int64)
                self._present[:top] += c

    def _demote(self) -> None:
        # Drain the numpy accumulators into the group-by (dict) state, then handle this and every later
        # batch there — it groups any key, including the null/negative one that triggered the demotion.
        sums, cnt = self._sum, self._cnt
        if sums is not None and cnt is not None:
            present = self._present if self._present is not None else cnt
            for k in np.nonzero(present)[0]:
                ki = int(k)
                self._dict[ki] = [float(sums[ki]) if cnt[ki] else None, int(cnt[ki])]
        self._sum = self._cnt = self._present = None
        self._fast = False

    def _dict_add(self, kcol: pa.Array, vcol: pa.Array) -> None:
        agg = (
            pa.table({"k": kcol, "v": vcol}).group_by("k").aggregate([("v", "sum"), ("v", "count")])
        )
        for k, s, c in zip(
            agg.column("k").to_pylist(),
            agg.column("v_sum").to_pylist(),  # None for an all-null group
            agg.column("v_count").to_pylist(),
            strict=True,
        ):
            cur = self._dict.get(k)
            if cur is None:
                self._dict[k] = [s, c]
            else:  # None is the additive identity, so an all-null batch contributes nothing (no crash)
                cur[0] = s if cur[0] is None else cur[0] if s is None else cur[0] + s
                cur[1] += c

    def on_eos(self, out: Collector) -> None:
        if not self._fast:  # demoted, or non-integer keys → group-by state
            if not self._dict:
                return
            keys = list(self._dict)
            means = [s / c if c else None for s, c in (self._dict[k] for k in keys)]
            cnts = [self._dict[k][1] for k in keys]
            out.emit(
                pa.RecordBatch.from_arrays(
                    [
                        pa.array(keys, _key_out_type(self._key_type)),
                        pa.array(means, pa.float64()),
                        pa.array(cnts, pa.int64()),
                    ],
                    names=[self.key_col, self.mean_col, "n"],
                )
            )
            return
        sarr, carr = self._sum, self._cnt
        if sarr is None or carr is None:  # fast path chosen but no rows
            return
        present = self._present if self._present is not None else carr
        nz = np.nonzero(present)[0]
        if nz.size == 0:
            return
        cnt = np.asarray(carr[nz])
        empty = cnt == 0  # a group with rows but no non-null value → NULL mean, like SQL AVG
        mean = np.asarray(sarr[nz]) / np.where(empty, 1, cnt)
        out.emit(
            pa.RecordBatch.from_arrays(
                [
                    pa.array(nz, _key_out_type(self._key_type)),
                    pa.array(mean, pa.float64(), mask=empty),
                    pa.array(cnt, pa.int64()),
                ],
                names=[self.key_col, self.mean_col, "n"],
            )
        )


#: Each aggregation func and the base per-key partials it needs (``mean`` = ``sum`` / ``count``).
_AGG_BASE: dict[str, tuple[str, ...]] = {
    "sum": ("sum",),
    "count": ("count",),
    "mean": ("sum", "count"),
    "min": ("min",),
    "max": ("max",),
}


def _merge_scalar(func: str, a: Any, b: Any) -> Any:
    """Merge two per-key base partials (numbers from Arrow, or ``None`` for an all-null group) on the
    general (dict) path. ``None`` is the identity — an all-null batch contributes nothing, and a group
    that is all-null across every batch stays ``None`` (SQL ``NULL``), never ``0`` and never a crash.
    """
    if a is None:
        return b
    if b is None:
        return a
    if func in ("sum", "count"):
        return a + b
    return min(a, b) if func == "min" else max(a, b)


def _key_out_type(t: pa.DataType) -> pa.DataType:
    """Output type for a key column: int64 for any integer key (so a narrow first-batch key type — int8 —
    cannot overflow when a later batch carries a larger value), else the key's own type."""
    return pa.int64() if pa.types.is_integer(t) else t


class KeyedAgg(OneInputOperator):
    """Grouped aggregation — the vectorized ``GROUP BY`` behind the DSL's ``.agg_by``.

    ``aggs`` maps each output column name to ``(input_col, func)`` with ``func`` one of ``sum``, ``count``,
    ``mean``, ``min``, ``max``; one row per key group is emitted at end of stream. For a single
    non-negative-integer key it bincounts / scatters the raw batch rows directly into running numpy
    accumulators indexed by the key value — no per-key Python and no per-batch group-by, matching the
    :class:`KeyedCount` / :class:`KeyedMean` fast path. Any other key — multi-column, string, or a single
    integer key the value-indexed path can't take (:func:`_int_fast_ok`), including one that first appears
    in a later batch, at which point the numpy accumulators are drained into the dict — reduces each batch by
    an Arrow group-by and folds the per-key partials through a dict. The two paths agree exactly, including
    on nulls: ``count`` / ``mean`` count non-null values of the input column (SQL ``COUNT(col)`` /
    ``AVG(col)``), and a group whose values are *all* null keeps its row but reports ``NULL`` for ``sum`` /
    ``mean`` / ``min`` / ``max`` and ``0`` for ``count``. An integer input column keeps an integer ``sum`` /
    ``min`` / ``max`` (no float widening); ``mean`` is always float64.

    Memory: the fast path's accumulators are dense arrays indexed by the key *value* (one per aggregation
    base), so a large or sparse integer key would size them to ``max(key) + 1``. :func:`_int_fast_ok` keeps
    such a key off the fast path — it demotes to the dict — so the arrays stay bounded. (:class:`KeyedCount`
    makes the opposite tradeoff and does not guard its sparse case; see its docstring.)"""

    def __init__(self, key_cols: tuple[str, ...], aggs: dict[str, tuple[str, str]]) -> None:
        self.key_cols = key_cols
        self.aggs = dict(aggs)
        for out, spec in self.aggs.items():
            if not (
                isinstance(spec, tuple) and len(spec) == 2 and all(isinstance(x, str) for x in spec)
            ):
                raise ValueError(f"agg_by: {out!r} must be (input_col, func), got {spec!r}")
        unknown = {f for _, f in self.aggs.values()} - set(_AGG_BASE)
        if unknown:
            raise ValueError(f"agg_by: unknown func(s) {sorted(unknown)}; use {sorted(_AGG_BASE)}")
        collide = set(self.aggs) & set(self.key_cols)
        if collide:
            raise ValueError(
                f"agg_by: output name(s) {sorted(collide)} collide with a key column; rename"
            )
        # Every aggregated column carries its non-null count too, so an all-null group is emitted as NULL
        # (not dropped, not inf) — group presence is decided from the key (below), never a value count.
        value_bases = [(inc, bf) for inc, f in self.aggs.values() for bf in _AGG_BASE[f]]
        count_bases = [(inc, "count") for inc, _ in self.aggs.values()]
        self._bases: tuple[tuple[str, str], ...] = tuple(
            dict.fromkeys([*value_bases, *count_bases])
        )

    def key_columns(self) -> tuple[str, ...]:
        return self.key_cols

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx
        self._fast: bool | None = (
            None  # single-integer key → numpy accumulators (demotes on a bad key)
        )
        self._acc: dict[tuple[str, str], np.ndarray] = {}  # fast-path per-base accumulators
        self._present: np.ndarray | None = (
            None  # fast-path bool: which keys had any row (group presence)
        )
        self._state: dict[tuple[Any, ...], dict[tuple[str, str], object]] = (
            {}
        )  # general-path partials
        self._key_types: list[pa.DataType] = []
        self._val_types: dict[str, pa.DataType] = (
            {}
        )  # input column → arrow type (for the output dtype)

    def _partial(self, batch: pa.RecordBatch) -> pa.Table:
        cols: dict[str, pa.Array] = {c: batch.column(c) for c in self.key_cols}
        for inc, _ in self._bases:
            cols.setdefault(inc, batch.column(inc))
        return (
            pa.table(cols)
            .group_by(list(self.key_cols))
            .aggregate([(inc, bf) for inc, bf in self._bases])
        )

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        if self._fast is None:
            self._fast = len(self.key_cols) == 1 and pa.types.is_integer(
                batch.column(self.key_cols[0]).type
            )
            self._key_types = [batch.column(c).type for c in self.key_cols]
            for inc, _ in self._bases:
                self._val_types.setdefault(inc, batch.column(inc).type)
        if self._fast:
            kcol = batch.column(self.key_cols[0])
            if kcol.null_count == 0:
                keys = np.asarray(kcol.to_numpy(zero_copy_only=False))
                if _int_fast_ok(keys):
                    self._process_fast(batch, keys)
                    return
            self._demote()  # a null or negative key: fall back to the general group-by path
        partial = self._partial(batch)
        if partial.num_rows == 0:
            return
        keytuples = list(zip(*[partial.column(c).to_pylist() for c in self.key_cols], strict=True))
        vals = {b: partial.column(f"{b[0]}_{b[1]}").to_pylist() for b in self._bases}
        for i, kt in enumerate(keytuples):
            cur = self._state.get(kt)
            if cur is None:
                self._state[kt] = {b: vals[b][i] for b in self._bases}
            else:
                for b in self._bases:
                    cur[b] = _merge_scalar(b[1], cur[b], vals[b][i])

    def _process_fast(self, batch: pa.RecordBatch, keys: np.ndarray) -> None:
        # Bincount / scatter the RAW rows directly, indexed by the key value, with no per-batch Arrow
        # group-by — the move that makes KeyedCount/KeyedMean fast. ``_present`` marks which keys had any row
        # (a boolean scatter-assign, cheaper than a count — a group is present even if all its values are
        # null); each base then filters its own input column's nulls (SQL COUNT/SUM/MIN/MAX skip them).
        if keys.size == 0:
            return
        top = int(keys.max()) + 1
        self._present = self._grow(self._present, top, False, np.bool_)
        self._present[keys] = True
        valcache: dict[str, tuple[np.ndarray, np.ndarray]] = (
            {}
        )  # inc → its (valid keys, valid values)
        for base in self._bases:
            inc, bf = base
            vcol = batch.column(inc)
            if (
                bf == "count"
            ):  # counts rows, so it needs the keys, not the values (skip the extraction)
                bkeys = keys if not vcol.null_count else keys[np.asarray(vcol.is_valid())]
                acc = self._acc[base] = self._grow(self._acc.get(base), top, 0, np.int64)
                if bkeys.size:
                    acc[:top] += np.bincount(bkeys, minlength=top)
                continue
            if (
                inc not in valcache
            ):  # extract each value column once, shared across its sum/min/max bases
                if vcol.null_count:
                    valid = vcol.is_valid()
                    valcache[inc] = (
                        keys[np.asarray(valid)],
                        np.asarray(vcol.filter(valid).to_numpy(zero_copy_only=False)),
                    )
                else:
                    valcache[inc] = (keys, np.asarray(vcol.to_numpy(zero_copy_only=False)))
            self._scatter_value(base, *valcache[inc], top)

    @staticmethod
    def _grow(arr: np.ndarray | None, top: int, init: object, dtype: type) -> np.ndarray:
        if arr is None or arr.size < top:
            grown: np.ndarray = np.full(top, init, dtype=dtype)
            if arr is not None:
                grown[: arr.size] = arr
            return grown
        return arr

    def _scatter_value(
        self, base: tuple[str, str], keys: np.ndarray, col: np.ndarray, top: int
    ) -> None:
        inc, bf = base
        if bf == "sum" and pa.types.is_integer(self._val_types[inc]):
            acc = self._acc[base] = self._grow(
                self._acc.get(base), top, 0, np.int64
            )  # exact int sums
            if keys.size:
                np.add.at(acc, keys, col.astype(np.int64, copy=False))
        elif bf == "sum":
            acc = self._acc[base] = self._grow(self._acc.get(base), top, 0.0, np.float64)
            if keys.size:
                acc[:top] += np.bincount(
                    keys, weights=col.astype(np.float64, copy=False), minlength=top
                )
        elif (
            bf == "min"
        ):  # float64 accumulator (±inf seed); empties are nulled via the count, not the seed
            acc = self._acc[base] = self._grow(self._acc.get(base), top, np.inf, np.float64)
            if keys.size:
                np.minimum.at(acc, keys, col.astype(np.float64, copy=False))
        else:  # max
            acc = self._acc[base] = self._grow(self._acc.get(base), top, -np.inf, np.float64)
            if keys.size:
                np.maximum.at(acc, keys, col.astype(np.float64, copy=False))

    def _demote(self) -> None:
        # A null or negative key appeared after the integer fast path was chosen. Drain the numpy
        # accumulators into the general dict state (matching what Arrow group_by would have produced) and
        # continue on the group-by path, which handles any key.
        if self._present is not None:
            for k in np.nonzero(self._present)[0]:
                ki = int(k)
                self._state[(ki,)] = {b: self._acc_value(b, ki) for b in self._bases}
        self._acc = {}
        self._present = None
        self._fast = False

    def _acc_value(self, base: tuple[str, str], k: int) -> object:
        inc, bf = base
        n = int(self._acc[(inc, "count")][k])
        if bf == "count":
            return n
        if n == 0:  # all-null group → NULL partial, as Arrow's aggregate returns
            return None
        v = self._acc[base][k]
        return int(v) if pa.types.is_integer(self._val_types[inc]) else float(v)

    def _emit_col(self, inc: str, func: str, empty: np.ndarray, at: np.ndarray) -> pa.Array:
        """One output column of the fast path: mask all-null groups to NULL and cast to the general path's
        output dtype (integer columns keep an integer sum/min/max; mean is float64)."""
        if func == "count":
            return pa.array(self._acc[(inc, "count")][at])
        if func == "mean":
            cnt = self._acc[(inc, "count")][at]
            means = self._acc[(inc, "sum")][at].astype(np.float64) / np.where(empty, 1, cnt)
            return pa.array(means, mask=empty)
        vals = self._acc[(inc, _AGG_BASE[func][0])][at]
        if pa.types.is_integer(self._val_types[inc]):
            vals = np.where(empty, 0, vals).astype(
                np.int64
            )  # min/max acc is float64; empties are masked
        return pa.array(vals, mask=empty)

    def on_eos(self, out: Collector) -> None:
        if self._fast:
            if self._present is None:
                return
            nz = np.nonzero(self._present)[0]
            if nz.size == 0:
                return
            arrays = [pa.array(nz, _key_out_type(self._key_types[0]))]
            for _out, (inc, func) in self.aggs.items():
                empty = self._acc[(inc, "count")][nz] == 0
                arrays.append(self._emit_col(inc, func, empty, nz))
            out.emit(pa.RecordBatch.from_arrays(arrays, names=[self.key_cols[0], *self.aggs]))
            return
        if not self._state:
            return
        kts = list(self._state)
        arrays = [
            pa.array([kt[i] for kt in kts], _key_out_type(self._key_types[i]))
            for i in range(len(self.key_cols))
        ]
        for _out, (inc, func) in self.aggs.items():
            vals: list[Any]
            if func == "mean":
                vals = [self._mean(self._state[kt], inc) for kt in kts]
            else:
                b = (inc, _AGG_BASE[func][0])
                vals = [self._state[kt][b] for kt in kts]
            arrays.append(pa.array(vals))
        out.emit(pa.RecordBatch.from_arrays(arrays, names=[*self.key_cols, *self.aggs]))

    @staticmethod
    def _mean(partials: dict[tuple[str, str], object], inc: str) -> float | None:
        total, n = partials[(inc, "sum")], partials[(inc, "count")]
        return None if not n or total is None else cast(float, total) / cast(int, n)


class _SideBuffer:
    """One join input's accumulated rows, indexed by integer key id for a vectorized probe.

    Rows are appended a whole batch at a time (O(1) — no per-key concatenation), each batch carried
    alongside an array of its rows' key ids. The grouped index — rows reordered so one key id's rows form
    one contiguous run, plus per-id ``start`` and ``count`` arrays — is built lazily on the first probe
    after a change and cached. So a side that stops growing (the bounded table in a stream-table join) is
    grouped once and reused for every probe, instead of being re-touched per key on every batch."""

    def __init__(self) -> None:
        self._batches: list[pa.RecordBatch] = []
        self._key_ids: list[np.ndarray] = []
        self._index: tuple[pa.RecordBatch, np.ndarray, np.ndarray] | None = None

    @property
    def empty(self) -> bool:
        return not self._batches

    def add(self, batch: pa.RecordBatch, key_ids: np.ndarray) -> None:
        self._batches.append(batch)
        self._key_ids.append(key_ids)
        self._index = (
            None  # invalidate; the next probe rebuilds the grouped index over the new rows
        )

    def grouped(self, num_ids: int) -> tuple[pa.RecordBatch, np.ndarray, np.ndarray]:
        """Rows reordered into contiguous per-key-id runs, with ``start`` / ``count`` arrays indexed by
        key id (an id with no buffered rows has count 0). Built once and cached until the next ``add``.
        """
        if self._index is None:
            rows = self._batches[0] if len(self._batches) == 1 else pa.concat_batches(self._batches)
            ids = self._key_ids[0] if len(self._key_ids) == 1 else np.concatenate(self._key_ids)
            order = np.argsort(ids, kind="stable")  # gather each id's rows into one run
            start = np.zeros(num_ids, dtype=np.int64)
            count = np.zeros(num_ids, dtype=np.int64)
            if len(ids):
                uniq, first, cnt = np.unique(ids[order], return_index=True, return_counts=True)
                start[uniq] = first
                count[uniq] = cnt
            self._index = (rows.take(pa.array(order)), start, count)
        return self._index

    def rows_and_ids(self) -> tuple[pa.RecordBatch, np.ndarray]:
        """All buffered rows and their key ids, in arrival order — for the end-of-stream outer-join pass
        that emits the rows whose key never matched the other side."""
        rows = self._batches[0] if len(self._batches) == 1 else pa.concat_batches(self._batches)
        ids = self._key_ids[0] if len(self._key_ids) == 1 else np.concatenate(self._key_ids)
        return rows, ids

    def present(self, num_ids: int) -> np.ndarray:
        """A boolean array indexed by key id: ``True`` where at least one buffered row carries that id. The
        other side reads it to decide which of its rows found no match (an outer join's unmatched rows).
        """
        seen = np.zeros(num_ids, dtype=bool)
        if self._key_ids:
            ids = self._key_ids[0] if len(self._key_ids) == 1 else np.concatenate(self._key_ids)
            if len(ids):
                seen[ids] = True
        return seen

    def clear(self) -> None:
        self._batches.clear()
        self._key_ids.clear()
        self._index = None


class HashJoin(TwoInputOperator):
    """An equi-join: for every left row and right row whose join keys are equal, emit one joined row.
    ``how`` selects which non-matching rows are also kept: ``"inner"`` (default) keeps only matched pairs;
    ``"left"`` also keeps each left row that found no right match (its right columns null); ``"right"``
    keeps each unmatched right row (its left columns null); ``"outer"`` keeps both. Unmatched rows are
    emitted at end of stream — the point at which a row's key is known to have no match on the other side.

    It is a *symmetric hash join* — each side's rows are buffered as they arrive, indexed by key, and a
    new batch on one side is matched against every buffered row on the other — so each pair is emitted
    exactly once, when the later of the two arrives, independent of arrival order. Both inputs are
    co-partitioned on the join value by the keyed shuffle, so every row of a given key reaches the same
    instance from either side and the match is purely local. An *inner* join runs at any parallelism. An
    *outer* join runs at parallelism 1 only: a wider shuffle can route an instance zero rows on a side,
    and because a side's schema is learned from its first batch, such an instance could not type the
    absent side's null columns — so it would silently drop that side's unmatched rows.

    The output row is the left row's columns followed by the right row's *non-key* columns: the join key
    appears once (from the left), and the right's key columns are dropped (they equal the left's by the
    join condition). A non-key right column whose name collides with a left column name is rejected —
    rename one side. For an unmatched *right* row (a right/outer join), that join-key column takes the
    right row's key value and the other left columns are null. ``left_on`` and ``right_on`` name the
    equi-join columns on each side (a string or a sequence) and must have equal length; column *i* of
    ``left_on`` is matched against column *i* of ``right_on``. Keys are matched by value *and* scalar type
    — the same distinction the keyed shuffle draws — so an integer key column does not join a boolean one
    (an int ``1`` and a bool ``True`` are different keys), matching how they co-partition. A null key
    matches a null key (``null == null``), as nulls co-partition like any other key.

    State is both sides' buffered rows, held until end of stream and then cleared — the same
    unbounded-until-EOS tradeoff the keyed aggregations carry, and fine for a bounded input. Matched pairs
    are emitted as they form; an outer join additionally flushes its unmatched rows once at end of stream.
    """

    def __init__(
        self,
        left_on: str | Sequence[str],
        right_on: str | Sequence[str] | None = None,
        how: str = "inner",
    ) -> None:
        self.left_on = (left_on,) if isinstance(left_on, str) else tuple(left_on)
        ro: str | Sequence[str] = left_on if right_on is None else right_on
        self.right_on = (ro,) if isinstance(ro, str) else tuple(ro)
        if len(self.left_on) != len(self.right_on):
            raise ValueError(
                f"left_on {self.left_on} and right_on {self.right_on} must name the same number of "
                "columns (column i of left_on is matched against column i of right_on)"
            )
        if how not in ("inner", "left", "right", "outer"):
            raise ValueError(f"how must be 'inner', 'left', 'right', or 'outer', got {how!r}")
        self.how = how

    def open(self, ctx: OperatorContext) -> None:
        # An outer join's end-of-stream flush needs, on every instance, both input schemas and every row
        # of each key — neither of which a keyed shuffle guarantees an instance that it routed no rows on
        # a side. So restrict it to a single instance (the DSL rejects this earlier; this is the backstop
        # for a hand-built graph). An inner join emits nothing at EOS, so it is unaffected.
        if self.how != "inner" and ctx.num_subtasks > 1:
            raise ValueError(
                f"an outer join (how={self.how!r}) runs at parallelism 1 only, but this instance is "
                f"1 of {ctx.num_subtasks}; use how='inner' to run wider, or set the join's parallelism "
                "to 1"
            )
        # Each side's rows accumulate in a _SideBuffer, indexed by key id for a vectorized probe. They
        # grow until close() — the documented unbounded-state tradeoff.
        self._left_buf = _SideBuffer()
        self._right_buf = _SideBuffer()
        # One id space shared by both sides, so equal keys on left and right get the same integer id.
        # A single-column key (the common case) interns through a nested value-type -> value -> id map,
        # which needs no per-value tuple; a composite key falls back to one tuple per row. Both draw ids
        # from one dense counter, so the ids stay 0..n-1 for the vectorized probe.
        self._single_ids: dict[type, dict[object, int]] = {}
        self._multi_ids: dict[tuple[tuple[type, object], ...], int] = {}
        self._num_ids = 0
        # The single-integer-key fast path's state (:meth:`_encode` picks it on the first non-empty key).
        self._int_fast: bool | None = None
        self._int_id = np.empty(0, dtype=np.int64)  # key value → dense id (-1 = unseen)
        # Output schema parts, captured from the first batch of each side (no schema exists until then).
        # The full schemas are kept too, so an outer join can build correctly-typed null columns at EOS.
        self._left_names: list[str] | None = None
        self._right_value_cols: list[str] | None = None
        self._left_schema: pa.Schema | None = None
        self._right_schema: pa.Schema | None = None
        self._checked = False

    def process_left(self, batch: pa.RecordBatch, out: Collector) -> None:
        if self._left_names is None:
            self._left_names = list(batch.schema.names)
            self._left_schema = batch.schema
            self._check_columns()
        ids = self._encode(batch, self.left_on)
        self._probe_and_emit(batch, ids, self._right_buf, out, query_is_left=True)
        if batch.num_rows:  # buffer for the right rows that arrive later
            self._left_buf.add(batch, ids)

    def process_right(self, batch: pa.RecordBatch, out: Collector) -> None:
        if self._right_value_cols is None:
            keys = set(self.right_on)
            self._right_value_cols = [c for c in batch.schema.names if c not in keys]
            self._right_schema = batch.schema
            self._check_columns()
        ids = self._encode(batch, self.right_on)
        self._probe_and_emit(batch, ids, self._left_buf, out, query_is_left=False)
        if batch.num_rows:
            self._right_buf.add(batch, ids)

    def on_eos(self, out: Collector) -> None:
        # Inner join: nothing to flush — every match was emitted as it formed. An outer join emits the rows
        # whose key never appeared on the other side: an unmatched left row (left/outer) with null right
        # columns, an unmatched right row (right/outer) with null left columns and the join key from the
        # right. A row is unmatched iff its key id has no rows on the other side (the shared id space makes
        # that a presence lookup, not a re-probe).
        if self.how in ("left", "outer"):
            self._flush_unmatched(self._left_buf, self._right_buf, out, query_is_left=True)
        if self.how in ("right", "outer"):
            self._flush_unmatched(self._right_buf, self._left_buf, out, query_is_left=False)

    def _flush_unmatched(
        self, query_buf: _SideBuffer, other_buf: _SideBuffer, out: Collector, *, query_is_left: bool
    ) -> None:
        # A side that never established its schema (sent no batch, not even an empty one) leaves the null
        # columns untypable, so there is nothing well-formed to emit; skip it.
        if query_buf.empty or self._left_names is None or self._right_value_cols is None:
            return
        assert self._left_schema is not None and self._right_schema is not None
        # A row is unmatched iff its key id has no rows on the other side. Every buffered id is < _num_ids
        # (the shared counter), so it indexes the other side's presence bitmap directly.
        other_present = other_buf.present(self._num_ids)
        rows, ids = query_buf.rows_and_ids()
        idx = np.nonzero(~other_present[ids])[0]
        if idx.size == 0:
            return
        picked = rows.take(pa.array(idx))
        n = int(idx.size)
        if query_is_left:
            left_arrays: list[pa.Array] = list(picked.columns)  # the whole left row
            right_arrays = [
                pa.nulls(n, self._right_schema.field(c).type) for c in self._right_value_cols
            ]
        else:
            key_of = dict(
                zip(self.left_on, self.right_on, strict=True)
            )  # left key col → right key col
            left_arrays = [
                self._left_null_or_key(name, key_of, picked, n) for name in self._left_names
            ]
            right_arrays = [picked.column(c) for c in self._right_value_cols]
        out.emit(
            pa.RecordBatch.from_arrays(
                [*left_arrays, *right_arrays], names=[*self._left_names, *self._right_value_cols]
            )
        )

    def _left_null_or_key(
        self, name: str, key_of: dict[str, str], right_rows: pa.RecordBatch, n: int
    ) -> pa.Array:
        # Building a left-side column for an unmatched *right* row: a join-key column takes the right row's
        # key value (cast to the left column's type, so the output key column keeps one type across matched
        # and unmatched rows); every other left column is null.
        assert self._left_schema is not None
        left_type = self._left_schema.field(name).type
        if name in key_of:
            col = right_rows.column(key_of[name])
            return col if col.type == left_type else pc.cast(col, left_type)
        return pa.nulls(n, left_type)

    def close(self) -> None:
        self._left_buf.clear()
        self._right_buf.clear()
        self._single_ids.clear()
        self._multi_ids.clear()
        self._num_ids = 0
        self._int_fast = None
        self._int_id = np.empty(0, dtype=np.int64)

    def _check_columns(self) -> None:
        """Once both input schemas are known, reject a right non-key column that collides with a left
        column name (the output would have two columns of that name). A no-op until both sides seen.
        """
        if self._checked or self._left_names is None or self._right_value_cols is None:
            return
        left = set(self._left_names)
        collide = [c for c in self._right_value_cols if c in left]
        if collide:
            raise ValueError(
                f"join output column name collision on {collide}: each appears on both the left input and "
                "the right input's non-key columns; rename one side before joining"
            )
        self._checked = True

    def _encode(self, batch: pa.RecordBatch, key_columns: tuple[str, ...]) -> np.ndarray:
        """Each row's integer key id, stable across batches and shared by both inputs, so equal keys on the
        two sides get the same id. The map keys on each scalar's value *and* Python type — the distinction
        the keyed shuffle's ``msgpack`` draws — so a left ``int`` 1 and a right ``bool`` ``True`` get
        different ids (they would collapse under ``(1,) == (True,)`` and join at parallelism 1, yet the
        shuffle routes them apart at parallelism > 1), while ``int32`` 1 and ``int64`` 1 share one id (both
        surface as Python ``int``, matching the shuffle, which encodes the value not the width).

        A single **non-negative integer** key (indices, ids — the common case) interns fully vectorized:
        a value→id numpy array gathers each row's id and assigns unseen keys in bulk, with no per-key
        Python. Other single keys use ``dictionary_encode`` so the value→id intern runs once per *distinct*
        key (not per row), the per-row expansion being a numpy take."""
        if len(key_columns) == 1:
            col = batch.column(key_columns[0])
            if self._int_fast is None and len(col):  # decide on the first non-empty single key
                self._int_fast = pa.types.is_integer(col.type)
            if self._int_fast and pa.types.is_integer(col.type):
                # Integer values intern vectorially; any non-integer batch (e.g. a bool side of an
                # int↔bool join) falls through to the dict path below, so int values and non-int values
                # keep disjoint id spaces and equal-typed keys still match.
                ids = self._encode_int(col)
                if ids is not None:
                    return ids
                # None: this batch's keys can't use the dense value→id array. _encode_int has migrated the
                # ids assigned so far into the general dict, so each value keeps its id; use the dict path.
            enc = pc.dictionary_encode(col, null_encoding="encode")
            local_to_global = np.array(
                [self._intern_single(type(v), v) for v in enc.dictionary.to_pylist()],
                dtype=np.int64,
            )
            indices = enc.indices.to_numpy(zero_copy_only=False)
            return cast(np.ndarray, local_to_global[indices])
        # Several key columns: read each column once and intern each row's whole key tuple.
        cols = [batch.column(c).to_pylist() for c in key_columns]
        out = np.empty(batch.num_rows, dtype=np.int64)
        for r in range(batch.num_rows):
            out[r] = self._intern_multi(tuple((type(col[r]), col[r]) for col in cols))
        return out

    def _intern_single(self, value_type: type, value: object) -> int:
        """Global id for a single-column key, interned by (value type, value) with no tuple built — the
        hot path of a single-key join. A new (type, value) pair takes the next free id."""
        by_value = self._single_ids.get(value_type)
        if by_value is None:
            by_value = self._single_ids[value_type] = {}
        gid = by_value.get(value)
        if gid is None:
            gid = self._num_ids
            self._num_ids += 1
            by_value[value] = gid
        return gid

    def _intern_multi(self, key: tuple[tuple[type, object], ...]) -> int:
        """Global id for a composite (multi-column) key tuple, assigning the next free id on first sight."""
        gid = self._multi_ids.get(key)
        if gid is None:
            gid = self._num_ids
            self._num_ids += 1
            self._multi_ids[key] = gid
        return gid

    def _encode_int(self, col: pa.Array) -> np.ndarray | None:
        """Per-row ids for a single-integer key column, fully vectorized — or ``None`` when this batch's
        keys are unsafe for the dense value→id array (:func:`_int_fast_ok`), having first migrated the ids
        assigned so far into the general dict (:meth:`_demote_int`) so the caller can fall through to the
        dict path. Null rows share the one null id (``_intern_single(NoneType, None)``, so a null matches a
        null on either path); the rest go through :meth:`_intern_ints`."""
        n = len(col)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if col.null_count:
            present = np.asarray(col.drop_null().to_numpy(zero_copy_only=False))
            if not _int_fast_ok(present):
                self._demote_int()
                return None
            valid = np.asarray(col.is_valid())
            out = np.empty(n, dtype=np.int64)
            out[~valid] = self._intern_single(type(None), None)  # same null id as the dict path
            out[valid] = self._intern_ints(present)
            return out
        keys = np.asarray(col.to_numpy(zero_copy_only=False))
        if not _int_fast_ok(keys):
            self._demote_int()
            return None
        return self._intern_ints(keys)

    def _demote_int(self) -> None:
        """Copy the vectorized integer id map (:attr:`_int_id`) into the general (type, value) dict, then
        drop it, so a later unsafe integer batch interns through the dict path while every value already
        seen keeps the id it was assigned. A value's id must never change: both join sides — and every
        future batch — resolve equal keys through one shared id space, so a shifted id would silently stop
        equal keys from matching."""
        if self._int_id.size:
            by_value = self._single_ids.setdefault(int, {})
            for v in np.nonzero(self._int_id >= 0)[0].tolist():
                by_value[v] = int(self._int_id[v])
        self._int_id = np.empty(0, dtype=np.int64)
        self._int_fast = False

    def _intern_ints(self, keys: np.ndarray) -> np.ndarray:
        """Gather a fast-path-safe integer key array — the non-negative, dense keys :meth:`_encode_int` has
        already checked (:func:`_int_fast_ok`), so no bounds guard is needed here — to dense global ids
        through the ``_int_id`` value→id lookup, growing it to fit and assigning the next free ids to unseen
        values in one bulk pass."""
        if keys.size == 0:
            return keys.astype(np.int64, copy=False)
        top = int(keys.max()) + 1
        if top > self._int_id.size:  # grow the lookup to cover this batch's largest key value
            grown = np.full(top, -1, dtype=np.int64)
            grown[: self._int_id.size] = self._int_id
            self._int_id = grown
        ids = self._int_id[keys]
        unseen = ids < 0
        if unseen.any():  # assign the next free ids to the distinct new key values, in order
            new_vals = np.unique(keys[unseen])
            self._int_id[new_vals] = np.arange(
                self._num_ids, self._num_ids + new_vals.size, dtype=np.int64
            )
            self._num_ids += int(new_vals.size)
            ids = self._int_id[keys]
        return cast(np.ndarray, ids)

    def _probe_and_emit(
        self,
        query: pa.RecordBatch,
        ids: np.ndarray,
        other: _SideBuffer,
        out: Collector,
        *,
        query_is_left: bool,
    ) -> None:
        """Emit every join of ``query``'s rows against the buffered other side, in bulk. For each query
        row, its matches are the other side's contiguous run for that key id; the per-row match counts
        drive one ragged ``take`` per side, so the whole batch's cross product is built without a per-key
        Python loop."""
        if other.empty:
            return
        grouped, start, count = other.grouped(self._num_ids)
        nq = len(ids)
        within = ids < count.shape[0]  # a key id unseen on the other side has no run there
        qstart = np.zeros(nq, dtype=np.int64)
        qcount = np.zeros(nq, dtype=np.int64)
        qstart[within] = start[ids[within]]
        qcount[within] = count[ids[within]]
        total = int(qcount.sum())
        if total == 0:
            return
        if total == nq and int(qcount.max()) == 1:
            # The foreign-key / stream-enrichment shape: every query row matches exactly one buffered row
            # (the other side's key is unique over this batch). Then `query_take` would be `arange(nq)` —
            # an identity take that needlessly copies the whole batch — and each row's lone match sits at
            # its run start, so the other-side gather is just `qstart`. Emit the query side in place and
            # do one take instead of two, skipping the repeat/cumsum index-expansion entirely.
            query_rows = query
            other_rows = grouped.take(pa.array(qstart))
        else:
            query_take = np.repeat(
                np.arange(nq), qcount
            )  # each query row repeated by its match count
            # The other-side rows for query row i are grouped[qstart[i] : qstart[i] + qcount[i]]; expand
            # those ranges into one index array via the running-offset trick (no per-row Python).
            out_starts = np.zeros(nq, dtype=np.int64)
            np.cumsum(qcount[:-1], out=out_starts[1:])
            other_take = np.repeat(qstart - out_starts, qcount) + np.arange(total)
            query_rows = query.take(pa.array(query_take))
            other_rows = grouped.take(pa.array(other_take))
        left_rows, right_rows = (
            (query_rows, other_rows) if query_is_left else (other_rows, query_rows)
        )
        assert self._left_names is not None and self._right_value_cols is not None
        arrays = [*left_rows.columns, *(right_rows.column(c) for c in self._right_value_cols)]
        out.emit(
            pa.RecordBatch.from_arrays(arrays, names=[*self._left_names, *self._right_value_cols])
        )


class Union(TwoInputOperator):
    """Concatenates two input streams into one (SQL ``UNION ALL``): every batch from either side is
    forwarded unchanged and duplicates are kept — no dedup, no key, no buffering, so it holds no state and
    runs at any parallelism. The actor forwards EOS downstream once *both* inputs close.

    A single output stream carries one schema, so the two inputs must share theirs. The first batch (from
    whichever side arrives first) fixes the schema; a later batch — from either side — whose schema differs
    is rejected. Schemas are compared by column names and types, ignoring metadata; nautilus learns them
    from the data, so a mismatch surfaces at run time, not when the graph is built."""

    def open(self, ctx: OperatorContext) -> None:
        self._schema: pa.Schema | None = None

    def _forward(self, batch: pa.RecordBatch, out: Collector) -> None:
        if self._schema is None:
            self._schema = batch.schema
        elif not batch.schema.equals(self._schema):
            raise ValueError(
                f"union: both inputs must share a schema; got {batch.schema} "
                f"after {self._schema}"
            )
        out.emit(batch)

    def process_left(self, batch: pa.RecordBatch, out: Collector) -> None:
        self._forward(batch, out)

    def process_right(self, batch: pa.RecordBatch, out: Collector) -> None:
        self._forward(batch, out)
