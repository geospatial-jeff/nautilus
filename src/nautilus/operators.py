"""The built-in operators — the implementations behind the fluent ``Stream`` combinators.

Concrete operators that exercise the streaming semantics. Most follow the synchronous
``process``/``on_eos`` contract (emit into the ``Collector``, never await; see
:mod:`nautilus.core.operator`): :class:`MapBatch`, :class:`FilterRows`, :class:`Tokenize`, and
:class:`KeyedCount` back the DSL's ``.map`` / ``.filter`` / ``.tokenize`` / ``.count_by``, and
:class:`HashJoin` backs ``.join``. :class:`AsyncMapBatch` is the one awaiting built-in — it backs
``.map_async``, doing its I/O in ``fetch`` and emitting in ``integrate``. What each one does is on its
own class.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import cast

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
    """Counts occurrences per key. A keyed *global* aggregation: results are emitted at end of stream
    (:meth:`on_eos`).

    For **non-negative integer keys** — the common case (indices, hashed buckets, the keyed shuffle's own
    routing ids) — it accumulates counts in a numpy array indexed by the key value (``np.bincount`` per
    batch, then a vectorized add into the running array). That folds a whole batch with no per-key Python
    object at all — no ``to_pylist`` materialization, no dict, no ``KeyContext`` — which is the dominant
    cost at high cardinality. Any other key type (strings, negatives, floats) falls back to the keyed-state
    fold. Null keys are counted as their own group either way."""

    _STATE = "count"  # state-backend name (distinct from the output column, which count_col names)

    def __init__(self, key_col: str, count_col: str = "count") -> None:
        self.key_col = key_col
        self.count_col = count_col

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx
        self._key_type: pa.DataType | None = None  # captured from the input so output keeps its type
        self._counts: np.ndarray | None = None  # numpy accumulator while the integer fast path is active
        self._nulls = 0  # null-key occurrences (fast path only; the dict path folds null like any key)

    def key_columns(self) -> tuple[str, ...]:
        return (self.key_col,)

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        col = batch.column(self.key_col)
        if self._key_type is None:
            self._key_type = col.type
            if pa.types.is_integer(col.type):
                self._counts = np.zeros(0, dtype=np.int64)
        if self._counts is not None:
            if col.null_count:
                self._nulls += col.null_count
                col = col.drop_null()
            keys = np.asarray(col.to_numpy(zero_copy_only=False))
            if keys.size:
                bc = np.bincount(keys)  # non-negative ints only (raises otherwise → drops to no fast path)
                if bc.size > self._counts.size:
                    grown = np.zeros(bc.size, dtype=np.int64)
                    grown[: self._counts.size] = self._counts
                    self._counts = grown
                self._counts[: bc.size] += bc
            return
        counts = pc.value_counts(col)
        # One bulk fold of the batch's per-key counts — no KeyContext or reducing-state handle per key.
        keys = ((v,) for v in counts.field("values").to_pylist())
        self._ctx.reduce_all(
            self._STATE, zip(keys, counts.field("counts").to_pylist(), strict=True), _add
        )

    def on_eos(self, out: Collector) -> None:
        if self._counts is not None:  # integer fast path: emit nonzero keys + counts, vectorized
            nz = np.nonzero(self._counts)[0]
            if nz.size or self._nulls:
                key_arr = pa.array(nz, self._key_type)
                cnt_arr = pa.array(self._counts[nz], pa.int64())
                if self._nulls:  # append the null-key group (rare; a small concat, not a per-key loop)
                    key_arr = pa.concat_arrays([key_arr, pa.array([None], self._key_type)])
                    cnt_arr = pa.concat_arrays([cnt_arr, pa.array([self._nulls], pa.int64())])
                out.emit(pa.RecordBatch.from_arrays([key_arr, cnt_arr], names=[self.key_col, self.count_col]))
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

    def clear(self) -> None:
        self._batches.clear()
        self._key_ids.clear()
        self._index = None


class HashJoin(TwoInputOperator):
    """An inner equi-join: for every left row and right row whose join keys are equal, emit one joined row.

    It is a *symmetric hash join* — each side's rows are buffered as they arrive, indexed by key, and a
    new batch on one side is matched against every buffered row on the other — so each pair is emitted
    exactly once, when the later of the two arrives, independent of arrival order. Both inputs are
    co-partitioned on the join value by the keyed shuffle, so every row of a given key reaches the same
    instance from either side and the match is purely local.

    The output row is the left row's columns followed by the right row's *non-key* columns: the join key
    appears once (from the left), and the right's key columns are dropped (they equal the left's by the
    join condition). A non-key right column whose name collides with a left column name is rejected —
    rename one side. ``left_on`` and ``right_on`` name the equi-join columns on each side (a string or a
    sequence) and must have equal length; column *i* of ``left_on`` is matched against column *i* of
    ``right_on``. Keys are matched by value *and* scalar type — the same distinction the keyed shuffle
    draws — so an integer key column does not join a boolean one (an int ``1`` and a bool ``True`` are
    different keys), matching how they co-partition. A null key matches a null key (``null == null``), as
    nulls co-partition like any other key.

    State is both sides' buffered rows, held until end of stream and then cleared — the same
    unbounded-until-EOS tradeoff the keyed aggregations carry, and fine for a bounded input. Matches are
    emitted as they form, so there is no end-of-stream flush.
    """

    def __init__(
        self, left_on: str | Sequence[str], right_on: str | Sequence[str] | None = None
    ) -> None:
        self.left_on = (left_on,) if isinstance(left_on, str) else tuple(left_on)
        ro: str | Sequence[str] = left_on if right_on is None else right_on
        self.right_on = (ro,) if isinstance(ro, str) else tuple(ro)
        if len(self.left_on) != len(self.right_on):
            raise ValueError(
                f"left_on {self.left_on} and right_on {self.right_on} must name the same number of "
                "columns (column i of left_on is matched against column i of right_on)"
            )

    def open(self, ctx: OperatorContext) -> None:
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
        # Single non-negative-integer key (the common case): intern vectorially through a value→id numpy
        # array instead of a Python loop over distinct values, decided on the first non-empty single key.
        self._int_fast: bool | None = None
        self._int_id = np.empty(0, dtype=np.int64)  # key value → dense id (-1 = unseen)
        # Output schema parts, captured from the first batch of each side (no schema exists until then).
        self._left_names: list[str] | None = None
        self._right_value_cols: list[str] | None = None
        self._checked = False

    def process_left(self, batch: pa.RecordBatch, out: Collector) -> None:
        if self._left_names is None:
            self._left_names = list(batch.schema.names)
            self._check_columns()
        ids = self._encode(batch, self.left_on)
        self._probe_and_emit(batch, ids, self._right_buf, out, query_is_left=True)
        if batch.num_rows:  # buffer for the right rows that arrive later
            self._left_buf.add(batch, ids)

    def process_right(self, batch: pa.RecordBatch, out: Collector) -> None:
        if self._right_value_cols is None:
            keys = set(self.right_on)
            self._right_value_cols = [c for c in batch.schema.names if c not in keys]
            self._check_columns()
        ids = self._encode(batch, self.right_on)
        self._probe_and_emit(batch, ids, self._left_buf, out, query_is_left=False)
        if batch.num_rows:
            self._right_buf.add(batch, ids)

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
        Python at all. Other single keys use ``dictionary_encode`` so the value→id intern runs once per
        *distinct* key (not per row), the per-row expansion being a numpy take."""
        if len(key_columns) == 1:
            col = batch.column(key_columns[0])
            if self._int_fast is None and len(col):  # decide on the first non-empty single key
                arr = col.drop_null() if col.null_count else col
                self._int_fast = pa.types.is_integer(col.type) and (
                    len(arr) == 0 or pc.min(arr).as_py() >= 0
                )
            if self._int_fast and pa.types.is_integer(col.type):
                # Integer values intern vectorially; any non-integer batch (e.g. a bool side of an
                # int↔bool join) falls through to the dict path below, so int values and non-int values
                # keep disjoint id spaces and equal-typed keys still match.
                return self._encode_int(col)
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

    def _encode_int(self, col: pa.Array) -> np.ndarray:
        """Per-row ids for a single non-negative-integer key column, fully vectorized. Null rows share the
        one null id (``_intern_single(NoneType, None)``, so a null matches a null on either path); the rest
        go through :meth:`_intern_ints`."""
        n = len(col)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if col.null_count:
            valid = np.asarray(col.is_valid())
            out = np.empty(n, dtype=np.int64)
            out[~valid] = self._intern_single(type(None), None)  # the same id the dict path gives a null
            out[valid] = self._intern_ints(np.asarray(col.drop_null().to_numpy(zero_copy_only=False)))
            return out
        return self._intern_ints(np.asarray(col.to_numpy(zero_copy_only=False)))

    def _intern_ints(self, keys: np.ndarray) -> np.ndarray:
        """Map a non-negative integer key array to dense global ids through a value→id lookup array,
        assigning ids to unseen values in one bulk pass — the vectorized replacement for the per-key
        :meth:`_intern_single` loop, which dominated a high-cardinality join (once per row when keys are
        all distinct)."""
        if keys.size == 0:
            return keys.astype(np.int64, copy=False)
        if keys.min() < 0:  # decided non-negative on the first batch; a later negative can't index _int_id
            raise ValueError("HashJoin integer fast path requires non-negative keys")
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
        return ids

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
        query_take = np.repeat(np.arange(nq), qcount)  # each query row repeated by its match count
        # The other-side rows for query row i are grouped[qstart[i] : qstart[i] + qcount[i]]; expand those
        # ranges into one index array via the running-offset trick (no per-row Python).
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
