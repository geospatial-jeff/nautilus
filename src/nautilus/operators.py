"""Built-in operators used by the Stage 0 demos, tests and examples.

Concrete operators that exercise the streaming semantics — each follows the synchronous
``process``/``on_watermark`` contract (emit into the ``Collector``, never await; see
:mod:`nautilus.core.operator`). In Stage 3 these become the implementations behind the fluent
``map``/``key_by``/``window``/``reduce`` combinators. What each one does is on its own class.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nautilus.core.operator import (
    Collector,
    OneInputOperator,
    OperatorContext,
    SourceOperator,
    TwoInputOperator,
)
from nautilus.core.records import EOS_FRAME, WATERMARK_MAX, Batch, Frame
from nautilus.core.time import to_epoch_micros
from nautilus.state import KeyContext
from nautilus.windows import TimeWindow, TumblingEventTimeWindows


def _add(a: int, b: int) -> int:
    return a + b


class InMemorySource(SourceOperator):
    """Yields a fixed, pre-built sequence of frames; used by deterministic tests. A bounded
    source must end its frame list with ``EOS_FRAME``; :func:`from_batches` appends it for you."""

    def __init__(self, frames: list[Frame]) -> None:
        for frame in frames:  # fail loudly at construction, not by silently vanishing in the actor
            if not isinstance(frame, Frame):
                raise TypeError(
                    f"InMemorySource frames must be Frame objects (Batch/Watermark/EOS/...), got "
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
        # Per-row str.split(). A columnar pc.utf8_split_whitespace + pc.list_flatten splits correctly but
        # the flattened child view transiently corrupted under load (a pyarrow buffer-lifetime issue that
        # passes on retry); a streaming engine cannot ship a nondeterministic tokenizer, so this stays
        # exact and the columnar form is deferred. The keyed shuffle is the measured hot path here.
        words: list[str] = []
        for s in batch.column(self.in_col).to_pylist():
            if s:
                words.extend((s.lower() if self.lowercase else s).split())
        if words:
            arr = pa.array(words, pa.string())
            out.emit(pa.RecordBatch.from_arrays([arr], names=[self.out_col]))


class KeyedCount(OneInputOperator):
    """Counts occurrences per key. A keyed *global* aggregation: results are emitted at EOS, when
    the watermark reaches ``WATERMARK_MAX``."""

    _STATE = "count"  # state-backend name (distinct from the output column, which count_col names)

    def __init__(self, key_col: str, count_col: str = "count") -> None:
        self.key_col = key_col
        self.count_col = count_col

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx
        self._key_type: pa.DataType | None = (
            None  # captured from the input so output keeps its type
        )

    def key_columns(self) -> tuple[str, ...]:
        return (self.key_col,)

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        if self._key_type is None:
            self._key_type = batch.column(self.key_col).type
        counts = pc.value_counts(batch.column(self.key_col))
        for value, count in zip(
            counts.field("values").to_pylist(), counts.field("counts").to_pylist(), strict=True
        ):
            self._ctx.reducing_state(self._STATE, KeyContext((value,)), _add).add(int(count))

    def on_watermark(self, t: int, out: Collector) -> None:
        if t < WATERMARK_MAX:
            return  # global aggregation: only the terminal watermark fires it
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
            # operator-author custom metric (separate recorder; no-op unless telemetry is on)
            self._ctx.metrics.incr("window.fires", 1, operator_id=self._ctx.operator_id)


class KeyedTumblingSum(OneInputOperator):
    """Sums a value column per key per tumbling event-time window. Each window fires when the
    operator watermark passes its end (and any still-open windows fire at EOS)."""

    _STATE = "acc"  # state-backend name; the window is the namespace, the key the key

    def __init__(
        self,
        key_col: str,
        value_col: str,
        ts_col: str,
        window: TumblingEventTimeWindows,
        *,
        start_col: str = "window_start",
        end_col: str = "window_end",
        sum_col: str = "sum",
    ) -> None:
        self.key_col = key_col
        self.value_col = value_col
        self.ts_col = ts_col
        self.window = window
        self.start_col = start_col
        self.end_col = end_col
        self.sum_col = sum_col

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx
        # Captured from the input so the output keeps the key's type and the sum's natural type (an int
        # column sums to int64, a float column to double — never silently truncated to int).
        self._key_type: pa.DataType | None = None
        self._sum_type: pa.DataType | None = None

    def key_columns(self) -> tuple[str, ...]:
        return (self.key_col,)

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        # A tumbling window assigns each row to exactly one window [start, start+size); the assigner owns
        # that boundary formula (computed columnar here). Partial-summing this batch per (key, window) in
        # Arrow first turns the old per-row state write into one write per distinct (key, window) — far
        # fewer Python-level state ops — and each per-batch partial folds into the running sum because
        # addition is associative.
        size = self.window.size
        ts = to_epoch_micros(batch.column(self.ts_col)).to_numpy(zero_copy_only=False)
        window_start = self.window.assign_starts(ts)
        grouped = (
            pa.table(
                {
                    "key": batch.column(self.key_col),
                    "ws": pa.array(window_start),
                    "val": batch.column(self.value_col),
                }
            )
            .group_by(["key", "ws"])
            .aggregate([("val", "sum")])
        )
        if self._key_type is None:
            self._key_type = batch.column(self.key_col).type
            self._sum_type = grouped.column("val_sum").type  # Arrow's chosen sum type, not int64
        keys = grouped.column("key").to_pylist()
        starts = grouped.column("ws").to_pylist()
        partials = grouped.column("val_sum").to_pylist()
        for key, start, partial in zip(keys, starts, partials, strict=True):
            window = TimeWindow(start, start + size)
            self._ctx.reducing_state(self._STATE, KeyContext((key,), window), _add).add(partial)

    def on_watermark(self, t: int, out: Collector) -> None:
        keys: list[object] = []
        starts: list[int] = []
        ends: list[int] = []
        sums: list[object] = []
        fired: list[KeyContext] = []
        for kctx, value in self._ctx.entries(self._STATE):
            window = cast(TimeWindow, kctx.namespace)
            if window.end <= t:
                keys.append(kctx.key[0])
                starts.append(window.start)
                ends.append(window.end)
                sums.append(value)
                fired.append(kctx)
        for kctx in fired:
            self._ctx.clear_state(self._STATE, kctx)
        if keys:
            out.emit(
                pa.RecordBatch.from_arrays(
                    [
                        pa.array(keys, self._key_type),
                        pa.array(starts, pa.int64()),
                        pa.array(ends, pa.int64()),
                        pa.array(sums, self._sum_type),
                    ],
                    names=[self.key_col, self.start_col, self.end_col, self.sum_col],
                )
            )
        if fired:
            # operator-author custom metric (separate recorder; no-op unless telemetry is on)
            self._ctx.metrics.incr("window.fires", len(fired), operator_id=self._ctx.operator_id)


def _group_rows_by_key(
    batch: pa.RecordBatch, key_columns: tuple[str, ...]
) -> dict[tuple[object, ...], pa.RecordBatch]:
    """Split a batch into one sub-batch per distinct key tuple. The key is read with ``to_pylist`` — the
    same Python scalars the keyed shuffle hashes on — so the join and the shuffle agree on every key, and
    co-partitioned rows meet here under the identical tuple."""
    cols = [batch.column(c).to_pylist() for c in key_columns]
    rows_by_key: dict[tuple[object, ...], list[int]] = {}
    for r in range(batch.num_rows):
        rows_by_key.setdefault(tuple(col[r] for col in cols), []).append(r)
    return {k: batch.take(pa.array(rows, pa.int64())) for k, rows in rows_by_key.items()}


class HashJoin(TwoInputOperator):
    """An inner equi-join: for every left row and right row whose join keys are equal, emit one joined row.

    It is a *symmetric hash join* — each side's rows are buffered by key as they arrive, and a new row on
    one side is immediately matched against every buffered row on the other — so each pair is emitted
    exactly once, when the later of the two arrives, independent of arrival order. Both inputs are
    co-partitioned on the join value by the keyed shuffle, so every row of a given key reaches the same
    instance from either side and the match is purely local.

    The output row is the left row's columns followed by the right row's *non-key* columns: the join key
    appears once (from the left), and the right's key columns are dropped (they equal the left's by the
    join condition). A non-key right column whose name collides with a left column name is rejected —
    rename one side. ``left_on`` and ``right_on`` name the equi-join columns on each side (a string or a
    sequence) and must have equal length; column *i* of ``left_on`` is matched against column *i* of
    ``right_on``.

    State is the per-key buffers of both sides, held until end of stream and then cleared — the same
    unbounded-until-EOS tradeoff the keyed aggregations carry, and fine for a bounded input. ``on_watermark``
    is a no-op: matches are emitted as they form, so nothing waits on event time. A windowed variant that
    bounds and evicts this state on the watermark is the additive next step, and this operator leaves that
    seam (``on_watermark``) open for it.
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
        # Per-key buffers: each holds one running RecordBatch of the rows seen so far for that key. They
        # grow until close() — the documented unbounded-state tradeoff.
        self._left: dict[tuple[object, ...], pa.RecordBatch] = {}
        self._right: dict[tuple[object, ...], pa.RecordBatch] = {}
        # Output schema parts, captured from the first batch of each side (no schema exists until then).
        self._left_names: list[str] | None = None
        self._right_value_cols: list[str] | None = None
        self._checked = False

    def process_left(self, batch: pa.RecordBatch, out: Collector) -> None:
        if self._left_names is None:
            self._left_names = list(batch.schema.names)
            self._check_columns()
        for key, sub in _group_rows_by_key(batch, self.left_on).items():
            buffered_right = self._right.get(key)
            if buffered_right is not None:  # these right rows arrived first; complete the pairs now
                out.emit(self._join(sub, buffered_right))
            self._left[key] = _concat(self._left.get(key), sub)

    def process_right(self, batch: pa.RecordBatch, out: Collector) -> None:
        if self._right_value_cols is None:
            keys = set(self.right_on)
            self._right_value_cols = [c for c in batch.schema.names if c not in keys]
            self._check_columns()
        for key, sub in _group_rows_by_key(batch, self.right_on).items():
            buffered_left = self._left.get(key)
            if buffered_left is not None:
                out.emit(self._join(buffered_left, sub))
            self._right[key] = _concat(self._right.get(key), sub)

    def close(self) -> None:
        self._left.clear()
        self._right.clear()

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

    def _join(self, left: pa.RecordBatch, right: pa.RecordBatch) -> pa.RecordBatch:
        """The cross product of ``left`` and ``right`` rows (all share the join key): each left row paired
        with every right row, left columns then right non-key columns."""
        assert self._left_names is not None and self._right_value_cols is not None
        nl, nr = left.num_rows, right.num_rows
        left_part = left.take(pa.array(np.repeat(np.arange(nl), nr), pa.int64()))
        right_part = right.take(pa.array(np.tile(np.arange(nr), nl), pa.int64()))
        arrays = [*left_part.columns, *(right_part.column(c) for c in self._right_value_cols)]
        return pa.RecordBatch.from_arrays(
            arrays, names=[*self._left_names, *self._right_value_cols]
        )


def _concat(buffered: pa.RecordBatch | None, sub: pa.RecordBatch) -> pa.RecordBatch:
    """Append ``sub``'s rows to a key's running buffer (or start one). Same schema both sides — they come
    from the one input — so the concatenation is well-defined."""
    return sub if buffered is None else pa.concat_batches([buffered, sub])
