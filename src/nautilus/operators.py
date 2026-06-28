"""Built-in operators used by the Stage 0 demos, tests and examples.

Concrete operators that exercise the streaming semantics — each follows the synchronous
``process``/``on_watermark`` contract (emit into the ``Collector``, never await; see
:mod:`nautilus.core.operator`). In Stage 3 these become the implementations behind the fluent
``map``/``key_by``/``window``/``reduce`` combinators. What each one does is on its own class.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc

from nautilus.core.operator import (
    Collector,
    OneInputOperator,
    OperatorContext,
    SourceOperator,
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
