"""Built-in operators used by the Stage 0 demos, tests and examples.

These are concrete, vectorized-where-cheap operators that exercise the streaming semantics:
stateless transforms, a keyed global aggregation (flushed at EOS), and a keyed tumbling-window sum
(fired on watermark advance). In Stage 3 these become the implementations behind the fluent
``map``/``key_by``/``window``/``reduce`` combinators.
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
from nautilus.core.records import EOS_FRAME, WATERMARK_MAX, Frame
from nautilus.state import KeyContext, StateScope
from nautilus.windows import TimeWindow, TumblingEventTimeWindows


def _add(a: int, b: int) -> int:
    return a + b


class InMemorySource(SourceOperator):
    """Yields a fixed, pre-built sequence of frames; used by deterministic tests. A bounded
    source must end its frame list with ``EOS_FRAME``; :func:`from_batches` appends it for you."""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    async def frames(self) -> AsyncIterator[Frame]:
        for frame in self._frames:
            yield frame


def from_batches(*frames: Frame) -> InMemorySource:
    """Build a bounded :class:`InMemorySource`, appending the terminal ``EOS_FRAME`` for you (omitting
    it yields a source that never signals completion). Use ``InMemorySource([...])`` directly when a
    test needs exact frame control (e.g. to place watermarks or omit EOS)."""
    return InMemorySource([*frames, EOS_FRAME])


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

    def __init__(self, key_col: str, count_col: str = "count") -> None:
        self.key_col = key_col
        self.count_col = count_col

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        counts = pc.value_counts(batch.column(self.key_col))
        for value, count in zip(
            counts.field("values").to_pylist(), counts.field("counts").to_pylist(), strict=True
        ):
            self._ctx.reducing_state("count", KeyContext((value,)), _add).add(int(count))

    def on_watermark(self, t: int, out: Collector) -> None:
        if t < WATERMARK_MAX:
            return  # global aggregation: only the terminal watermark fires it
        keys: list[object] = []
        totals: list[int] = []
        for key, _ns, value in self._ctx.state_backend.entries(self._ctx.operator_id, "count"):
            keys.append(key[0])
            totals.append(cast(int, value))
            self._ctx.state_backend.clear(StateScope(self._ctx.operator_id, "count", key, _ns))
        if keys:
            out.emit(
                pa.RecordBatch.from_arrays(
                    [pa.array(keys), pa.array(totals, pa.int64())],
                    names=[self.key_col, self.count_col],
                )
            )
            # operator-author custom metric (separate recorder; no-op unless telemetry is on)
            self._ctx.metrics.incr("window.fires", 1, operator_id=self._ctx.operator_id)


class KeyedTumblingSum(OneInputOperator):
    """Sums a value column per key per tumbling event-time window. Each window fires when the
    operator watermark passes its end (and any still-open windows fire at EOS)."""

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

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        keys = batch.column(self.key_col).to_pylist()
        values = batch.column(self.value_col).to_pylist()
        times = pc.cast(batch.column(self.ts_col), pa.int64()).to_pylist()
        for key, value, ts in zip(keys, values, times, strict=True):
            for window in self.window.assign(ts):
                self._ctx.reducing_state("acc", KeyContext((key,), window), _add).add(value)

    def on_watermark(self, t: int, out: Collector) -> None:
        keys: list[object] = []
        starts: list[int] = []
        ends: list[int] = []
        sums: list[int] = []
        fired: list[tuple[tuple[object, ...], TimeWindow]] = []
        for key, namespace, value in self._ctx.state_backend.entries(self._ctx.operator_id, "acc"):
            window = cast(TimeWindow, namespace)
            if window.end <= t:
                keys.append(key[0])
                starts.append(window.start)
                ends.append(window.end)
                sums.append(cast(int, value))
                fired.append((key, window))
        for key, window in fired:
            self._ctx.state_backend.clear(StateScope(self._ctx.operator_id, "acc", key, window))
        if keys:
            out.emit(
                pa.RecordBatch.from_arrays(
                    [
                        pa.array(keys),
                        pa.array(starts, pa.int64()),
                        pa.array(ends, pa.int64()),
                        pa.array(sums, pa.int64()),
                    ],
                    names=[self.key_col, self.start_col, self.end_col, self.sum_col],
                )
            )
        if fired:
            # operator-author custom metric (separate recorder; no-op unless telemetry is on)
            self._ctx.metrics.incr("window.fires", len(fired), operator_id=self._ctx.operator_id)
