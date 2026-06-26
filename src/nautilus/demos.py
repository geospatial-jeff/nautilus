"""Long-running demo sources for the live dashboard.

:class:`DemoStreamSource` is a :class:`~nautilus.core.operator.SourceOperator`: it emits a small
event-time batch every ``interval_s`` and interleaves watermarks so a downstream window operator fires
live, ``await``-ing between batches so the event loop (and the hardware sampler) stay responsive. Kept
out of :mod:`nautilus.operators` to keep an unbounded demo source separate from the production operators.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pyarrow as pa

from nautilus.core.operator import OperatorContext, SourceOperator
from nautilus.core.records import EOS_FRAME, Batch, Frame, Watermark


class DemoStreamSource(SourceOperator):
    """Emits ``rows_per_batch`` rows (key, val, ts) every ``interval_s`` seconds, each batch falling in
    the next tumbling window; a watermark after each batch fires the previous window. Bounded after
    ``max_batches`` (set it to ``None`` for an unbounded stream)."""

    def __init__(
        self,
        *,
        interval_s: float = 0.2,
        max_batches: int | None = 20,
        rows_per_batch: int = 3,
        keys: tuple[str, ...] = ("a", "b", "c"),
        window_micros: int = 1_000_000,
    ) -> None:
        self.interval_s = interval_s
        self.max_batches = max_batches
        self.rows_per_batch = rows_per_batch
        self.keys = keys
        self.window_micros = window_micros
        self.closed = False

    def open(self, ctx: OperatorContext) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def frames(self) -> AsyncIterator[Frame]:
        n = 0
        while self.max_batches is None or n < self.max_batches:
            base = n * self.window_micros
            keys = [self.keys[(n + i) % len(self.keys)] for i in range(self.rows_per_batch)]
            vals = [((n * self.rows_per_batch + i) % 17) + 1 for i in range(self.rows_per_batch)]
            tss = [base + i * 1000 + 1 for i in range(self.rows_per_batch)]
            yield Batch(
                pa.RecordBatch.from_arrays(
                    [pa.array(keys), pa.array(vals, pa.int64()), pa.array(tss, pa.int64())],
                    names=["key", "val", "ts"],
                )
            )
            yield Watermark(base)  # monotonic; fires the previous window
            n += 1
            await asyncio.sleep(self.interval_s)
        yield EOS_FRAME
