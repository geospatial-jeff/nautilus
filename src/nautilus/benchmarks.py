"""Large-scale synthetic sources for performance work — the workloads the built-in examples don't reach.

The examples in ``pipelines.py`` are tiny (tens of rows) so they read clearly and run instantly; none
fills a channel, grows state, or pushes a histogram past its low buckets. These sources generate
millions of rows with *no* real-time pacing (unlike :class:`~nautilus.demos.DemoStreamSource`, which
sleeps), so a run is bound by the engine, not a timer — which is what a throughput measurement needs.

Generation is **deterministic**: keys, values, and timestamps are pure functions of the row index, so a
re-run produces byte-identical input. That is what lets the dev loop trust a structural digest — the
provably-reproducible fingerprint of a run's results — as an unchanged-output check across a code change,
and read a throughput delta as a real effect rather than input noise.

Scale is read from the environment so the same registered pipeline serves both a quick check and a real
benchmark without code edits:

* ``NAUTILUS_BENCH_ROWS``  — total data rows (default 1,000,000)
* ``NAUTILUS_BENCH_BATCH`` — rows per batch (default 4096)
* ``NAUTILUS_BENCH_KEYS``  — distinct keys / word vocabulary (default 1000)
* ``NAUTILUS_BENCH_WM_EVERY`` — emit a watermark every N batches (default 8)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import numpy as np
import pyarrow as pa

from nautilus.core.operator import SourceOperator
from nautilus.core.records import EOS_FRAME, Batch, Frame, Watermark


def passthrough(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Identity map — the linear benchmark's transform. A module-level function (not a lambda) so a
    ``--workers`` run can cloudpickle the operator to a spawned worker."""
    return batch


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def bench_params() -> dict[str, int]:
    """Resolve the benchmark scale from the environment (with defaults). One place so every builder and
    any reporting agent reads the same knobs."""
    rows = _env_int("NAUTILUS_BENCH_ROWS", 1_000_000)
    batch_rows = _env_int("NAUTILUS_BENCH_BATCH", 4096)
    return {
        "rows": rows,
        "batch_rows": batch_rows,
        "num_batches": max(1, -(-rows // batch_rows)),  # ceil; at least one batch
        "key_cardinality": _env_int("NAUTILUS_BENCH_KEYS", 1000),
        "wm_every": _env_int("NAUTILUS_BENCH_WM_EVERY", 8),
    }


class SyntheticKeyedSource(SourceOperator):
    """Yields ``num_batches`` batches of ``batch_rows`` rows: ``key`` (int64, ``row_index %
    key_cardinality``), ``value`` (int64, constant 1), and ``ts`` (int64, the row index — strictly
    monotonic event time). A watermark advancing to the latest emitted ``ts`` is broadcast every
    ``wm_every`` batches, then one EOS. No ``await`` between batches, so the source runs flat out.

    Uniform round-robin keys spread evenly across a keyed shuffle's instances; because consecutive row
    indices cycle through every key, one window of at least ``key_cardinality`` rows holds all keys, so
    open-window state is ``~key_cardinality`` entries — large enough to be visible, bounded by regular
    window firing."""

    def __init__(
        self,
        *,
        num_batches: int,
        batch_rows: int,
        key_cardinality: int,
        wm_every: int = 8,
        extra_value_cols: int = 0,
    ) -> None:
        self._num_batches = num_batches
        self._batch_rows = batch_rows
        self._key_cardinality = key_cardinality
        self._wm_every = wm_every
        self._extra_value_cols = extra_value_cols  # widen the schema (transport / IPC stress)

    async def frames(self) -> AsyncIterator[Frame]:
        n = self._batch_rows
        ones = pa.array(np.ones(n, dtype=np.int64))
        next_ts = 0
        for b in range(self._num_batches):
            idx = np.arange(next_ts, next_ts + n, dtype=np.int64)
            columns: dict[str, pa.Array] = {
                "key": pa.array(idx % self._key_cardinality),
                "value": ones,
                "ts": pa.array(idx),
            }
            for c in range(self._extra_value_cols):
                columns[f"v{c}"] = ones
            yield Batch(pa.record_batch(columns))
            next_ts += n
            if (b + 1) % self._wm_every == 0:
                yield Watermark(next_ts - 1)  # advance event time to the latest ts emitted so far
        yield EOS_FRAME


class SyntheticTextSource(SourceOperator):
    """Yields ``num_batches`` batches of ``rows_per_batch`` lines, each line ``tokens_per_row`` space-
    separated words drawn round-robin from a ``vocabulary``-sized set (``w0``..``w{vocabulary-1}``).
    Feeds the tokenize → keyed-count fan-out: one short input row explodes into many output rows, so the
    output batch-size histogram and a keyed shuffle on ``word`` both run hot. Deterministic and unpaced."""

    def __init__(
        self,
        *,
        num_batches: int,
        rows_per_batch: int,
        tokens_per_row: int,
        vocabulary: int,
    ) -> None:
        self._num_batches = num_batches
        self._rows_per_batch = rows_per_batch
        self._tokens_per_row = tokens_per_row
        self._vocabulary = vocabulary

    async def frames(self) -> AsyncIterator[Frame]:
        token = 0
        for _ in range(self._num_batches):
            lines = []
            for _ in range(self._rows_per_batch):
                words = [f"w{(token + j) % self._vocabulary}" for j in range(self._tokens_per_row)]
                token += self._tokens_per_row
                lines.append(" ".join(words))
            yield Batch(pa.record_batch({"line": pa.array(lines, pa.string())}))
        yield EOS_FRAME
