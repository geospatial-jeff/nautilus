"""Large-scale synthetic sources for performance work — the workloads the built-in examples don't reach.

The examples in ``pipelines.py`` are tiny (tens of rows) so they read clearly and run instantly; none
fills a channel, grows state, or pushes a histogram past its low buckets. These sources generate
millions of rows with *no* real-time pacing (unlike :class:`~nautilus.demos.DemoStreamSource`, which
sleeps), so a run is bound by the engine, not a timer — which is what a throughput measurement needs.

Generation is **deterministic** — even the randomized-looking knobs (skew, varied values, nulls)
draw from a fixed seed — so a re-run produces byte-identical input. That is what lets the dev loop trust a
structural digest as an unchanged-output check across a code change, and read a throughput delta as a real
effect rather than input noise.

The clean stream (uniform keys, constant values) is good for *isolating* one cost, but it is not a real
workload. :class:`SyntheticKeyedSource`'s knobs add the realism a real stream has — key **skew**,
**varied values**, **null** keys, a wider **payload** — each isolating a stressor the clean stream cannot
reach; the ``bench-skew`` / ``bench-backpressure`` pipelines wire them up. See the class docstring.

Scale is read from the environment so the same registered pipeline serves both a quick check and a real
benchmark without code edits:

* ``NAUTILUS_BENCH_ROWS``  — total data rows (default 1,000,000)
* ``NAUTILUS_BENCH_BATCH`` — rows per batch (default 4096)
* ``NAUTILUS_BENCH_KEYS``  — distinct keys / word vocabulary (default 1000)
* ``NAUTILUS_BENCH_SKEW`` — zipfian key-skew exponent for ``bench-skew`` (default 1.2)
* ``NAUTILUS_BENCH_DELAY_US`` — per-batch stall for ``bench-backpressure`` (default 200)
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from time import perf_counter_ns

import numpy as np
import pyarrow as pa

from nautilus.core.operator import Collector, OneInputOperator, SourceOperator
from nautilus.core.records import EOS_FRAME, Batch, Frame

#: Fixed seed for the realism knobs (skew, nulls, varied values). Fixed so a re-run reproduces
#: byte-identical input even with randomized-looking data — the structural digest stays a usable gate.
_SEED = 0x5EED


def passthrough(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Identity map — the linear benchmark's transform. A module-level function (not a lambda) so a
    ``--workers`` run can cloudpickle the operator to a spawned worker."""
    return batch


async def async_passthrough(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Identity ``fetch`` for the async-transform benchmark: one event-loop yield (so the overlap/reorder
    machinery actually runs) then the batch unchanged — no real I/O, so what is measured is the async
    loop's per-batch engine overhead, the async analog of :func:`passthrough`. Identity output keeps the
    structural digest stable. Module-level (not a lambda) so a ``--workers`` run can cloudpickle it.
    """
    await asyncio.sleep(0)
    return batch


async def async_io_wait(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Latency ``fetch`` for the I/O-bound async-transform benchmark: sleeps ``NAUTILUS_BENCH_FETCH_US``
    microseconds (default 1000) to stand in for a real external lookup, then returns the batch unchanged.
    Where :func:`async_passthrough` isolates the engine's per-batch overhead, this exercises the thing an
    async transform exists for — *overlapping* awaited I/O — so throughput should approach
    ``max_in_flight`` batches per fetch-latency, far above the serial `1 / latency`.

    ``NAUTILUS_BENCH_SLOW_EVERY`` (default 0 = uniform) makes roughly one batch in N sleep
    ``NAUTILUS_BENCH_SLOW_FACTOR``× longer (default 20), keyed off the batch's first key so it stays
    deterministic. That skew is the head-of-line blocking that separates ordered from unordered emission:
    under ordered a slow head pins buffer slots that finished tails could reuse, so ``ordered=False``
    (completion order) reads further ahead and runs faster. Identity output keeps the structural digest
    stable — latency, and hence emission order, is not a digest input — so this benchmarks safely either
    way; module-level so a ``--workers`` run can cloudpickle it."""
    base_us = _env_int("NAUTILUS_BENCH_FETCH_US", 1000)
    slow_every = _env_int("NAUTILUS_BENCH_SLOW_EVERY", 0)
    if slow_every and int(batch.column(0)[0].as_py()) % slow_every == 0:
        base_us *= _env_int("NAUTILUS_BENCH_SLOW_FACTOR", 20)
    await asyncio.sleep(base_us / 1_000_000)
    return batch


class SlowMap(OneInputOperator):
    """Busy-spins ``delay_micros`` per batch, then emits it unchanged — a deterministic CPU-bound
    consumer. Behind a fast source it forces backpressure: the bounded channel fills, the producer stalls,
    and ``edge.queue_depth_hist`` / ``edge.send_wait_micros`` (and, across a socket, ``edge.credit_wait_micros``)
    finally populate. Identity output, so the structural digest is unaffected — only timing is."""

    def __init__(self, delay_micros: int) -> None:
        self._delay_ns = delay_micros * 1000

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        deadline = (
            perf_counter_ns() + self._delay_ns
        )  # busy-wait (not sleep): a real CPU-bound stage
        while perf_counter_ns() < deadline:
            pass
        out.emit(batch)


# Default scale, shared by bench_params and the `nautilus bench` harness so both agree on one baseline.
DEFAULT_ROWS = 1_000_000
DEFAULT_BATCH = 4096
DEFAULT_KEYS = 1000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def bench_params() -> dict[str, int]:
    """Resolve the benchmark scale from the environment (with defaults). One place so every builder and
    any reporting agent reads the same knobs."""
    rows = _env_int("NAUTILUS_BENCH_ROWS", DEFAULT_ROWS)
    batch_rows = _env_int("NAUTILUS_BENCH_BATCH", DEFAULT_BATCH)
    return {
        "rows": rows,
        "batch_rows": batch_rows,
        "num_batches": max(1, -(-rows // batch_rows)),  # ceil; at least one batch
        "key_cardinality": _env_int("NAUTILUS_BENCH_KEYS", DEFAULT_KEYS),
    }


class SyntheticKeyedSource(SourceOperator):
    """A configurable keyed stream. With every realism knob at its default it is the simple, clean
    stream: ``key`` (``row_index % key_cardinality``), ``value`` (constant 1), ``ts`` (the row index),
    then EOS. The knobs make it resemble a real stream, each isolating one stressor the clean stream
    cannot exercise:

    * ``skew`` — > 0 draws keys from a zipfian distribution (exponent ``skew``) instead of uniform, so a
      few keys are hot. This is the classic source of partition imbalance: across a parallel shuffle one
      instance gets most of the rows (visible as per-subtask ``operator.rows_in`` / ``process_micros`` skew).
    * ``value_spread`` — > 0 makes ``value`` vary over ``[1, value_spread]`` instead of a constant, so an
      aggregation sums real spread.
    * ``null_fraction`` — > 0 makes that fraction of keys null (a real stream has missing keys).
    * ``payload_bytes`` — > 0 adds a fixed-width ``payload`` string column, widening the schema and the
      bytes an Arrow-IPC frame carries across a socket.
    * ``extra_value_cols`` — adds N constant int columns (cheap schema width).

    Randomized knobs draw from a fixed-seed generator, so the stream is still byte-for-byte reproducible
    and the structural digest remains a valid correctness gate."""

    def __init__(
        self,
        *,
        num_batches: int,
        batch_rows: int,
        key_cardinality: int,
        extra_value_cols: int = 0,
        skew: float = 0.0,
        value_spread: int = 0,
        null_fraction: float = 0.0,
        payload_bytes: int = 0,
    ) -> None:
        self._num_batches = num_batches
        self._batch_rows = batch_rows
        self._key_cardinality = key_cardinality
        self._extra_value_cols = extra_value_cols
        self._skew = skew
        self._value_spread = value_spread
        self._null_fraction = null_fraction
        self._payload_bytes = payload_bytes
        if skew > 0:  # zipfian PMF over [0, K): key 0 is hottest, mass ∝ 1 / (rank ** skew)
            weights = np.arange(1, key_cardinality + 1, dtype=np.float64) ** -skew
            self._pmf = weights / weights.sum()

    async def frames(self) -> AsyncIterator[Frame]:
        n = self._batch_rows
        ones = pa.array(np.ones(n, dtype=np.int64))
        payload = pa.array(["x" * self._payload_bytes] * n) if self._payload_bytes else None
        # Always built; the default (all-knobs-off) path never draws from it, so its output is unchanged.
        rng = np.random.default_rng(_SEED)
        next_ts = 0
        for _b in range(self._num_batches):
            base = np.arange(next_ts, next_ts + n, dtype=np.int64)
            if self._skew > 0:
                key_ids = rng.choice(self._key_cardinality, size=n, p=self._pmf)
            else:
                key_ids = base % self._key_cardinality
            mask = rng.random(n) < self._null_fraction if self._null_fraction > 0 else None
            value = (
                pa.array(rng.integers(1, self._value_spread + 1, size=n))
                if self._value_spread > 0
                else ones
            )
            columns: dict[str, pa.Array] = {
                "key": pa.array(key_ids, mask=mask),
                "value": value,
                "ts": pa.array(base),
            }
            if payload is not None:
                columns["payload"] = payload
            for c in range(self._extra_value_cols):
                columns[f"v{c}"] = ones
            yield Batch(pa.record_batch(columns))
            next_ts += n
        yield EOS_FRAME


class SyntheticTextSource(SourceOperator):
    """Yields ``num_batches`` batches of ``rows_per_batch`` lines, each line ``tokens_per_row`` space-
    separated words drawn round-robin from a ``vocabulary``-sized set (``w0``..``w{vocabulary-1}``).
    Feeds the tokenize → keyed-count fan-out: one short input row explodes into many output rows, so the
    output batch-size histogram and a keyed shuffle on ``word`` both run hot. Deterministic and unpaced.
    """

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


class SyntheticJoinStreamSource(SourceOperator):
    """The large *probe* side of the join benchmark: a ``key`` that recurs over ``[0, key_cardinality)``
    across every batch — the way a real streaming join re-touches each key many times, not once — plus a
    constant ``lval`` payload. ``num_batches`` × ``batch_rows`` rows, then EOS. Deterministic and
    unpaced."""

    def __init__(self, *, num_batches: int, batch_rows: int, key_cardinality: int) -> None:
        self._num_batches = num_batches
        self._batch_rows = batch_rows
        self._key_cardinality = key_cardinality

    async def frames(self) -> AsyncIterator[Frame]:
        n = self._batch_rows
        lval = pa.array(np.ones(n, dtype=np.int64))
        idx = 0
        for _ in range(self._num_batches):
            keys = pa.array(np.arange(idx, idx + n) % self._key_cardinality)
            yield Batch(pa.record_batch({"key": keys, "lval": lval}))
            idx += n
        yield EOS_FRAME


class SyntheticJoinTableSource(SourceOperator):
    """The small bounded *build* side of the join benchmark: each key ``0..key_cardinality-1`` exactly
    once with an ``rval`` payload, in one batch, then EOS — the dimension table a stream-table join
    enriches against. Every probe row matches exactly one table row, so it is a 1:1 join whose output row
    count equals the stream's: the cost measured is the join's internals, not a cross-product blow-up.
    """

    def __init__(self, *, key_cardinality: int) -> None:
        self._key_cardinality = key_cardinality

    async def frames(self) -> AsyncIterator[Frame]:
        keys = pa.array(np.arange(self._key_cardinality))
        yield Batch(pa.record_batch({"key": keys, "rval": keys}))
        yield EOS_FRAME
