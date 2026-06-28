"""The synthetic benchmark sources: deterministic, env-scaled, and shaped for the perf hot paths."""

import asyncio
from collections import Counter

import pyarrow as pa

from nautilus.benchmarks import SlowMap, SyntheticKeyedSource, SyntheticTextSource, bench_params
from nautilus.core.operator import ListCollector
from nautilus.core.records import EOS, Batch, Watermark
from nautilus.pipelines import (
    bench_backpressure,
    bench_fanout,
    bench_keyed,
    bench_late,
    bench_linear,
    bench_skew,
)
from nautilus.runtime.local import run_local_chain


async def _drain(source):
    return [f async for f in source.frames()]


def _batches(source):
    return [f.data for f in asyncio.run(_drain(source)) if isinstance(f, Batch)]


async def test_keyed_source_shape_keys_and_watermarks():
    src = SyntheticKeyedSource(num_batches=6, batch_rows=10, key_cardinality=4, wm_every=2)
    frames = await _drain(src)
    batches = [f for f in frames if isinstance(f, Batch)]
    assert len(batches) == 6
    assert all(b.data.num_rows == 10 for b in batches)
    assert batches[0].data.schema.names == ["key", "value", "ts"]
    # ts is the global row index (strictly monotonic across batches); keys cycle 0..3.
    assert batches[0].data.column("ts").to_pylist() == list(range(10))
    assert batches[1].data.column("ts").to_pylist() == list(range(10, 20))
    assert set(batches[0].data.column("key").to_pylist()) == {0, 1, 2, 3}
    # a watermark every 2 batches, then exactly one EOS to end the bounded stream.
    assert sum(isinstance(f, Watermark) for f in frames) == 3
    assert isinstance(frames[-1], EOS)


async def test_text_source_explodes_into_tokens():
    src = SyntheticTextSource(num_batches=2, rows_per_batch=5, tokens_per_row=8, vocabulary=4)
    batches = [f for f in await _drain(src) if isinstance(f, Batch)]
    assert len(batches) == 2
    first_line = batches[0].data.column("line").to_pylist()[0]
    assert first_line.split() == ["w0", "w1", "w2", "w3", "w0", "w1", "w2", "w3"]


def test_bench_params_reads_environment(monkeypatch):
    monkeypatch.setenv("NAUTILUS_BENCH_ROWS", "10000")
    monkeypatch.setenv("NAUTILUS_BENCH_BATCH", "1000")
    p = bench_params()
    assert p["rows"] == 10000 and p["batch_rows"] == 1000 and p["num_batches"] == 10


def test_extra_value_cols_widen_the_schema():
    src = SyntheticKeyedSource(num_batches=1, batch_rows=4, key_cardinality=2, extra_value_cols=3)
    batch = next(f for f in asyncio.run(_drain(src)) if isinstance(f, Batch))
    assert batch.data.schema.names == ["key", "value", "ts", "v0", "v1", "v2"]


def test_skew_concentrates_rows_on_a_few_hot_keys():
    counts = Counter()
    for b in _batches(
        SyntheticKeyedSource(num_batches=8, batch_rows=4096, key_cardinality=200, skew=1.2)
    ):
        counts.update(b.column("key").to_pylist())
    total = sum(counts.values())
    assert counts.most_common(1)[0][1] / total > 0.05  # hottest key >> uniform 1/200 = 0.5%
    assert len(counts) > 1  # still more than one key (it is skewed, not collapsed)


def test_jitter_makes_timestamps_out_of_order():
    ordered = _batches(SyntheticKeyedSource(num_batches=1, batch_rows=256, key_cardinality=8))[0]
    ts = ordered.column("ts").to_pylist()
    assert ts == sorted(ts)  # default stream is perfectly ordered
    jittered = _batches(
        SyntheticKeyedSource(num_batches=1, batch_rows=256, key_cardinality=8, jitter=30)
    )[0]
    jts = jittered.column("ts").to_pylist()
    assert jts != sorted(jts) and min(jts) >= 0  # out of order, never negative


def test_watermark_lag_holds_the_watermark_behind_the_latest_event():
    frames = asyncio.run(
        _drain(
            SyntheticKeyedSource(
                num_batches=2, batch_rows=10, key_cardinality=4, wm_every=2, watermark_lag=5
            )
        )
    )
    wms = [f.t for f in frames if isinstance(f, Watermark)]
    assert wms == [20 - 1 - 5]  # without lag it would be the latest index, 19


def test_nulls_varied_values_and_payload_widen_the_data():
    batch = _batches(
        SyntheticKeyedSource(
            num_batches=1,
            batch_rows=2000,
            key_cardinality=50,
            null_fraction=0.3,
            value_spread=100,
            payload_bytes=16,
        )
    )[0]
    assert batch.column("key").null_count > 0  # some keys are missing
    assert len(set(batch.column("value").to_pylist())) > 1  # values vary, not constant 1
    assert "payload" in batch.schema.names and len(batch.column("payload")[0].as_py()) == 16


def test_slowmap_busy_waits_and_passes_the_batch_through():
    op = SlowMap(2000)  # 2 ms
    out = ListCollector()
    batch = pa.record_batch({"x": pa.array([1, 2, 3])})
    from time import perf_counter_ns

    t0 = perf_counter_ns()
    op.process(batch, out)
    assert (perf_counter_ns() - t0) >= 1_000_000  # busy-waited at least ~1 ms of the 2 ms
    assert out.buffer[0] is batch  # identity: output unchanged


def test_benchmark_pipelines_are_deterministic(monkeypatch):
    monkeypatch.setenv("NAUTILUS_BENCH_ROWS", "8000")
    monkeypatch.setenv("NAUTILUS_BENCH_BATCH", "1000")
    monkeypatch.setenv("NAUTILUS_BENCH_KEYS", "50")
    builders = (bench_keyed, bench_linear, bench_fanout, bench_skew, bench_late, bench_backpressure)
    for builder in builders:
        digests = set()
        for _ in range(3):
            src, ops = builder()
            digests.add(asyncio.run(run_local_chain(src, ops)).telemetry.structural_digest())
        assert len(digests) == 1, f"{builder.__name__} is not deterministic: {digests}"
