"""The synthetic benchmark sources: deterministic, env-scaled, and shaped for the perf hot paths."""

import asyncio

from nautilus.benchmarks import SyntheticKeyedSource, SyntheticTextSource, bench_params
from nautilus.core.records import EOS, Batch, Watermark
from nautilus.pipelines import bench_fanout, bench_keyed, bench_linear
from nautilus.runtime.local import run_local_chain


async def _drain(source):
    return [f async for f in source.frames()]


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


def test_benchmark_pipelines_are_deterministic(monkeypatch):
    monkeypatch.setenv("NAUTILUS_BENCH_ROWS", "8000")
    monkeypatch.setenv("NAUTILUS_BENCH_BATCH", "1000")
    monkeypatch.setenv("NAUTILUS_BENCH_KEYS", "50")
    for builder in (bench_keyed, bench_linear, bench_fanout):
        digests = set()
        for _ in range(3):
            src, ops = builder()
            digests.add(asyncio.run(run_local_chain(src, ops)).telemetry.structural_digest())
        assert len(digests) == 1, f"{builder.__name__} is not deterministic: {digests}"
