"""The synthetic benchmark sources: deterministic, env-scaled, and shaped for the perf hot paths."""

import asyncio
import math
from collections import Counter

import pyarrow as pa

from nautilus.benchmarks import (
    SlowMap,
    SyntheticGridSource,
    SyntheticKeyedSource,
    SyntheticTextSource,
    bench_params,
)
from nautilus.core.operator import ListCollector
from nautilus.core.records import EOS, EOS_FRAME, Batch
from nautilus.driver.local import run_local_chain
from nautilus.driver.run import run_plan
from nautilus.operators import InMemorySource, KeyedMean
from nautilus.pipelines import (
    bench_backpressure,
    bench_fanout,
    bench_geo_anomaly,
    bench_geo_climatology,
    bench_geo_forecast,
    bench_geo_ndvi,
    bench_geo_zonal,
    bench_geo_zonal_vector,
    bench_keyed,
    bench_linear,
    bench_skew,
)


async def _drain(source):
    return [f async for f in source.frames()]


def _batches(source):
    return [f.data for f in asyncio.run(_drain(source)) if isinstance(f, Batch)]


async def test_keyed_source_shape_and_key_distribution():
    src = SyntheticKeyedSource(num_batches=6, batch_rows=10, key_cardinality=4)
    frames = await _drain(src)
    batches = [f for f in frames if isinstance(f, Batch)]
    assert len(batches) == 6
    assert all(b.data.num_rows == 10 for b in batches)
    assert batches[0].data.schema.names == ["key", "value", "ts"]
    # ts is the global row index (strictly monotonic across batches); keys cycle 0..3.
    assert batches[0].data.column("ts").to_pylist() == list(range(10))
    assert batches[1].data.column("ts").to_pylist() == list(range(10, 20))
    assert set(batches[0].data.column("key").to_pylist()) == {0, 1, 2, 3}
    # the bounded stream is exactly the 6 data batches then one terminal EOS — no other frames.
    assert len(frames) == 7 and isinstance(frames[-1], EOS)


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
    builders = (bench_keyed, bench_linear, bench_fanout, bench_skew, bench_backpressure)
    for builder in builders:
        digests = set()
        for _ in range(3):
            src, ops = builder()
            digests.add(asyncio.run(run_local_chain(src, ops)).telemetry.structural_digest())
        assert len(digests) == 1, f"{builder.__name__} is not deterministic: {digests}"


def _geo_env(monkeypatch):
    monkeypatch.setenv("NAUTILUS_GEO_DAYS", "2")
    monkeypatch.setenv("NAUTILUS_GEO_NLAT", "8")
    monkeypatch.setenv("NAUTILUS_GEO_NLON", "10")
    monkeypatch.setenv("NAUTILUS_GEO_BATCH", "256")
    monkeypatch.setenv("NAUTILUS_GEO_REPLICAS", "3")


def test_geo_grid_source_shape_and_optional_columns():
    # gid is always present; the rest are opt-in so a pipeline emits only what it keys or reduces on.
    lean = _batches(SyntheticGridSource(n_days=1, nlat=4, nlon=5, rows_per_batch=64))
    assert [b.schema.names for b in lean][0] == ["gid", "value"]
    assert sum(b.num_rows for b in lean) == 1 * 24 * 4 * 5  # one row per (time, lat, lon) cell

    full = _batches(
        SyntheticGridSource(
            n_days=1, nlat=4, nlon=5, rows_per_batch=64, lat_index=True, coords=True, bands=True
        )
    )
    assert set(full[0].schema.names) == {
        "gid",
        "value",
        "lat_idx",
        "latitude",
        "longitude",
        "red",
        "nir",
    }

    # replicas>1 emits that many tagged copies of the grid — the many side of the forecast join.
    rep = _batches(SyntheticGridSource(n_days=1, nlat=4, nlon=5, rows_per_batch=1024, replicas=3))
    assert sum(b.num_rows for b in rep) == 3 * 24 * 4 * 5
    assert Counter(r for b in rep for r in b.column("replica").to_pylist()) == {
        0: 480,
        1: 480,
        2: 480,
    }


def _keyed_mean(batches, key_col="key", value_col="val"):
    """Run KeyedMean over a list of record batches; return {key: (mean, n)}."""
    frames = [Batch(b) for b in batches] + [EOS_FRAME]
    out = asyncio.run(run_local_chain(InMemorySource(frames), [KeyedMean(key_col, value_col, "m")]))
    return {
        k: (m, n)
        for rb in out.batches
        for k, m, n in zip(
            rb.column(key_col).to_pylist(),
            rb.column("m").to_pylist(),
            rb.column("n").to_pylist(),
            strict=True,
        )
    }


def test_keyed_mean_integer_fast_path_averages_per_key():
    b1 = pa.record_batch(
        {"key": pa.array([0, 1, 0], pa.int64()), "val": pa.array([1.0, 10.0, 3.0])}
    )
    b2 = pa.record_batch({"key": pa.array([1, 0], pa.int64()), "val": pa.array([20.0, 2.0])})
    assert _keyed_mean([b1, b2]) == {0: (2.0, 3), 1: (15.0, 2)}  # (1+3+2)/3, (10+20)/2


def test_keyed_mean_non_integer_keys_fold_through_state():
    b = pa.record_batch({"key": pa.array(["a", "b", "a"]), "val": pa.array([1.0, 5.0, 3.0])})
    assert _keyed_mean([b]) == {"a": (2.0, 2), "b": (5.0, 1)}


def test_keyed_mean_skips_null_values_but_groups_null_keys():
    # A null value joins no mean (AVG skips it); a null key is its own group (like SQL GROUP BY).
    b = pa.record_batch(
        {
            "key": pa.array([0, 0, None, 1], pa.int64()),
            "val": pa.array([2.0, None, 9.0, 4.0]),
        }
    )
    assert _keyed_mean([b]) == {0: (2.0, 1), 1: (4.0, 1), None: (9.0, 1)}


def test_keyed_mean_propagates_nan_within_a_group():
    b = pa.record_batch({"key": pa.array([0, 0], pa.int64()), "val": pa.array([1.0, float("nan")])})
    mean, n = _keyed_mean([b])[0]
    assert math.isnan(mean) and n == 2


def test_geo_linear_pipelines_are_deterministic(monkeypatch):
    _geo_env(monkeypatch)
    for builder in (bench_geo_ndvi, bench_geo_zonal, bench_geo_climatology, bench_geo_zonal_vector):
        digests = set()
        for _ in range(3):
            src, ops = builder()
            digests.add(asyncio.run(run_local_chain(src, ops)).telemetry.structural_digest())
        assert len(digests) == 1, f"{builder.__name__} is not deterministic: {digests}"


def test_geo_graph_pipelines_are_deterministic(monkeypatch):
    _geo_env(monkeypatch)
    for builder in (bench_geo_anomaly, bench_geo_forecast):
        digests = {
            asyncio.run(run_plan(builder(1))).telemetry.structural_digest() for _ in range(3)
        }
        assert len(digests) == 1, f"{builder.__name__} is not deterministic: {digests}"
