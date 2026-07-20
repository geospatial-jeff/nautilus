"""Runnable example pipelines, and a loader so the CLI can run them by name.

A *pipeline* is just ``(source, transforms)`` — what ``run_local_chain`` takes. The CLI can run a
built-in example by name, or your own pipeline given as ``module:function`` (a zero-arg function that
returns ``(source, transforms)``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pyarrow as pa

from nautilus.api import LogicalGraph
from nautilus.benchmarks import (
    GEO_REGIONS,
    SlowMap,
    SyntheticGridSource,
    SyntheticJoinStreamSource,
    SyntheticJoinTableSource,
    SyntheticKeyedSource,
    SyntheticTextSource,
    anomaly_subtract,
    async_io_wait,
    async_passthrough,
    bench_params,
    geo_bench_params,
    make_region_tagger,
    ndvi_map,
    passthrough,
    squared_error,
)
from nautilus.core.operator import OneInputOperator, SourceOperator
from nautilus.demos import DemoStreamSource
from nautilus.dsl import source as dsl_source
from nautilus.operators import (
    KeyedCount,
    KeyedMean,
    MapBatch,
    Tokenize,
    from_batches,
)
from nautilus.tensors import embedding_array, tensor_array, to_numpy
from nautilus.testing import data

Pipeline = tuple[SourceOperator, list[OneInputOperator]]
Builder = Callable[[], Pipeline]


def wordcount() -> Pipeline:
    """Bounded word-count over a small in-memory text stream."""
    source = from_batches(
        data(line=["the quick brown fox", "the lazy dog"]),
        data(line=["the fox jumped", "the lazy fox ran"]),
    )
    return source, [Tokenize("line", "word"), KeyedCount("word")]


def demo_stream() -> Pipeline:
    """A long-running stream for the live dashboard (emits for a while, then ends): a keyed count over a
    paced :class:`~nautilus.demos.DemoStreamSource`."""
    return DemoStreamSource(), [KeyedCount("key")]


_TILE_H, _TILE_W, _TILE_C = 8, 8, 3


def _image_tiles(n: int, start_id: int) -> pa.RecordBatch:
    rng = np.random.default_rng(start_id)
    images = rng.integers(0, 256, size=(n, _TILE_H, _TILE_W, _TILE_C), dtype=np.uint8)
    tile_ids = pa.array(range(start_id, start_id + n), pa.int64())
    return pa.record_batch({"tile_id": tile_ids, "image": tensor_array(images)})


def _embed_tiles(batch: pa.RecordBatch) -> pa.RecordBatch:
    images = to_numpy(batch.column("image"))  # (N, H, W, C) uint8
    vectors = images.mean(axis=(1, 2)).astype(np.float32)  # (N, C): per-tile channel means
    return pa.record_batch(
        {"tile_id": batch.column("tile_id"), "embedding": embedding_array(vectors)}
    )


def image_embed() -> Pipeline:
    """Image tiles in, one embedding per tile out, using fixed_shape_tensor columns."""
    source = from_batches(_image_tiles(4, 0), _image_tiles(3, 100))
    return source, [MapBatch(_embed_tiles)]


def _load_example_builder(filename: str, fn_name: str) -> Callable[..., Any]:
    """Load a builder defined in an ``examples/`` file by path. Those files aren't an installed package
    (and ship only in a source checkout, not the wheel), so the CLI reaches a heavier example — one that
    pulls an optional extra — without that extra's imports landing at ``nautilus.pipelines`` import time.
    """
    path = Path(__file__).resolve().parents[2] / "examples" / filename
    if not path.exists():
        raise ImportError(
            f"example file {path} not found (examples/ ships in a source checkout only)"
        )
    spec = importlib.util.spec_from_file_location(f"nautilus_example_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # so the example's @dataclass can resolve its own annotations
    spec.loader.exec_module(module)
    # A file-loaded example isn't importable by name, so cloudpickle would pickle its operators by
    # reference to this synthetic module — and a spawned worker, which never loaded the file, couldn't
    # resolve it (ModuleNotFoundError on plan load). Registering the module pickles those operators by
    # value instead, so the plan is self-contained and runs on any worker, local or remote.
    cloudpickle.register_pickle_by_value(module)
    builder: Callable[..., Any] = getattr(module, fn_name)
    return builder


def sentinel2_ndvi(parallelism: int = 1) -> LogicalGraph:
    """Average NDVI over a Sentinel-2 L2A scene read straight from cloud COGs, as a *graph* pipeline: STAC
    item ids -> async open + range-read + decode + NDVI reduce (the awaiting transform) -> average per scene
    (keyed reduce). A graph, not a linear ``(source, transforms)``, because the decode is an awaiting
    transform; the NDVI reduction is fused into it so only per-tile partials, not raw pixels, cross to the
    reduce. Needs the geo extra (``pip install 'nautilus[geo]'``) and network; see examples/sentinel2_ndvi.py.
    """
    build = _load_example_builder("sentinel2_ndvi.py", "sentinel2_ndvi")
    graph: LogicalGraph = build(
        parallelism=parallelism
    )  # the example's builder returns a LogicalGraph
    return graph


def bench_keyed() -> Pipeline:
    """Benchmark: a large keyed count — the keyed-shuffle + per-key-state + end-of-stream-flush hot path.
    Scale via NAUTILUS_BENCH_* (default 1M rows, 1000 keys). Run parallel to exercise the shuffle:
    ``nautilus run bench-keyed --parallelism 4``."""
    p = bench_params()
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
    )
    return source, [KeyedCount("key")]


def bench_linear() -> Pipeline:
    """Benchmark: a linear identity pipeline (source -> map -> sink), no shuffle or state. Isolates the
    per-batch runtime overhead — mailbox fan-in, the send path, telemetry. Sweep NAUTILUS_BENCH_BATCH to
    see how throughput scales with batch size."""
    p = bench_params()
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
    )
    return source, [MapBatch(passthrough)]


def bench_chain() -> Pipeline:
    """Benchmark: two keyless identity stages (source -> map -> map -> sink) — the shape whose middle
    edge is a same-width keyless hop, which `bench-linear` (one stage) has not got. That edge forwards
    (sender i -> instance i, co-located) rather than round-robining, so at `--workers` > 1 it crosses no
    socket; round-robin would send half the rows over one. Run it distributed to exercise the forward
    edge: `nautilus bench bench-chain --workers 2 --parallelism 4 --label bench-chain-dist`. Each row
    carries a fixed 256-byte payload so the difference is real wire bytes, not just per-batch overhead —
    the lever that separates the forward and round-robin cost on the inter-stage edge."""
    p = bench_params()
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
        payload_bytes=256,  # fixed so bench-check reproduces the workload; makes the shuffle cost real
    )
    return source, [MapBatch(passthrough), MapBatch(passthrough)]


def bench_fanout() -> Pipeline:
    """Benchmark: tokenize -> keyed count over generated text — a flat-map that explodes each input row
    into many output rows, then a keyed shuffle on the word. Stresses per-row Python in Tokenize and the
    outbound batch-size growth."""
    p = bench_params()
    # Fewer, wider rows: each line is key_cardinality tokens, so the vocabulary is fully covered.
    rows = max(1, p["rows"] // max(1, p["key_cardinality"]))
    source = SyntheticTextSource(
        num_batches=max(1, -(-rows // 256)),
        rows_per_batch=256,
        tokens_per_row=p["key_cardinality"],
        vocabulary=p["key_cardinality"],
    )
    return source, [Tokenize("line", "word"), KeyedCount("word")]


def bench_skew() -> Pipeline:
    """Benchmark: hot-key (zipfian) keyed count — partition skew, the classic distributed killer. Run
    parallel to watch one instance take most of the rows: `nautilus run bench-skew --parallelism 4`. Tune
    the skew with NAUTILUS_BENCH_SKEW (exponent; higher = hotter, default 1.2)."""
    p = bench_params()
    skew = float(os.environ.get("NAUTILUS_BENCH_SKEW", "1.2"))
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
        skew=skew,
    )
    return source, [KeyedCount("key")]


def bench_backpressure() -> Pipeline:
    """Benchmark: a fast source feeding a deliberately slow stage, so the bounded channel saturates and
    the backpressure metrics (queue depth at capacity, send-wait, cross-process credit-wait) populate.
    Tune the per-batch stall with NAUTILUS_BENCH_DELAY_US (default 200µs)."""
    p = bench_params()
    delay = int(os.environ.get("NAUTILUS_BENCH_DELAY_US", "200"))
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
    )
    return source, [SlowMap(delay)]


def bench_join(parallelism: int = 1) -> LogicalGraph:
    """Benchmark: a stream-table inner equi-join — a large ``key``-recurring stream joined on ``key`` to a
    small bounded table (1:1 match, so output rows = stream rows). Stresses HashJoin's per-batch probe;
    at parallelism > 1 the keyed shuffle co-partitions both sides onto the same instance. Scale via
    NAUTILUS_BENCH_* (the stream is ``rows`` rows; the table is ``keys`` rows). A *graph* pipeline (two
    sources), so it is run via ``run_plan`` / ``deploy``, not the linear ``(source, transforms)`` path.
    """
    p = bench_params()
    stream = SyntheticJoinStreamSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
    )
    table = SyntheticJoinTableSource(key_cardinality=p["key_cardinality"])
    return dsl_source(stream).join(dsl_source(table), on="key").to_graph(parallelism=parallelism)


def _bench_inflight(default: int) -> int:
    """The async benchmarks' ``max_in_flight`` (env ``NAUTILUS_BENCH_INFLIGHT``). Raising it is how the
    async loop's wakeup mechanism is stressed: the cost of tracking the in-flight fetches per completion is
    what separates an O(N)-per-wakeup loop from an O(1) one."""
    raw = os.environ.get("NAUTILUS_BENCH_INFLIGHT")
    return int(raw) if raw else default


def bench_async(parallelism: int = 1) -> LogicalGraph:
    """Benchmark: a stateless async map (``.map_async``) over a large stream — the async-transform loop's
    per-batch engine overhead (a task per fetch, the reorder buffer, the wakeup), with a near-free fetch
    (:func:`~nautilus.benchmarks.async_passthrough`, no real I/O) so the loop, not the I/O, is what is
    measured. The async analog of ``bench-linear``; a *graph* pipeline because the async kind needs
    explicit edges, so it is run via ``run_plan`` / ``deploy``. Scale via NAUTILUS_BENCH_*
    (``NAUTILUS_BENCH_INFLIGHT`` sets ``max_in_flight``, default 8); raising ``--parallelism`` fans the
    I/O out that many ways."""
    p = bench_params()
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
    )
    return (
        dsl_source(source)
        .map_async(async_passthrough, max_in_flight=_bench_inflight(8))
        .to_graph(parallelism=parallelism)
    )


def bench_async_io(parallelism: int = 1) -> LogicalGraph:
    """Benchmark: a stateless async map whose fetch actually *awaits* — an I/O-bound enrich in the middle
    of a pipeline (:func:`~nautilus.benchmarks.async_io_wait` sleeps ``NAUTILUS_BENCH_FETCH_US`` µs,
    default 1000). Where ``bench-async`` measures the loop's overhead with a free fetch, this measures the
    overlap the loop exists to deliver: with ``max_in_flight`` fetches in flight, throughput should reach
    ~``max_in_flight`` batches per fetch-latency, far above serial. ``NAUTILUS_BENCH_INFLIGHT`` (default
    64) sets the concurrency; raising ``--parallelism`` fans it out further. ``NAUTILUS_BENCH_ORDERED=0`` runs it
    unordered (completion-order emission); paired with ``NAUTILUS_BENCH_SLOW_EVERY`` (skewed latency) that
    measures the unordered throughput win under head-of-line blocking. The stateless map's digest is
    identical either way, so the two runs stay comparable against one baseline."""
    p = bench_params()
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
    )
    ordered = os.environ.get("NAUTILUS_BENCH_ORDERED", "1") != "0"
    return (
        dsl_source(source)
        .map_async(async_io_wait, max_in_flight=_bench_inflight(64), ordered=ordered)
        .to_graph(parallelism=parallelism)
    )


def bench_geo_ndvi() -> Pipeline:
    """Benchmark: per-pixel NDVI ``(nir - red) / (nir + red)`` over a gridded field — a pure elementwise
    map, the streaming form of the xarray-sql suite's per-pixel band arithmetic. The geo analog of
    ``bench-linear``, but with real two-column arithmetic instead of identity. Scale via NAUTILUS_GEO_*.
    """
    p = geo_bench_params()
    src = SyntheticGridSource(
        n_days=p["n_days"],
        nlat=p["nlat"],
        nlon=p["nlon"],
        rows_per_batch=p["rows_per_batch"],
        lat_index=True,
        bands=True,
    )
    return src, [MapBatch(ndvi_map)]


def bench_geo_zonal(parallelism: int = 1) -> LogicalGraph:
    """Benchmark: ``AVG(temperature) GROUP BY latitude`` via ``.agg_by`` — a *low*-cardinality keyed mean
    (one group per latitude band, so a few huge groups). Stresses KeyedAgg's per-batch bincount fold far
    more than its end-of-stream flush. A *graph* pipeline (the ``.agg_by`` verb lowers to one), run via
    ``run_plan`` / ``deploy``; run parallel to exercise the keyed shuffle:
    ``nautilus run bench-geo-zonal --parallelism 4``. Scale via NAUTILUS_GEO_*."""
    p = geo_bench_params()
    src = SyntheticGridSource(
        n_days=p["n_days"],
        nlat=p["nlat"],
        nlon=p["nlon"],
        rows_per_batch=p["rows_per_batch"],
        lat_index=True,
    )
    return (
        dsl_source(src)
        .agg_by("lat_idx", mean_k=("value", "mean"), parallelism=parallelism)
        .to_graph(parallelism=parallelism)
    )


def bench_geo_climatology(parallelism: int = 1) -> LogicalGraph:
    """Benchmark: ``AVG(temperature) GROUP BY (latitude, longitude, hour)`` via ``.agg_by`` — a
    *high*-cardinality keyed mean (hundreds of thousands of small groups, encoded in one integer ``gid``
    key). The hardest aggregation in the suite, and the one the vectorized KeyedAgg fast path targets: every
    group survives to the end-of-stream flush. A *graph* pipeline, run via ``run_plan`` / ``deploy``. Scale
    via NAUTILUS_GEO_* (``NAUTILUS_GEO_DAYS`` sets the group size)."""
    p = geo_bench_params()
    src = SyntheticGridSource(
        n_days=p["n_days"],
        nlat=p["nlat"],
        nlon=p["nlon"],
        rows_per_batch=p["rows_per_batch"],
    )
    return (
        dsl_source(src)
        .agg_by("gid", mean_k=("value", "mean"), parallelism=parallelism)
        .to_graph(parallelism=parallelism)
    )


def bench_keyed_mean(parallelism: int = 1) -> LogicalGraph:
    """Benchmark: the same high-cardinality ``AVG(value) GROUP BY gid`` as ``bench-geo-climatology`` (~380k
    groups over the grid), but through the specialized :class:`~nautilus.operators.KeyedMean` operator
    (applied with ``.apply``) instead of ``.agg_by`` / KeyedAgg. It exercises KeyedMean's own value-indexed
    fast path — the one path ``.agg_by`` never reaches — at the sparse shape (every batch's ``gid`` spans
    the full key space), so it gates the shape-aware fold in KeyedMean the way climatology does for KeyedAgg.
    A *graph* pipeline. Scale via NAUTILUS_GEO_*."""
    p = geo_bench_params()
    src = SyntheticGridSource(
        n_days=p["n_days"],
        nlat=p["nlat"],
        nlon=p["nlon"],
        rows_per_batch=p["rows_per_batch"],
    )
    return (
        dsl_source(src)
        .apply(KeyedMean("gid", "value", "mean_k"), key_columns="gid")
        .to_graph(parallelism=parallelism)
    )


def bench_geo_zonal_vector(parallelism: int = 1) -> LogicalGraph:
    """Benchmark: ``AVG(temperature) ... JOIN regions ON lat/lon BETWEEN`` — a raster×vector range join
    written as a broadcast region-tag map feeding ``.agg_by`` (one group per region box). Stresses the
    per-batch numpy range test in the tagger and a tiny-cardinality keyed mean. A *graph* pipeline, run via
    ``run_plan`` / ``deploy``. Scale via NAUTILUS_GEO_*."""
    p = geo_bench_params()
    src = SyntheticGridSource(
        n_days=p["n_days"],
        nlat=p["nlat"],
        nlon=p["nlon"],
        rows_per_batch=p["rows_per_batch"],
        coords=True,
    )
    tag = make_region_tagger(GEO_REGIONS, "value")
    return (
        dsl_source(src)
        .map(tag, parallelism=parallelism)
        .agg_by("region_id", mean_k=("value", "mean"), parallelism=parallelism)
        .to_graph(parallelism=parallelism)
    )


def bench_geo_anomaly(parallelism: int = 1) -> LogicalGraph:
    """Benchmark: a per-cell temperature anomaly = observation − its own ``(lat, lon, hour)`` climatology —
    a self-join. The raw field is aggregated to a per-``gid`` climatology (``.agg_by``), then that
    climatology is joined back to the same raw field on ``gid`` (HashJoin, a large-to-large co-partitioned
    join) and subtracted. A *graph* pipeline (two sources over one field), so it is run via ``run_plan`` /
    ``deploy``. Scale via NAUTILUS_GEO_*."""
    p = geo_bench_params()

    def grid() -> SyntheticGridSource:
        return SyntheticGridSource(
            n_days=p["n_days"],
            nlat=p["nlat"],
            nlon=p["nlon"],
            rows_per_batch=p["rows_per_batch"],
        )

    clim = dsl_source(grid()).agg_by("gid", clim=("value", "mean"), parallelism=parallelism)
    return (
        dsl_source(grid())
        .join(clim, on="gid", parallelism=parallelism)
        .map(anomaly_subtract, parallelism=parallelism)
        .to_graph(parallelism=parallelism)
    )


def bench_geo_forecast(parallelism: int = 1) -> LogicalGraph:
    """Benchmark: forecast skill = RMSE of each forecast against truth, grouped by model×lead — a join then
    a keyed mean. Many model×lead forecast copies of a field (each a ``replica`` id) are joined to one truth
    field on the cell id ``gid`` (HashJoin), the squared error is taken, and a per-``replica`` mean
    (``.agg_by``) gives the mean squared error (its root is RMSE). A *graph* pipeline (forecast + truth
    sources), run via ``run_plan`` / ``deploy``. ``NAUTILUS_GEO_REPLICAS`` sets the model×lead count
    (default 8); scale the grid via NAUTILUS_GEO_*."""
    p = geo_bench_params()
    replicas = int(os.environ.get("NAUTILUS_GEO_REPLICAS", "8"))
    # One day, so gid is unique per (cell, hour): truth has one row per gid and each forecast replica one,
    # making the join a clean replicas:1 — the many forecasts against the single truth field.
    truth = SyntheticGridSource(
        n_days=1,
        nlat=p["nlat"],
        nlon=p["nlon"],
        rows_per_batch=p["rows_per_batch"],
        value_col="temp_e",
    )
    forecast = SyntheticGridSource(
        n_days=1,
        nlat=p["nlat"],
        nlon=p["nlon"],
        rows_per_batch=p["rows_per_batch"],
        value_col="temp_f",
        replicas=replicas,
    )
    return (
        dsl_source(forecast)
        .join(dsl_source(truth), on="gid", parallelism=parallelism)
        .map(squared_error, parallelism=parallelism)
        .agg_by("replica", mse=("se", "mean"), parallelism=parallelism)
        .to_graph(parallelism=parallelism)
    )


EXAMPLES: dict[str, Builder] = {
    "wordcount": wordcount,
    "demo-stream": demo_stream,
    "image-embed": image_embed,
    "bench-keyed": bench_keyed,
    "bench-linear": bench_linear,
    "bench-chain": bench_chain,
    "bench-fanout": bench_fanout,
    "bench-skew": bench_skew,
    "bench-backpressure": bench_backpressure,
    "bench-geo-ndvi": bench_geo_ndvi,
}

#: Graph pipelines are a LogicalGraph the harness runs with run_plan/deploy rather than the linear
#: (source, transforms) an EXAMPLES entry is — either because they have more than one source (a join) or
#: because they use a kind only explicit edges express (an async transform).
GraphBuilder = Callable[[int], LogicalGraph]
GRAPH_EXAMPLES: dict[str, GraphBuilder] = {
    "sentinel2-ndvi": sentinel2_ndvi,
    "bench-join": bench_join,
    "bench-async": bench_async,
    "bench-async-io": bench_async_io,
    "bench-geo-zonal": bench_geo_zonal,
    "bench-geo-climatology": bench_geo_climatology,
    "bench-keyed-mean": bench_keyed_mean,
    "bench-geo-zonal-vector": bench_geo_zonal_vector,
    "bench-geo-anomaly": bench_geo_anomaly,
    "bench-geo-forecast": bench_geo_forecast,
}


def is_graph_pipeline(spec: str) -> bool:
    """Whether ``spec`` names a graph pipeline — one run via run_plan/deploy from a :class:`LogicalGraph`
    rather than a linear ``(source, transforms)`` — because it has more than one source (a join) or an
    awaiting async stage (the Sentinel-2 example, the async benchmarks)."""
    return spec in GRAPH_EXAMPLES


def load_graph_pipeline(spec: str, parallelism: int) -> LogicalGraph:
    """Build a graph pipeline's :class:`LogicalGraph` at the given operator parallelism."""
    return GRAPH_EXAMPLES[spec](parallelism)


def load_pipeline(spec: str) -> Pipeline:
    """Resolve ``spec`` to ``(source, transforms)``: a built-in example name, or ``module:function``."""
    if spec in EXAMPLES:
        return EXAMPLES[spec]()
    if ":" in spec:
        module_name, fn_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        builder: Builder = getattr(module, fn_name)
        return builder()
    raise KeyError(
        f"unknown pipeline {spec!r}; use a built-in ({', '.join(EXAMPLES)}) or 'module:function'"
    )
