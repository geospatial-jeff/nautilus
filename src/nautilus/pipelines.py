"""Runnable example pipelines, and a loader so the CLI can run them by name.

A *pipeline* is just ``(source, transforms)`` — what ``run_local_chain`` takes. The CLI can run a
built-in example by name, or your own pipeline given as ``module:function`` (a zero-arg function that
returns ``(source, transforms)``).
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable

import numpy as np
import pyarrow as pa

from nautilus.benchmarks import (
    SlowMap,
    SyntheticKeyedSource,
    SyntheticTextSource,
    bench_params,
    passthrough,
)
from nautilus.core.operator import OneInputOperator, SourceOperator
from nautilus.demos import DemoStreamSource
from nautilus.operators import (
    KeyedCount,
    KeyedTumblingSum,
    MapBatch,
    Tokenize,
    from_batches,
)
from nautilus.tensors import embedding_array, tensor_array, to_numpy
from nautilus.testing import data, wm
from nautilus.windows import TumblingEventTimeWindows

Pipeline = tuple[SourceOperator, list[OneInputOperator]]
Builder = Callable[[], Pipeline]


def wordcount() -> Pipeline:
    """Bounded word-count over a small in-memory text stream."""
    source = from_batches(
        data(line=["the quick brown fox", "the lazy dog"]),
        data(line=["the fox jumped", "the lazy fox ran"]),
    )
    return source, [Tokenize("line", "word"), KeyedCount("word")]


def windowed_sum() -> Pipeline:
    """Keyed tumbling-window sum over an event-time stream (windows fire on watermarks)."""
    source = from_batches(
        data(key=["a", "a", "b"], val=[1, 2, 5], ts=[1, 5, 7]),
        wm(10),
        data(key=["a", "b"], val=[10, 3], ts=[12, 14]),
        wm(20),
    )
    return source, [KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))]


def demo_stream() -> Pipeline:
    """A long-running event-time stream for the live dashboard (emits for a while, then ends)."""
    window = TumblingEventTimeWindows(1_000_000)
    return DemoStreamSource(), [KeyedTumblingSum("key", "val", "ts", window)]


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


def bench_keyed() -> Pipeline:
    """Benchmark: a large keyed tumbling-window sum — the keyed-shuffle + per-key-state + window-fire hot
    path. Scale via NAUTILUS_BENCH_* (default 1M rows, 1000 keys). Run parallel to exercise the shuffle:
    ``nautilus run bench-keyed --parallelism 4``."""
    p = bench_params()
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
        wm_every=p["wm_every"],
    )
    window = TumblingEventTimeWindows(p["batch_rows"])  # one window per batch of event time
    return source, [KeyedTumblingSum("key", "value", "ts", window)]


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
    """Benchmark: hot-key (zipfian) keyed window sum — partition skew, the classic distributed killer.
    Run parallel to watch one instance take most of the rows: `nautilus run bench-skew --parallelism 4`.
    Tune the skew with NAUTILUS_BENCH_SKEW (exponent; higher = hotter, default 1.2)."""
    p = bench_params()
    skew = float(os.environ.get("NAUTILUS_BENCH_SKEW", "1.2"))
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
        wm_every=p["wm_every"],
        skew=skew,
    )
    return source, [
        KeyedTumblingSum("key", "value", "ts", TumblingEventTimeWindows(p["batch_rows"]))
    ]


def bench_late() -> Pipeline:
    """Benchmark: out-of-order events with watermark lag (allowed lateness) and varied values — the
    event-time path a perfectly-ordered stream never exercises (late data, windows held open, state
    growth before they fire)."""
    p = bench_params()
    source = SyntheticKeyedSource(
        num_batches=p["num_batches"],
        batch_rows=p["batch_rows"],
        key_cardinality=p["key_cardinality"],
        wm_every=p["wm_every"],
        jitter=p["batch_rows"] * 2,
        watermark_lag=p["batch_rows"],
        value_spread=1000,
    )
    return source, [
        KeyedTumblingSum("key", "value", "ts", TumblingEventTimeWindows(p["batch_rows"]))
    ]


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


EXAMPLES: dict[str, Builder] = {
    "wordcount": wordcount,
    "windowed-sum": windowed_sum,
    "demo-stream": demo_stream,
    "image-embed": image_embed,
    "bench-keyed": bench_keyed,
    "bench-linear": bench_linear,
    "bench-fanout": bench_fanout,
    "bench-skew": bench_skew,
    "bench-late": bench_late,
    "bench-backpressure": bench_backpressure,
}


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
