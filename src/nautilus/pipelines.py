"""Runnable example pipelines, and a loader so the CLI can run them by name.

A *pipeline* is just ``(source, transforms)`` — what ``run_local_chain`` takes. The CLI can run a
built-in example by name, or your own pipeline given as ``module:function`` (a zero-arg function that
returns ``(source, transforms)``).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import pyarrow as pa

from nautilus.core.operator import OneInputOperator, SourceOperator
from nautilus.core.records import EOS_FRAME, Batch, Frame
from nautilus.demos import DemoStreamSource
from nautilus.operators import InMemorySource, KeyedCount, KeyedTumblingSum, MapBatch, Tokenize
from nautilus.tensors import embedding_array, tensor_array, to_numpy
from nautilus.testing import data, wm
from nautilus.windows import TumblingEventTimeWindows

Pipeline = tuple[SourceOperator, list[OneInputOperator]]
Builder = Callable[[], Pipeline]


def wordcount() -> Pipeline:
    """Bounded word-count over a small in-memory text stream."""
    frames: list[Frame] = [
        data(line=["the quick brown fox", "the lazy dog"]),
        data(line=["the fox jumped", "the lazy fox ran"]),
        EOS_FRAME,
    ]
    return InMemorySource(frames), [Tokenize("line", "word"), KeyedCount("word")]


def windowed_sum() -> Pipeline:
    """Keyed tumbling-window sum over an event-time stream (windows fire on watermarks)."""
    frames: list[Frame] = [
        data(key=["a", "a", "b"], val=[1, 2, 5], ts=[1, 5, 7]),
        wm(10),
        data(key=["a", "b"], val=[10, 3], ts=[12, 14]),
        wm(20),
        EOS_FRAME,
    ]
    return InMemorySource(frames), [
        KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))
    ]


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
    frames: list[Frame] = [Batch(_image_tiles(4, 0)), Batch(_image_tiles(3, 100)), EOS_FRAME]
    return InMemorySource(frames), [MapBatch(_embed_tiles)]


EXAMPLES: dict[str, Builder] = {
    "wordcount": wordcount,
    "windowed-sum": windowed_sum,
    "demo-stream": demo_stream,
    "image-embed": image_embed,
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
