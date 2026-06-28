"""Nautilus: a decentralized, entirely-streaming parallel compute framework.

A fluent DSL is planned for a later stage; until then the curated names below are the public surface,
importable straight from the top level::

    import pyarrow as pa
    from nautilus import run, from_batches, Tokenize, KeyedCount

    source = from_batches(pa.record_batch({"line": ["the quick brown fox", "the lazy dog"]}))
    result = run(source, [Tokenize("line", "word"), KeyedCount("word")])
    print(result.to_pylist(), result.telemetry.summary)

``from_batches`` wraps a bare ``pyarrow.RecordBatch`` for you; for an event-time stream, pass
:class:`Batch` / :class:`Watermark` frames (and a terminal :data:`EOS_FRAME`) explicitly. Anything not
re-exported here is still importable from its concrete module (e.g. ``nautilus.runtime.local``,
``nautilus.operators``, ``nautilus.telemetry``) during early development.
"""

__version__ = "0.0.1"

from nautilus.core.operator import (
    Collector,
    OneInputOperator,
    OperatorContext,
    SourceOperator,
)
from nautilus.core.records import EOS_FRAME, Batch, Watermark
from nautilus.operators import (
    FilterRows,
    InMemorySource,
    KeyedCount,
    KeyedTumblingSum,
    MapBatch,
    Tokenize,
    from_batches,
)
from nautilus.runtime.local import run, run_local_chain
from nautilus.runtime.result import RunResult
from nautilus.telemetry import TelemetryConfig, Tier
from nautilus.tensors import embedding_array, tensor_array, tensor_type, to_numpy

__all__ = [
    "__version__",
    # runners
    "run",
    "run_local_chain",
    "RunResult",
    # authoring a source / operator
    "SourceOperator",
    "OneInputOperator",
    "OperatorContext",
    "Collector",
    # built-in operators + source factory
    "InMemorySource",
    "from_batches",
    "MapBatch",
    "FilterRows",
    "Tokenize",
    "KeyedCount",
    "KeyedTumblingSum",
    # data frames (for building event-time inputs by hand)
    "Batch",
    "Watermark",
    "EOS_FRAME",
    # tensor columns (imagery + embeddings)
    "tensor_array",
    "embedding_array",
    "tensor_type",
    "to_numpy",
    # telemetry configuration
    "TelemetryConfig",
    "Tier",
]
