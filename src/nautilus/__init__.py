"""Nautilus: a decentralized, entirely-streaming parallel compute framework.

The fluent :class:`~nautilus.dsl.Stream` DSL is the readable way to build and run a pipeline::

    import pyarrow as pa
    from nautilus import source

    lines = pa.record_batch({"line": ["the quick brown fox", "the lazy dog"]})
    result = source(lines).tokenize("line", "word").count_by("word").run()
    print(result.to_pylist())

``.run(workers=N)`` deploys the *same* graph across N worker processes; ``.join`` combines two streams
into an inner equi-join::

    joined = source(orders).join(source(customers), on="customer_id").run()

For the simplest case the one-liner :func:`run` takes a source and a list of operators directly —
``run(from_batches(lines), [Tokenize("line", "word"), KeyedCount("word")])``. ``from_batches`` wraps a
bare ``pyarrow.RecordBatch`` and appends the terminal :data:`EOS_FRAME`; for an event-time stream it also
accepts :class:`Batch` / :class:`Watermark` frames. Reach for ``InMemorySource([...])`` only when you need
exact frame control. (:func:`run_local_chain` is the ``await``-able form of ``run`` for an async caller.)
Anything not re-exported here is still importable from its concrete module (e.g. ``nautilus.dsl``,
``nautilus.operators``, ``nautilus.telemetry``).
"""

from nautilus._version import __version__
from nautilus.core.operator import (
    AsyncSink,
    Collector,
    OneInputOperator,
    OperatorContext,
    SourceOperator,
)
from nautilus.core.records import EOS_FRAME, Batch, Watermark
from nautilus.driver.local import run, run_local_chain
from nautilus.driver.result import RunResult
from nautilus.dsl import SinkHandle, Stream, source
from nautilus.operators import (
    FilterRows,
    HashJoin,
    InMemorySource,
    KeyedCount,
    KeyedTumblingSum,
    MapBatch,
    Tokenize,
    from_batches,
)
from nautilus.telemetry import TelemetryConfig, Tier
from nautilus.tensors import embedding_array, tensor_array, tensor_type, to_numpy

__all__ = [
    "__version__",
    # the fluent DSL (the primary way to build a pipeline)
    "Stream",
    "SinkHandle",
    "source",
    # runners
    "run",
    "run_local_chain",
    "RunResult",
    # authoring a source / operator / sink
    "SourceOperator",
    "OneInputOperator",
    "AsyncSink",
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
    "HashJoin",
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
