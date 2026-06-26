"""Operator execution contracts — the code a user or built-in operator implements.

The operator types differ by how many inputs they have:

* :class:`SourceOperator` has no input; it produces the frame sequence (data, watermarks, EOS).
* :class:`OneInputOperator` transforms a single input stream.
* :class:`TwoInputOperator` (reserved for joins) combines two inputs with a min-watermark.

An actor drives each operator (see :mod:`nautilus.runtime.actor`). ``process`` and ``on_watermark``
are synchronous and must not ``await``; they emit into a :class:`Collector`, and the actor performs the
awaiting (backpressured) sends between calls. Each per-batch step therefore runs as one critical
section, which the GIL makes safe without locks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from nautilus.core.records import Frame
from nautilus.core.time import Clock, SystemClock
from nautilus.state import (
    InMemoryStateBackend as _InMemoryStateBackend,
)
from nautilus.state import (
    KeyContext,
    ReducingState,
    StateBackend,
    ValueState,
)
from nautilus.telemetry.recorder import NULL_RECORDER, Recorder


class Collector(ABC):
    """Buffer for the batches an operator emits during one synchronous step."""

    @abstractmethod
    def emit(self, batch: pa.RecordBatch) -> None: ...


class ListCollector(Collector):
    """Buffers emitted (non-empty) batches in order; the actor drains it after each step."""

    def __init__(self) -> None:
        self.buffer: list[pa.RecordBatch] = []

    def emit(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows:
            self.buffer.append(batch)

    def drain(self) -> list[pa.RecordBatch]:
        out, self.buffer = self.buffer, []
        return out


@dataclass
class OperatorContext:
    """The state backend, clock, config, and metrics recorder passed to an operator at ``open`` time."""

    operator_id: str
    subtask_index: int = 0
    num_subtasks: int = 1
    state_backend: StateBackend = field(default_factory=_InMemoryStateBackend)
    clock: Clock = field(default_factory=SystemClock)
    config: Mapping[str, Any] = field(default_factory=dict)
    #: Operator-author custom-metric recorder — a SEPARATE recorder from the actor's built-in one, so
    #: the single-writer invariant is never violated. Defaults to a zero-cost no-op.
    metrics: Recorder = NULL_RECORDER

    def value_state(self, name: str, kctx: KeyContext) -> ValueState[Any]:
        return ValueState(self.state_backend, self.operator_id, name, kctx)

    def reducing_state(self, name: str, kctx: KeyContext, reducer: Any) -> ReducingState[Any]:
        return ReducingState(self.state_backend, self.operator_id, name, kctx, reducer)


class SourceOperator(ABC):
    """Generates the frame sequence for a stream. Has no inputs.

    ``frames()`` is an async generator, so a source can ``await`` (network I/O, ``asyncio.sleep``)
    between batches without freezing the event loop — the basis for long-running / unbounded streams.
    A purely in-memory source simply never awaits::

        async def frames(self):
            for frame in self._frames:
                yield frame

    ``frames()`` must not *block* the loop. A source doing blocking work offloads it itself, e.g.
    ``rows = await asyncio.to_thread(blocking_read)`` — nautilus does not wrap source code in a hidden
    thread pool, which would break the single-writer-per-actor, deterministic-step model.
    """

    def open(self, ctx: OperatorContext) -> None:
        """Called once before iteration. Override to acquire resources."""

    @abstractmethod
    def frames(self) -> AsyncIterator[Frame]:
        """Async-yield the stream's frames in order: data :class:`~nautilus.core.records.Batch` es,
        :class:`~nautilus.core.records.Watermark` s, idleness markers, and finally exactly one
        :class:`~nautilus.core.records.EOS` for a bounded source (an unbounded source simply never
        yields EOS)."""

    def close(self) -> None:
        """Called once after iteration completes (including after cancellation). Release resources."""


class OneInputOperator(ABC):
    """An operator with one input stream and one output stream. It can implement, for example:

    - **map** — one batch in, one transformed batch out (:class:`~nautilus.operators.MapBatch`).
    - **filter** — drop rows (:class:`~nautilus.operators.FilterRows`).
    - **flat-map** — one row to many (:class:`~nautilus.operators.Tokenize`).
    - **reduce / aggregate** — accumulate per key in state and emit on a watermark
      (:class:`~nautilus.operators.KeyedCount`, :class:`~nautilus.operators.KeyedTumblingSum`).
    """

    def open(self, ctx: OperatorContext) -> None:
        """Called once before any record. Override to set up state/resources."""

    @abstractmethod
    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        """Handle a data batch. Synchronous; emit results via ``out``. Must not block/await."""

    def on_watermark(self, t: int, out: Collector) -> None:
        """Event-time progress reached ``t``. Fire due windows/timers; emit via ``out``. The actor
        forwards the watermark downstream *after* this returns. Default: no-op (stateless ops)."""

    def close(self) -> None:
        """Called once after EOS has been fully processed. Override to release resources."""


class TwoInputOperator(ABC):
    """Combines two input streams (e.g. a join). **Reserved** — implemented in Stage 3.

    The operator watermark is ``min(left, right)``; EOS is emitted only after *both* inputs close.
    """

    def open(self, ctx: OperatorContext) -> None: ...

    @abstractmethod
    def process_left(self, batch: pa.RecordBatch, out: Collector) -> None: ...

    @abstractmethod
    def process_right(self, batch: pa.RecordBatch, out: Collector) -> None: ...

    def on_watermark(self, t: int, out: Collector) -> None: ...

    def close(self) -> None: ...
