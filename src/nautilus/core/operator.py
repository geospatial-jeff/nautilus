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
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter_ns
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
    StateScope,
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
    """What an operator is handed at ``open`` time: its ``operator_id``, this instance's
    ``subtask_index`` of ``num_subtasks``, the state backend, the clock, and a custom-metric recorder.
    """

    operator_id: str
    subtask_index: int = 0
    num_subtasks: int = 1
    state_backend: StateBackend = field(default_factory=_InMemoryStateBackend)
    clock: Clock = field(default_factory=SystemClock)
    #: Operator-author custom-metric recorder — a SEPARATE recorder from the actor's built-in one, so
    #: the single-writer invariant is never violated. Defaults to a zero-cost no-op. A custom metric must
    #: be declared in the catalog with ``owner=Owner.AUTHOR`` (every metric is catalog-declared, and this
    #: recorder may write only author-owned ones).
    metrics: Recorder = NULL_RECORDER

    def value_state(self, name: str, kctx: KeyContext) -> ValueState[Any]:
        return ValueState(self.state_backend, self.operator_id, name, kctx)

    def reducing_state(self, name: str, kctx: KeyContext, reducer: Any) -> ReducingState[Any]:
        return ReducingState(self.state_backend, self.operator_id, name, kctx, reducer)

    def entries(self, name: str) -> Iterator[tuple[KeyContext, object]]:
        """Iterate ``(KeyContext, value)`` for every entry of this operator's named state — the
        flush-time counterpart to :meth:`value_state` / :meth:`reducing_state`. Operator code uses this
        to enumerate all keys/windows at a watermark without naming its own ``operator_id`` or building a
        ``StateScope`` by hand. The snapshot is stable to mutate during iteration (e.g. to clear).
        """
        for key, namespace, value in self.state_backend.entries(self.operator_id, name):
            yield KeyContext(key, namespace), value

    def clear_state(self, name: str, kctx: KeyContext) -> None:
        """Clear one entry of this operator's named state (the keyed-handle ``clear`` for a key/window
        enumerated via :meth:`entries`)."""
        self.state_backend.clear(StateScope(self.operator_id, name, kctx.key, kctx.namespace))

    @asynccontextmanager
    async def io_wait(self) -> AsyncIterator[None]:
        """Record the wall time of an awaited I/O region as ``io.wait_micros`` (an author metric).

        A :class:`SourceOperator` is the one operator that may ``await`` inside its own code, so its
        ``runtime.step_micros`` counts the I/O it waits on together with the CPU it spends building frames.
        Wrapping the network awaits — ``async with ctx.io_wait(): batch = await fetch()`` — records that
        wait on its own, so the report can tell an I/O-bound source from a compute-bound one. A no-op when
        telemetry is off (``metrics`` is then the null recorder)."""
        start = perf_counter_ns()
        try:
            yield
        finally:
            self.metrics.incr(
                "io.wait_micros", (perf_counter_ns() - start) // 1000, operator_id=self.operator_id
            )


class SourceOperator(ABC):
    """Generates the frame sequence for a stream. Has no inputs.

    ``frames()`` is an async generator, so a source can ``await`` (network I/O, ``asyncio.sleep``)
    between batches without blocking the event loop — the basis for unbounded streams. An in-memory
    source simply never awaits::

        async def frames(self):
            for frame in self._frames:
                yield frame

    It must not *block* the loop, though: offload blocking work yourself (e.g.
    ``await asyncio.to_thread(read)``). nautilus does not wrap sources in a hidden thread pool, which
    would break the single-writer-per-actor model.
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

    def key_columns(self) -> tuple[str, ...] | None:
        """The columns this operator's input must be co-partitioned on, or ``None`` if it is keyless.

        A keyed operator — one that keeps per-key state — declares its key here so that a parallel run
        routes its input through the keyed shuffle and never splits a key across instances. The default
        ``None`` means the operator is stateless-per-row, so any row may go to any instance. A pipeline
        driven by parallelism alone (the CLI) reads this to choose each edge's partitioner."""
        return None

    def close(self) -> None:
        """Called once after EOS has been fully processed. Override to release resources."""


class TwoInputOperator(ABC):
    """An operator with two input streams and one output — a join. The actor drives it like a one-input
    operator (see :func:`~nautilus.runtime.actor.run_two_input`), with one difference: a data batch
    arrives on the **left** input (:meth:`process_left`) or the **right** (:meth:`process_right`). Event
    time and termination stay the actor's job: the operator watermark is the minimum over *both* inputs
    (``min(left, right)``), :meth:`on_watermark` fires at that combined watermark, and the actor forwards
    EOS downstream only after both inputs have closed — advancing to ``WATERMARK_MAX`` first, so a final
    :meth:`on_watermark` flushes any buffered state.

    Both inputs are co-partitioned on the join key by the keyed shuffle, so every row of a given key
    reaches the same instance from either side; the operator buffers and matches per key locally.
    """

    def open(self, ctx: OperatorContext) -> None:
        """Called once before any record. Override to set up the per-side buffers / state / resources."""

    @abstractmethod
    def process_left(self, batch: pa.RecordBatch, out: Collector) -> None:
        """Handle a batch from the left input. Synchronous; emit results via ``out``; must not block/await."""

    @abstractmethod
    def process_right(self, batch: pa.RecordBatch, out: Collector) -> None:
        """Handle a batch from the right input. Synchronous; emit results via ``out``; must not block/await."""

    def on_watermark(self, t: int, out: Collector) -> None:
        """Event-time progress reached ``t`` (the minimum over both inputs). Fire due windows/timers and
        emit via ``out``. Default: no-op (a global join emits matches as they arrive)."""

    def close(self) -> None:
        """Called once after EOS on both inputs. Override to release resources / clear buffers."""
