"""Operator execution contracts — the code a user or built-in operator implements.

The operator types differ by how many inputs they have and whether they may ``await``:

* :class:`SourceOperator` has no input; it produces the frame sequence (data, watermarks, EOS).
* :class:`OneInputOperator` transforms a single input stream, synchronously.
* :class:`TwoInputOperator` (reserved for joins) combines two inputs with a min-watermark.
* :class:`AsyncOneInputOperator` transforms one input but does its I/O in an awaiting ``fetch`` the
  engine runs as bounded concurrent tasks, then folds each result into state and emits in a synchronous
  ``integrate`` — so I/O overlaps while keyed state stays single-writer.
* :class:`AsyncSink` is a terminal that writes each batch to an external store in an awaiting ``write``.

An actor drives each operator (see :mod:`nautilus.runtime.actor`). The synchronous methods —
``process``/``on_watermark``/``integrate`` — must not ``await``; they emit into a :class:`Collector` and
the actor performs the awaiting (backpressured) sends between calls, so each runs as one critical section
the GIL makes safe without locks. Only a source's ``frames``, an async transform's ``fetch``, and a
sink's ``write`` may ``await``; that awaiting half is handed no :class:`Collector` and no state, and
reaching keyed state or the recorder from it raises :class:`StateAccessError` (why: ``DESIGN.md``
mechanism 9).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
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
from nautilus.telemetry.model import Counter, Gauge, Histogram, InstanceSnapshot
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


class StateAccessError(RuntimeError):
    """Raised when an async transform's awaiting half (:meth:`AsyncOneInputOperator.fetch`) reaches keyed
    state or the metric recorder. ``fetch`` runs concurrently with sibling fetches on the event loop, so a
    read-modify-write of keyed state spanning its ``await`` would lose updates. The engine therefore closes
    the context's state and metrics for the duration of every fetch (see
    :meth:`OperatorContext._install_async_guard`), turning the contract violation into a loud failure
    instead of silent corruption. The guard wraps the *backend* and *recorder*, not merely the context
    accessors, so a state handle cached during ``integrate`` and reused in ``fetch`` raises too. A
    synchronous operator never installs the guard, so it never sees this."""


_GUARD_MSG = (
    "an async transform reached keyed state or ctx.metrics from its awaiting half (fetch); fetch is "
    "handed only its batch — do all state access and emission in integrate(batch, result, ctx, out), "
    "which the engine runs on the actor task"
)


class _StateGuard:
    """The open/closed flag an async transform's guard shares between its wrapped ``state_backend`` and
    ``metrics``. The driver installs it closed and opens it only inside the synchronous
    open/integrate/on_watermark/close steps (:meth:`OperatorContext._state_section`), so any access from a
    concurrent ``fetch`` finds it closed."""

    __slots__ = ("open",)

    def __init__(self) -> None:
        self.open = False


class _GuardedStateBackend(StateBackend):
    """Wraps the real backend so every keyed read/write raises :class:`StateAccessError` while the guard is
    closed. Because a :class:`~nautilus.state.ValueState` / :class:`~nautilus.state.ReducingState` handle
    holds *this backend*, not the context, enforcing here — not on the context accessors — is what catches
    a fetch that reaches state through a handle it cached in ``integrate``. ``sizes``/``snapshot``/
    ``restore`` pass through: they are the engine's own sampling and lifecycle calls, never a fetch's.
    """

    __slots__ = ("_inner", "_guard")

    def __init__(self, inner: StateBackend, guard: _StateGuard) -> None:
        self._inner = inner
        self._guard = guard

    def _check(self) -> None:
        if not self._guard.open:
            raise StateAccessError(_GUARD_MSG)

    def get(self, scope: StateScope) -> object | None:
        self._check()
        return self._inner.get(scope)

    def put(self, scope: StateScope, value: object) -> None:
        self._check()
        self._inner.put(scope, value)

    def clear(self, scope: StateScope) -> None:
        self._check()
        self._inner.clear(scope)

    def entries(self, operator_id: str, name: str) -> Iterator[tuple[Any, Any, object]]:
        self._check()
        return self._inner.entries(operator_id, name)

    def sizes(self) -> dict[tuple[str, str], tuple[int, int]]:
        return self._inner.sizes()

    def snapshot(self) -> bytes:
        return self._inner.snapshot()

    def restore(self, blob: bytes) -> None:
        self._inner.restore(blob)


class _GuardedRecorder(Recorder):
    """The metrics counterpart to :class:`_GuardedStateBackend`: writing author telemetry while the guard
    is closed (from ``fetch``) raises. A counter/gauge *handle* cached in ``integrate`` is not itself
    guarded, but a stray fetch-time metric write is a telemetry inaccuracy, not the keyed-state corruption
    the state guard exists to stop."""

    __slots__ = ("_inner", "_guard")

    def __init__(self, inner: Recorder, guard: _StateGuard) -> None:
        self._inner = inner
        self._guard = guard

    def _check(self) -> None:
        if not self._guard.open:
            raise StateAccessError(_GUARD_MSG)

    def counter(self, name: str, **labels: object) -> Counter:
        self._check()
        return self._inner.counter(name, **labels)

    def gauge(self, name: str, **labels: object) -> Gauge:
        self._check()
        return self._inner.gauge(name, **labels)

    def histogram(self, name: str, **labels: object) -> Histogram:
        self._check()
        return self._inner.histogram(name, **labels)

    def event(self, name: str, **fields: object) -> None:
        self._check()
        self._inner.event(name, **fields)

    def snapshot(self) -> InstanceSnapshot:
        return (
            self._inner.snapshot()
        )  # a reporting call at teardown — never a fetch's, so not guarded


class OperatorContext:
    """What an operator is handed at ``open`` time: its ``operator_id``, this instance's ``subtask_index``
    of ``num_subtasks``, the ``state_backend``, the ``clock``, and the author ``metrics`` recorder.

    For an :class:`AsyncOneInputOperator` the engine installs an await-time guard
    (:meth:`_install_async_guard`) so ``state_backend`` / ``metrics`` — and any handle cached from them —
    raise :class:`StateAccessError` if touched from the concurrent ``fetch``. A synchronous operator never
    installs it and pays nothing: the accessors below reach the backend directly."""

    def __init__(
        self,
        operator_id: str,
        subtask_index: int = 0,
        num_subtasks: int = 1,
        state_backend: StateBackend | None = None,
        clock: Clock | None = None,
        metrics: Recorder | None = None,
    ) -> None:
        self.operator_id = operator_id
        self.subtask_index = subtask_index
        self.num_subtasks = num_subtasks
        self.clock = clock if clock is not None else SystemClock()
        self.state_backend = state_backend if state_backend is not None else _InMemoryStateBackend()
        #: Operator-author custom-metric recorder — a SEPARATE recorder from the actor's built-in one, so
        #: the single-writer invariant is never violated. Defaults to a zero-cost no-op. A custom metric
        #: must be declared in the catalog with ``owner=Owner.AUTHOR`` (every metric is catalog-declared,
        #: and this recorder may write only author-owned ones).
        self.metrics = metrics if metrics is not None else NULL_RECORDER
        self._guard: _StateGuard | None = None  # installed only for an async transform

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

    # --- the async-stage await-time guard (engine-only) -----------------------------------------

    def _install_async_guard(self) -> StateBackend:
        """Engine-only: install the await-time guard for an async transform and return the *raw* state
        backend, which the engine's own state-size sampling uses to bypass the guard. Wraps
        ``state_backend`` and ``metrics`` so a ``fetch`` — which runs while the guard is closed — that
        reaches keyed state or telemetry, directly or through a handle cached in ``integrate``, raises
        :class:`StateAccessError`. The driver opens the guard only inside :meth:`_state_section`."""
        raw = self.state_backend
        self._guard = _StateGuard()
        self.state_backend = _GuardedStateBackend(raw, self._guard)
        self.metrics = _GuardedRecorder(self.metrics, self._guard)
        return raw

    @contextmanager
    def _state_section(self) -> Iterator[None]:
        """Engine-only: open the guard for one synchronous ``open``/``integrate``/``on_watermark``/
        ``close`` call, restoring the prior state on exit so nested calls compose. A no-op when no guard is
        installed (every synchronous operator)."""
        guard = self._guard
        if guard is None:
            yield
            return
        prev = guard.open
        guard.open = True
        try:
            yield
        finally:
            guard.open = prev

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


class AsyncOneInputOperator(ABC):
    """A one-input transform whose I/O is awaited, split into an awaiting half and a synchronous half so
    the engine — not the operator — owns concurrency and ordering.

    Where :class:`OneInputOperator` transforms each batch in one synchronous ``process``, this splits the
    work in two:

    * :meth:`fetch` is the awaiting half — the external I/O (a range read, a DB or HTTP lookup). The
      engine runs up to :meth:`max_in_flight` of them as concurrent tasks, so their I/O overlaps. It is
      handed only the batch and returns an opaque per-batch result; it must not emit nor touch ``ctx``
      (it runs while sibling fetches are in flight, so a keyed-state read-modify-write spanning its
      ``await`` would lose updates — the :class:`OperatorContext` raises if it tries).
    * :meth:`integrate` is the synchronous half — run on the actor task, one batch at a time and in input
      order (:meth:`ordered`). It folds the fetched ``result`` into keyed state through ``ctx`` and emits
      through ``out``, exactly like :meth:`OneInputOperator.process`. It never ``await``s, so its
      keyed-state read-modify-write never spans a yield.

    Splitting this way lets a *keyed* async enrich keep many lookups in flight — strictly more than a
    stateless-only async map. ``DESIGN.md`` mechanism 9 is why awaiting here is safe;
    :func:`~nautilus.runtime.actor.run_async_transform` is how the engine drives it.
    """

    def open(self, ctx: OperatorContext) -> None:
        """Called once on the actor task before any record. Acquire the I/O client/pool here, not in
        ``__init__``: the executor builds a fresh operator per subtask and may cloudpickle the factory to
        a worker that never imported your module, so a live client must not ride along. Do NOT stash
        ``ctx`` to reach state from :meth:`fetch` — that is exactly what the guard forbids."""

    @abstractmethod
    async def fetch(self, batch: pa.RecordBatch) -> object:
        """Do this batch's external I/O and return an opaque per-batch result for :meth:`integrate`. Runs
        as one of up to :meth:`max_in_flight` concurrent tasks; may ``await``. Handed only the batch — it
        must not emit, must not touch nautilus keyed state, and must not write telemetry: ``ctx`` is not
        passed, and reaching keyed state or the recorder through ``self`` — even a handle cached in
        :meth:`integrate` — raises :class:`StateAccessError`. Raising fails the whole job; exceeding
        :meth:`timeout_micros` cancels it and fails the job."""

    @abstractmethod
    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        """Fold one batch's fetched ``result`` into keyed state via ``ctx`` and emit via ``out``.
        Synchronous; must not block/await (the engine runs it serially, per the class contract)."""

    def on_watermark(self, t: int, ctx: OperatorContext, out: Collector) -> None:
        """Event-time progress reached ``t`` (forwarded downstream after this returns). Fire due
        windows/timers; touch state via ``ctx`` and emit via ``out``. Default: no-op (stateless ops).
        """

    def key_columns(self) -> tuple[str, ...] | None:
        """The columns this operator's input must be co-partitioned on, or ``None`` if keyless — the same
        contract as :meth:`OneInputOperator.key_columns`. A keyed async enrich declares its key here so a
        parallel run routes each key to one instance and never splits its per-key state."""
        return None

    def max_in_flight(self) -> int:
        """How many :meth:`fetch` tasks may run at once (>= 1). This is the stage's backpressure bound:
        once it is reached the actor stops reading, so a slow external store stalls upstream with bounded
        memory."""
        return 8

    def ordered(self) -> bool:
        """Whether to integrate and emit strictly in input order. Ordered (the default) makes emission,
        the keyed-state fold order, and the structural digest reproducible while still overlapping I/O
        (later fetches run behind the in-order frontier). Only ordered is implemented today; unordered
        (completion-order, lower latency, stateless-only) is a planned addition."""
        return True

    def timeout_micros(self) -> int | None:
        """Per-:meth:`fetch` deadline in microseconds, or ``None`` for no timeout. A fetch that exceeds it
        is cancelled and the job fails fast (counted as ``async.timeouts``). Retry is the author's
        concern."""
        return None

    async def close(self) -> None:
        """Called once on the actor task after EOS has drained every in-flight fetch (or on teardown). May
        ``await`` — the awaited client acquired in :meth:`open` is released here (``await client.aclose()``),
        mirroring :meth:`AsyncSink.close`."""


class AsyncSink(ABC):
    """A terminal that writes each batch to an external store — the one operator besides a source that may
    ``await``. It is the graph's leaf: no output, so it emits and forwards nothing.

    Implement :meth:`write` (the awaited write); override :meth:`open`/:meth:`close` to acquire and
    release the client and :meth:`max_in_flight`/:meth:`timeout_micros`/:meth:`key_columns` to tune
    concurrency, deadline, and partitioning. Writes are **at-least-once** — a failed run re-runs the whole
    job — so a write must be idempotent under replay (deterministic keys / upsert). Why a sink may
    ``await`` where a transform may not, and how it replaces the synthesized collecting sink, is
    ``DESIGN.md`` mechanism 9; how the engine drives it is :func:`~nautilus.runtime.actor.run_async_sink`.
    """

    def open(self, ctx: OperatorContext) -> None:
        """Called once on the actor task before any write. Acquire the external client/connection here,
        not in ``__init__``: the executor builds a fresh sink per subtask and may cloudpickle the factory
        to a worker that never imported your module, so a live client must not ride along."""

    @abstractmethod
    async def write(self, batch: pa.RecordBatch) -> None:
        """Write one batch to the external store. Runs as one of up to :meth:`max_in_flight` concurrent
        tasks, so several writes overlap; may ``await``. Handed only the batch — it must not emit (a sink
        has no downstream) nor mutate nautilus keyed state. Raising fails the whole job (fail-fast).
        """

    def key_columns(self) -> tuple[str, ...] | None:
        """The columns this sink's input is co-partitioned on, or ``None`` if keyless. A keyed sink
        declares its key so a parallel run routes each key to one instance (e.g. for per-key upsert);
        keyless, a parallel run round-robins batches across instances — the write fan-out."""
        return None

    def max_in_flight(self) -> int:
        """How many :meth:`write` tasks may be in flight at once (>= 1). The bound is this sink's
        backpressure: once it is reached the actor stops reading, so a slow external store stalls upstream
        with bounded memory rather than buffering without limit."""
        return 8

    def timeout_micros(self) -> int | None:
        """Per-write deadline in microseconds, or ``None`` for no timeout. A write that exceeds it is
        cancelled and the job fails fast (counted as ``async.timeouts``). Retry is the author's concern.
        """
        return None

    async def close(self) -> None:
        """Called once on the actor task after every in-flight write has finished (at end of stream, or on
        teardown). Flush/commit any buffered writes and release the client. Under at-least-once a re-run
        repeats every write, so a commit here need not be transactional across the run."""


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
