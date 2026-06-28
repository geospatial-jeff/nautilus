"""The per-actor recorder — the single, lock-free writer of one instance's telemetry.

Each actor loop owns one :class:`InstanceRecorder` and is its sole writer, so increments are a plain
``int +=`` with no lock (the GIL and the synchronous per-record step make that safe). A metric whose
``min_tier`` exceeds the configured tier resolves to a shared no-op instrument, so disabled telemetry
costs nothing.

Resolving an instrument is not free: ``counter``/``gauge``/``histogram`` re-run a catalog lookup and a
label-tuple allocation on every call (they return the cached instrument, but only after that work). So
per-record callers hoist the instrument once outside the loop — ``rows_out = recorder.counter(...)``,
then ``rows_out.add(...)`` per row. The ``incr``/``observe``/``set_gauge`` verbs resolve and act in one
call; use them only off the hot path.

Operator-author metrics get a *separate* recorder via ``ctx.metrics``, so operator code never shares
the actor's recorder and the single-writer invariant holds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from nautilus.core.time import Clock, SystemClock
from nautilus.telemetry.catalog import MetricSpec, Owner, Tier, event_spec, metric_spec
from nautilus.telemetry.model import (
    Counter,
    EventRecord,
    Gauge,
    Histogram,
    InstanceSnapshot,
    Labels,
    make_labels,
)


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """Run-level telemetry settings, read once at the job boundary and threaded to recorders."""

    tier: Tier = Tier.COUNTERS
    clock: Clock = field(default_factory=SystemClock)
    #: Reserved, not yet wired: would populate OperatorNode.source_file/source_line on lifecycle events.
    capture_source_lines: bool = False
    event_log_capacity: int = 1024
    run_id: str | None = None
    #: Reserved, not yet wired: intended threshold for a future backpressure-stall event.
    stall_threshold_micros: int = 5000
    validate: bool = True
    #: Sample process CPU/memory + event-loop lag periodically (only when tier > OFF). Excluded from
    #: the config digest because it affects only non-deterministic, non-structural metrics.
    sample_system: bool = True
    sample_interval_micros: int = 500_000


DEFAULT_CONFIG = TelemetryConfig()


# --- No-op instruments (shared singletons for disabled metrics) --------------------------------


class _NoOpCounter(Counter):
    def add(self, n: int = 1) -> None:
        pass


class _NoOpGauge(Gauge):
    def set(self, v: float) -> None:
        pass

    def add(self, v: float) -> None:
        pass


class _NoOpHistogram(Histogram):
    def __init__(self) -> None:
        super().__init__(())

    def observe(self, v: int) -> None:
        pass


NOOP_COUNTER = _NoOpCounter()
NOOP_GAUGE = _NoOpGauge()
NOOP_HISTOGRAM = _NoOpHistogram()


# --- Recorder interface ------------------------------------------------------------------------


class Recorder(ABC):
    """What instrumentation calls. Two implementations: :class:`InstanceRecorder`, :class:`NullRecorder`."""

    @abstractmethod
    def counter(self, name: str, **labels: object) -> Counter: ...

    @abstractmethod
    def gauge(self, name: str, **labels: object) -> Gauge: ...

    @abstractmethod
    def histogram(self, name: str, **labels: object) -> Histogram: ...

    @abstractmethod
    def event(self, name: str, **fields: object) -> None: ...

    @abstractmethod
    def snapshot(self) -> InstanceSnapshot: ...

    # Convenience verbs (resolve-and-act in one call).
    def incr(self, name: str, n: int = 1, **labels: object) -> None:
        self.counter(name, **labels).add(n)

    def observe(self, name: str, v: int, **labels: object) -> None:
        self.histogram(name, **labels).observe(v)

    def set_gauge(self, name: str, v: float, **labels: object) -> None:
        self.gauge(name, **labels).set(v)


class InstanceRecorder(Recorder):
    """The real per-instance recorder."""

    __slots__ = (
        "operator_id",
        "op_class",
        "kind",
        "subtask_index",
        "node",
        "_owner",
        "_config",
        "_counters",
        "_gauges",
        "_histograms",
        "_events",
        "_seq",
        "_dropped",
    )

    def __init__(
        self,
        *,
        operator_id: str,
        op_class: str,
        kind: str,
        subtask_index: int = 0,
        node: str = "local",
        config: TelemetryConfig = DEFAULT_CONFIG,
        owner: Owner = Owner.ENGINE,
    ) -> None:
        self.operator_id = operator_id
        self.op_class = op_class
        self.kind = kind
        self.subtask_index = subtask_index
        self.node = node
        self._owner = owner
        self._config = config
        self._counters: dict[tuple[str, Labels], Counter] = {}
        self._gauges: dict[tuple[str, Labels], Gauge] = {}
        self._histograms: dict[tuple[str, Labels], Histogram] = {}
        self._events: list[EventRecord] = []
        self._seq = 0
        self._dropped = 0

    def _spec(self, name: str) -> MetricSpec:
        spec = metric_spec(name)
        if spec.owner != self._owner:
            # An ENGINE recorder may not write AUTHOR metrics and vice-versa — they merge by name in the
            # report, so a crossed write would corrupt the engine's totals (see catalog.Owner).
            raise KeyError(
                f"a {self._owner.value!r} recorder may not write metric {name!r} "
                f"(owned by {spec.owner.value!r})"
            )
        return spec

    def counter(self, name: str, **labels: object) -> Counter:
        spec = self._spec(name)
        if self._config.tier < spec.min_tier:
            return NOOP_COUNTER
        key = (name, make_labels(labels))
        inst = self._counters.get(key)
        if inst is None:
            inst = Counter()
            self._counters[key] = inst
        return inst

    def gauge(self, name: str, **labels: object) -> Gauge:
        spec = self._spec(name)
        if self._config.tier < spec.min_tier:
            return NOOP_GAUGE
        key = (name, make_labels(labels))
        inst = self._gauges.get(key)
        if inst is None:
            inst = Gauge()
            self._gauges[key] = inst
        return inst

    def histogram(self, name: str, **labels: object) -> Histogram:
        spec = self._spec(name)
        if self._config.tier < spec.min_tier:
            return NOOP_HISTOGRAM
        key = (name, make_labels(labels))
        inst = self._histograms.get(key)
        if inst is None:
            inst = Histogram(spec.boundaries)
            self._histograms[key] = inst
        return inst

    def event(self, name: str, **fields: object) -> None:
        spec = event_spec(name)
        if self._config.tier < spec.min_tier:
            return
        if self._config.validate:
            unknown = set(fields) - set(spec.fields)
            if unknown:
                raise KeyError(f"event {name!r} got fields not in its EventSpec: {sorted(unknown)}")
        self._events.append(
            EventRecord(
                self._seq,
                self._config.clock.now_micros(),
                self.operator_id,
                name,
                tuple(sorted(fields.items())),
            )
        )
        self._seq += 1
        if len(self._events) > self._config.event_log_capacity:
            self._events.pop(0)
            self._dropped += 1

    def snapshot(self) -> InstanceSnapshot:
        return InstanceSnapshot(
            operator_id=self.operator_id,
            op_class=self.op_class,
            kind=self.kind,
            subtask_index=self.subtask_index,
            node=self.node,
            counters={k: c.value for k, c in self._counters.items()},
            gauges={k: (g.last, g.min, g.max) for k, g in self._gauges.items()},
            histograms={k: h.data() for k, h in self._histograms.items()},
            events=tuple(self._events),
            events_dropped=self._dropped,
        )


class NullRecorder(Recorder):
    """A no-op recorder: every verb does nothing and skips the catalog lookup. Used for the OFF tier and
    as the default ``ctx.metrics`` for operators that emit no custom metrics."""

    def counter(self, name: str, **labels: object) -> Counter:
        return NOOP_COUNTER

    def gauge(self, name: str, **labels: object) -> Gauge:
        return NOOP_GAUGE

    def histogram(self, name: str, **labels: object) -> Histogram:
        return NOOP_HISTOGRAM

    def event(self, name: str, **fields: object) -> None:
        pass

    def snapshot(self) -> InstanceSnapshot:
        return InstanceSnapshot("", "", "", 0, "local")


NULL_RECORDER = NullRecorder()


def make_recorder(
    *,
    operator_id: str,
    op_class: str,
    kind: str,
    subtask_index: int = 0,
    node: str = "local",
    config: TelemetryConfig = DEFAULT_CONFIG,
    owner: Owner = Owner.ENGINE,
) -> Recorder:
    """Return a real recorder, or :data:`NULL_RECORDER` when telemetry is OFF (zero catalog lookups).
    ``owner`` is the metric-ownership role the recorder may write (``ENGINE`` for the actor's built-in
    recorder, ``AUTHOR`` for the per-operator ``ctx.metrics`` recorder)."""
    if config.tier <= Tier.OFF:
        return NULL_RECORDER
    return InstanceRecorder(
        operator_id=operator_id,
        op_class=op_class,
        kind=kind,
        subtask_index=subtask_index,
        node=node,
        config=config,
        owner=owner,
    )
