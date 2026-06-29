"""Telemetry value types — the instruments an actor writes and the snapshot it produces.

These are pure, lock-free, single-writer primitives (the owning actor loop is the only writer). They
hold plain ints/lists and allocate nothing per record. A :class:`Histogram` uses fixed, explicit
bucket boundaries so ``observe`` is an allocation-free ``bisect`` + increment and the result is exact
and deterministic (a future diff tool compares by boundary *value*, never array index).

The instruments live on the data path. A point-in-time :class:`InstanceSnapshot` (frozen, picklable —
so it survives crossing a process boundary in Stage 2) is read at the job boundary and handed to the
report layer.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

#: A label set, normalized to a sorted tuple of ``(name, value)`` string pairs so it is hashable,
#: deterministic, and picklable. Build with :func:`make_labels`.
Labels = tuple[tuple[str, str], ...]


def make_labels(values: dict[str, object]) -> Labels:
    """Normalize a label mapping into a sorted, stringified :data:`Labels` tuple."""
    return tuple(sorted((k, str(v)) for k, v in values.items()))


# --- Instruments (mutable, single-writer) ------------------------------------------------------


class Counter:
    """A monotonically increasing integer."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0

    def add(self, n: int = 1) -> None:
        self.value += n


class Gauge:
    """A level that can move up and down. Tracks last/min/max so a rollup can reduce per spec."""

    __slots__ = ("last", "min", "max", "_seen")

    def __init__(self) -> None:
        self.last = 0.0
        self.min = 0.0
        self.max = 0.0
        self._seen = False

    def set(self, v: float) -> None:
        self.last = v
        if not self._seen:
            self.min = self.max = v
            self._seen = True
        else:
            if v < self.min:
                self.min = v
            if v > self.max:
                self.max = v


@dataclass(frozen=True, slots=True)
class HistogramData:
    """An immutable read of a histogram: per-bucket counts plus the boundaries they correspond to."""

    boundaries: tuple[int, ...]
    buckets: tuple[int, ...]  # length == len(boundaries) + 1 (last is the overflow bucket)
    count: int
    sum: int
    min: int | None
    max: int | None


class Histogram:
    """Fixed-bucket distribution. ``boundaries`` are upper-inclusive edges; ``observe`` is O(log n)."""

    __slots__ = ("boundaries", "buckets", "count", "sum", "min", "max")

    def __init__(self, boundaries: tuple[int, ...]) -> None:
        self.boundaries = boundaries
        self.buckets = [0] * (len(boundaries) + 1)
        self.count = 0
        self.sum = 0
        self.min: int | None = None
        self.max: int | None = None

    def observe(self, v: int) -> None:
        # boundaries are upper-inclusive ("le"): a value equal to a boundary lands in that bucket.
        self.buckets[bisect_left(self.boundaries, v)] += 1
        self.count += 1
        self.sum += v
        if self.min is None or v < self.min:
            self.min = v
        if self.max is None or v > self.max:
            self.max = v

    def data(self) -> HistogramData:
        return HistogramData(
            self.boundaries, tuple(self.buckets), self.count, self.sum, self.min, self.max
        )


# --- Events & snapshot --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventRecord:
    """A discrete, structured event. ``fields`` is a sorted tuple (closed per its EventSpec)."""

    seq: int
    at_micros: int
    operator_id: str
    name: str
    fields: tuple[tuple[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return dict(self.fields)


@dataclass(frozen=True, slots=True)
class InstanceSnapshot:
    """A frozen, picklable point-in-time read of one instance's instruments."""

    operator_id: str
    op_class: str
    kind: str
    subtask_index: int
    node: str
    counters: dict[tuple[str, Labels], int] = field(default_factory=dict)
    gauges: dict[tuple[str, Labels], tuple[float, float, float]] = field(default_factory=dict)
    histograms: dict[tuple[str, Labels], HistogramData] = field(default_factory=dict)
    events: tuple[EventRecord, ...] = ()
    events_dropped: int = 0
