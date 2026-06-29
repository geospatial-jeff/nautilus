"""Windowing primitives: window definitions and assigners.

A :class:`WindowAssigner` maps a record's event time to the window(s) it belongs to and owns the
boundary math, so an operator never recomputes it. Triggering (when to compute a window) is implicitly
*on watermark*: a window fires once the operator watermark passes its end. Only tumbling windows
(:class:`TumblingEventTimeWindows`) exist today; the assigner abstraction is shaped to admit sliding and
session windows and pluggable triggers later (``assign`` already returns a *list*).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A half-open event-time interval ``[start, end)`` in microseconds."""

    start: int
    end: int


class WindowAssigner(ABC):
    @abstractmethod
    def assign(self, timestamp: int) -> list[TimeWindow]:
        """Return every window the given event time belongs to."""


class TumblingEventTimeWindows(WindowAssigner):
    """Fixed-size, non-overlapping windows aligned to the epoch."""

    def __init__(self, size_micros: int) -> None:
        if size_micros <= 0:
            raise ValueError("window size must be positive")
        self.size = size_micros

    def assign(self, timestamp: int) -> list[TimeWindow]:
        start = (timestamp // self.size) * self.size  # same floor formula as assign_starts, scalar
        return [TimeWindow(start, start + self.size)]

    def assign_starts(self, timestamps: np.ndarray) -> np.ndarray:
        """The window start for each event time — the vectorized form of :meth:`assign`'s
        ``(ts // size) * size`` that the keyed window operator uses on a whole column. Floor division is
        correct for negative event times too (Python/NumPy ``//`` floors)."""
        return (timestamps // self.size) * self.size
