"""Windowing primitives: window definitions and assigners.

A :class:`WindowAssigner` maps a record's event time to the window(s) it belongs to and owns the
boundary math, so an operator never recomputes it. Triggering (when to compute a window) is, for
Stage 0, implicitly *on watermark*: a window fires once the operator watermark passes its end.
Sliding/session windows and pluggable triggers arrive in Stage 3; the assigner abstraction is shaped to
accommodate them (``assign`` already returns a *list*).
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
        start = timestamp - (timestamp % self.size)
        return [TimeWindow(start, start + self.size)]

    def assign_starts(self, timestamps: np.ndarray) -> np.ndarray:
        """The window start for each event time, vectorized — the columnar form the keyed window
        operator uses, so the tumbling-boundary formula lives only here. Floor division matches the
        scalar ``ts - ts % size`` in :meth:`assign` for negative event times too (both floor)."""
        return (timestamps // self.size) * self.size
