"""Processing-time clocks: the source of wall-clock time, injectable so tests are deterministic.

Processing time is an integer number of microseconds since the Unix epoch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Clock(ABC):
    """Source of processing time, in integer microseconds since the epoch."""

    @abstractmethod
    def now_micros(self) -> int: ...


class SystemClock(Clock):
    def now_micros(self) -> int:
        import time

        return time.time_ns() // 1000


class TestClock(Clock):
    """A manually advanced clock for deterministic tests. Never goes backwards."""

    __test__ = False  # not a pytest test class

    def __init__(self, start: int = 0) -> None:
        self._now = start

    def now_micros(self) -> int:
        return self._now

    def advance(self, delta_micros: int) -> int:
        if delta_micros < 0:
            raise ValueError("TestClock cannot go backwards")
        self._now += delta_micros
        return self._now

    def set(self, micros: int) -> int:
        if micros < self._now:
            raise ValueError("TestClock cannot go backwards")
        self._now = micros
        return self._now
