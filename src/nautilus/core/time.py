"""Event time, processing-time clocks, watermark generation and watermark combination.

Event time is integer microseconds (see :mod:`nautilus.core.records`). Three concerns live here:

* :class:`Clock` — processing time, injectable so tests are deterministic (:class:`TestClock`).
* :class:`TimestampAssigner` / :class:`WatermarkStrategy` — how a *source* reads event times from
  data and turns the maximum observed time into a watermark.
* :class:`WatermarkTracker` — how a *downstream operator* combines the watermarks arriving on its
  several input channels into a single operator watermark (the minimum over non-idle inputs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pyarrow as pa
import pyarrow.compute as pc

from nautilus.core.records import WATERMARK_MAX, WATERMARK_MIN, check_event_time

# --- Processing-time clock ---------------------------------------------------------------------


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


# --- Event-time extraction & watermark generation (source side) --------------------------------


class TimestampAssigner(ABC):
    """Extracts per-row event times (int64 microseconds) from a batch."""

    @abstractmethod
    def timestamps(self, batch: pa.RecordBatch) -> pa.Array:
        """Return an int64 Arrow array of event times (microseconds), one per row."""

    def max_timestamp(self, batch: pa.RecordBatch) -> int | None:
        if batch.num_rows == 0:
            return None
        m = pc.max(self.timestamps(batch)).as_py()
        return None if m is None else check_event_time(int(m))


class ColumnTimestampAssigner(TimestampAssigner):
    """Reads event time from a named column (an int64 micros column, or an Arrow timestamp)."""

    def __init__(self, column: str) -> None:
        self.column = column

    def timestamps(self, batch: pa.RecordBatch) -> pa.Array:
        col = batch.column(self.column)
        if pa.types.is_timestamp(col.type):
            col = pc.cast(col, pa.int64())  # timestamp units are already micros-or-less ints
        return pc.cast(col, pa.int64())


class WatermarkStrategy(ABC):
    """Turns the maximum event time a source has observed into an emittable watermark."""

    @abstractmethod
    def watermark_for(self, max_event_time_seen: int) -> int: ...


class MonotonicTimestamps(WatermarkStrategy):
    """For perfectly ordered sources: the watermark is the max event time seen."""

    def watermark_for(self, max_event_time_seen: int) -> int:
        return max_event_time_seen


class BoundedOutOfOrder(WatermarkStrategy):
    """Tolerates out-of-order data up to ``delay`` microseconds: watermark = max_seen - delay."""

    def __init__(self, delay_micros: int) -> None:
        if delay_micros < 0:
            raise ValueError("delay must be non-negative")
        self.delay = delay_micros

    def watermark_for(self, max_event_time_seen: int) -> int:
        return max_event_time_seen - self.delay


# --- Watermark combination (downstream side) ---------------------------------------------------


class WatermarkTracker:
    """Combines the watermarks of several input channels into one operator watermark.

    The combined watermark is the **minimum over non-idle inputs**, and only ever moves forward.
    Idle inputs are excluded so a silent partition cannot stop event-time progress; an idle input
    that becomes active again never causes the combined watermark to move backward.
    """

    def __init__(self, num_inputs: int) -> None:
        if num_inputs < 1:
            raise ValueError("need at least one input")
        self._wms = [WATERMARK_MIN] * num_inputs
        self._idle = [False] * num_inputs
        self._combined = WATERMARK_MIN

    @property
    def combined(self) -> int:
        return self._combined

    def update(self, input_index: int, t: int) -> int | None:
        """Record watermark ``t`` on ``input_index`` (which becomes active). Returns the new
        combined watermark if it advanced, else ``None``. Raises on a per-channel regression."""
        if t < self._wms[input_index]:
            raise ValueError(
                f"watermark regression on input {input_index}: {t} < {self._wms[input_index]}"
            )
        self._wms[input_index] = t
        self._idle[input_index] = False
        return self._recompute()

    def set_idle(self, input_index: int) -> int | None:
        self._idle[input_index] = True
        return self._recompute()

    def set_active(self, input_index: int) -> int | None:
        self._idle[input_index] = False
        return self._recompute()

    def close_input(self, input_index: int) -> int | None:
        """Mark an input as ended (EOS): pin its watermark at the maximum and make it active."""
        return self.update(input_index, WATERMARK_MAX)

    def _recompute(self) -> int | None:
        active = [self._wms[i] for i in range(len(self._wms)) if not self._idle[i]]
        candidate = min(active) if active else self._combined
        if candidate > self._combined:
            self._combined = candidate
            return self._combined
        return None
