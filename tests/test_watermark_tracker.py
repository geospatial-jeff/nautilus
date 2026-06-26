import pytest

from nautilus.core.records import WATERMARK_MAX, WATERMARK_MIN
from nautilus.core.time import (
    BoundedOutOfOrder,
    ColumnTimestampAssigner,
    MonotonicTimestamps,
    TestClock,
    WatermarkTracker,
)
from nautilus.testing import batch


def test_combined_is_min_over_inputs():
    t = WatermarkTracker(2)
    assert t.update(0, 10) is None  # input 1 unseen -> combined stays at MIN
    assert t.combined == WATERMARK_MIN
    assert t.update(1, 5) == 5
    assert t.combined == 5


def test_per_channel_regression_raises():
    t = WatermarkTracker(1)
    t.update(0, 10)
    with pytest.raises(ValueError):
        t.update(0, 9)


def test_idle_input_is_excluded_and_no_regression_on_rejoin():
    t = WatermarkTracker(2)
    t.update(0, 10)
    assert t.set_idle(1) == 10  # now combined follows input 0 alone
    # input 1 rejoins behind: combined must not regress
    assert t.update(1, 3) is None
    assert t.combined == 10


def test_close_input_pins_to_max():
    t = WatermarkTracker(1)
    assert t.close_input(0) == WATERMARK_MAX


def test_watermark_strategies():
    assert MonotonicTimestamps().watermark_for(100) == 100
    assert BoundedOutOfOrder(5).watermark_for(100) == 95


def test_column_timestamp_assigner_max():
    assigner = ColumnTimestampAssigner("ts")
    assert assigner.max_timestamp(batch(ts=[3, 9, 5])) == 9
    assert assigner.max_timestamp(batch(ts=[])) is None


def test_test_clock_monotonic():
    c = TestClock(0)
    assert c.advance(5) == 5
    assert c.now_micros() == 5
    with pytest.raises(ValueError):
        c.advance(-1)
