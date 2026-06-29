import pyarrow as pa
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


def test_set_active_rejoins_input_without_regression():
    # C110: an idle input rejoining via set_active (the StatusActive seam) must not move the combined
    # watermark backward, even though it rejoins behind the current combined.
    t = WatermarkTracker(2)
    t.update(0, 10)
    t.update(1, 20)
    assert t.combined == 10
    assert t.set_idle(0) == 20  # input 0 idle -> combined advances to follow input 1 alone
    assert t.set_active(0) is None  # input 0 rejoins at 10 (behind 20) -> no advance, no regression
    assert t.combined == 20


def test_watermark_strategies():
    assert MonotonicTimestamps().watermark_for(100) == 100
    assert BoundedOutOfOrder(5).watermark_for(100) == 95


def test_column_timestamp_assigner_max():
    assigner = ColumnTimestampAssigner("ts")
    assert assigner.max_timestamp(batch(ts=[3, 9, 5])) == 9
    assert assigner.max_timestamp(batch(ts=[])) is None


@pytest.mark.parametrize(
    ("unit", "value", "expected_micros"),
    [
        ("s", 5, 5_000_000),  # 5 s  -> 5_000_000 µs
        ("ms", 1500, 1_500_000),  # 1500 ms -> 1_500_000 µs
        ("us", 1500, 1500),  # already µs
        ("ns", 1500, 1),  # 1500 ns -> 1 µs (truncates)
    ],
)
def test_column_timestamp_assigner_normalizes_timestamp_unit(unit, value, expected_micros):
    # Arrow timestamp columns of any unit must be read as microseconds, not as raw underlying ints.
    col = pa.array([value], pa.timestamp(unit))
    rb = pa.record_batch([col], names=["ts"])
    assigner = ColumnTimestampAssigner("ts")
    assert assigner.timestamps(rb).to_pylist() == [expected_micros]
    assert assigner.max_timestamp(rb) == expected_micros


def test_test_clock_monotonic():
    c = TestClock(0)
    assert c.advance(5) == 5
    assert c.now_micros() == 5
    with pytest.raises(ValueError):
        c.advance(-1)
