import dataclasses

import pytest

from nautilus.core.records import (
    EOS,
    EOS_FRAME,
    MAX_LEGAL_EVENT_TIME,
    WATERMARK_MAX,
    WATERMARK_MIN,
    Barrier,
    Batch,
    StatusActive,
    StatusIdle,
    Watermark,
    check_event_time,
    is_control,
    is_data,
)
from nautilus.testing import batch


def test_data_frame_is_not_control():
    b = Batch(batch(x=[1, 2]))
    assert is_data(b)
    assert not is_control(b)
    assert b.is_control is False
    assert b.num_rows == 2


@pytest.mark.parametrize("frame", [Watermark(5), EOS(), StatusIdle(), StatusActive(), Barrier(1)])
def test_control_frames_are_control(frame):
    assert is_control(frame)
    assert frame.is_control is True
    assert not is_data(frame)


def test_zero_field_frames_compare_equal():
    assert EOS() == EOS_FRAME
    assert StatusIdle() == StatusIdle()


def test_event_time_bounds():
    check_event_time(0)
    check_event_time(WATERMARK_MIN)
    check_event_time(MAX_LEGAL_EVENT_TIME)
    with pytest.raises(ValueError):
        check_event_time(WATERMARK_MAX)
    with pytest.raises(ValueError):
        check_event_time(WATERMARK_MIN - 1)


def test_frames_are_immutable():
    w = Watermark(5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.t = 6  # frozen dataclass
