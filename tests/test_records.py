import dataclasses

import pytest

from nautilus.core.records import (
    EOS,
    EOS_FRAME,
    Barrier,
    Batch,
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


@pytest.mark.parametrize("frame", [EOS(), Barrier(1)])
def test_control_frames_are_control(frame):
    assert is_control(frame)
    assert frame.is_control is True
    assert not is_data(frame)


def test_zero_field_frames_compare_equal():
    assert EOS() == EOS_FRAME


def test_frames_are_immutable():
    b = Barrier(5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.checkpoint_id = 6  # frozen dataclass
