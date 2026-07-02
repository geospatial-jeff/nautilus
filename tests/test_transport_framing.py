"""Wire framing round-trips for the cross-process transport."""

from __future__ import annotations

import numpy as np

from nautilus.core.records import EOS_FRAME, Barrier, Batch
from nautilus.testing import batch
from nautilus.transport.framing import Kind, decode, encode_credit, encode_frame, split


def _roundtrip(frame: object) -> object:
    kind, payload = split(encode_frame(frame))  # type: ignore[arg-type]
    return decode(kind, payload)


def test_control_frames_roundtrip() -> None:
    for frame in [EOS_FRAME, Barrier(7)]:
        assert _roundtrip(frame) == frame


def test_data_batch_roundtrip() -> None:
    rb = batch(word=["a", "b", "c"], n=[1, 2, 3])
    out = _roundtrip(Batch(rb))
    assert isinstance(out, Batch)
    assert out.data.equals(rb)


def test_tensor_batch_roundtrip() -> None:
    imgs = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    rb = batch(tile_id=[0, 1], image=imgs)
    out = _roundtrip(Batch(rb))
    assert isinstance(out, Batch)
    assert out.data.equals(rb)
    assert out.data.schema.field("image").type == rb.schema.field("image").type


def test_credit_roundtrip() -> None:
    kind, payload = split(encode_credit(5))
    assert kind == Kind.CREDIT
    assert decode(kind, payload) == 5


def test_kind_tags() -> None:
    assert split(encode_frame(Barrier(1)))[0] == Kind.CONTROL
    assert split(encode_frame(Batch(batch(x=[1]))))[0] == Kind.DATA
    assert split(encode_credit(1))[0] == Kind.CREDIT
