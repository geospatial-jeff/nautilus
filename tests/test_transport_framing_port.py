"""Edge-behavior pins for the data-plane wire framing (:mod:`nautilus.transport.framing`).

The happy-path round-trips live in ``test_transport_framing.py``; this file pins the boundary and
failure behavior a faithful rewrite must reproduce: rejecting an unknown frame kind rather than
mis-decoding it as credit, bounding the declared payload length before allocating, integer barrier
ids at their extremes, and IPC preservation of an empty batch and of null bitmaps in a nullable
column.
"""

from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

from nautilus.core.records import EOS_FRAME, Barrier, Batch
from nautilus.transport.framing import (
    _MAX_FRAME_BYTES,
    Kind,
    decode,
    encode_credit,
    encode_frame,
    read_message,
    split,
)


class _FakeReader:
    """A minimal :class:`asyncio.StreamReader` stand-in that serves bytes from an in-memory buffer.

    Counts ``readexactly`` calls so a test can prove the length guard fires *before* the payload read
    (one call = header only), instead of after allocating the payload (two calls).
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self.read_calls = 0

    async def readexactly(self, n: int) -> bytes:
        self.read_calls += 1
        chunk = self._data[self._pos : self._pos + n]
        if len(chunk) < n:
            raise asyncio.IncompleteReadError(chunk, n)
        self._pos += n
        return chunk


def _roundtrip(frame: object) -> object:
    kind, payload = split(encode_frame(frame))  # type: ignore[arg-type]
    return decode(kind, payload)


# (a) unknown frame kind is rejected, never mis-decoded as credit ------------------------------


def test_decode_unknown_kind_raises() -> None:
    # 99 is not a valid Kind member; decode compares kind == Kind.DATA/CONTROL/CREDIT, so an
    # unknown value falls through to the guard rather than being treated as a credit count.
    with pytest.raises(ValueError, match="unknown frame kind 99"):
        decode(99, b"")  # type: ignore[arg-type]


def test_decode_unknown_kind_is_not_credit() -> None:
    # A msgpack-int payload would decode fine as CREDIT; with an unknown kind it must still raise,
    # proving the unknown kind is never silently reinterpreted as credit.
    import msgpack

    with pytest.raises(ValueError, match="unknown frame kind"):
        decode(99, msgpack.packb(5))  # type: ignore[arg-type]


# (b) read_message length guard bounds allocation before reading the payload -------------------
#
# This is the data-plane read_message in nautilus.transport.framing, guarded by _MAX_FRAME_BYTES
# (256 MiB) — distinct from the control-plane read_message in nautilus.cluster.control_link, which
# has its own much smaller _MAX_PAYLOAD guard.


def test_max_frame_bytes_is_256_mib() -> None:
    assert _MAX_FRAME_BYTES == 256 * 1024 * 1024


def test_read_message_rejects_oversized_length_before_payload() -> None:
    length = _MAX_FRAME_BYTES + 1
    header = bytes([int(Kind.DATA)]) + length.to_bytes(4, "big")
    reader = _FakeReader(header)  # header only: no payload bytes are available

    with pytest.raises(ValueError, match=r"frame length 268435457 exceeds max 268435456"):
        asyncio.run(read_message(reader))  # type: ignore[arg-type]

    # Exactly one readexactly (the 5-byte header): the guard fired before any payload read.
    assert reader.read_calls == 1


def test_read_message_allows_boundary_length() -> None:
    # A declared length of exactly _MAX_FRAME_BYTES passes the guard and proceeds to read the
    # payload; with no payload bytes available that read fails with IncompleteReadError, proving
    # the guard did not reject the boundary value.
    header = bytes([int(Kind.DATA)]) + _MAX_FRAME_BYTES.to_bytes(4, "big")
    reader = _FakeReader(header)

    with pytest.raises(asyncio.IncompleteReadError):
        asyncio.run(read_message(reader))  # type: ignore[arg-type]

    # Two readexactly calls: header, then the payload attempt — i.e. the guard was passed.
    assert reader.read_calls == 2


# (c) Barrier checkpoint_id round-trips at integer boundaries ----------------------------------


@pytest.mark.parametrize("checkpoint_id", [0, 2**31, 2**53, 2**63 - 1])
def test_barrier_checkpoint_id_roundtrips_at_boundaries(checkpoint_id: int) -> None:
    out = _roundtrip(Barrier(checkpoint_id))
    assert isinstance(out, Barrier)
    assert out.checkpoint_id == checkpoint_id
    assert out == Barrier(checkpoint_id)


# (d) IPC preserves an empty batch and null bitmaps in a nullable column -----------------------


def test_zero_row_batch_roundtrips_with_schema() -> None:
    schema = pa.schema([pa.field("n", pa.int64(), nullable=True)])
    rb = pa.record_batch([pa.array([], type=pa.int64())], schema=schema)

    out = _roundtrip(Batch(rb))

    assert isinstance(out, Batch)
    assert out.data.num_rows == 0
    assert out.data.schema.equals(schema)  # types + field metadata survive
    assert out.data.equals(rb)


def test_nullable_int64_null_bitmap_roundtrips() -> None:
    schema = pa.schema([pa.field("n", pa.int64(), nullable=True)])
    rb = pa.record_batch([pa.array([1, None, 3, None], type=pa.int64())], schema=schema)

    out = _roundtrip(Batch(rb))

    assert isinstance(out, Batch)
    assert out.data.equals(rb)
    assert out.data.schema.field("n").nullable is True
    assert out.data.column("n").null_count == 2
    assert out.data.column("n").to_pylist() == [1, None, 3, None]


# (e) DATA/CONTROL/CREDIT round-trip and the kind byte is exactly 1/2/3 ------------------------


def test_kind_bytes_are_exactly_1_2_3() -> None:
    # The wire kind byte is the first byte of a full message; pin the exact numeric encoding.
    assert encode_frame(Batch(pa.record_batch({"x": pa.array([1])})))[0] == 1
    assert encode_frame(Barrier(1))[0] == 2
    assert encode_frame(EOS_FRAME)[0] == 2  # EOS is also a CONTROL frame
    assert encode_credit(1)[0] == 3
    assert (int(Kind.DATA), int(Kind.CONTROL), int(Kind.CREDIT)) == (1, 2, 3)


def test_data_frame_roundtrips() -> None:
    rb = pa.record_batch({"x": pa.array([1, 2, 3], type=pa.int64())})
    kind, payload = split(encode_frame(Batch(rb)))
    assert kind == Kind.DATA
    out = decode(kind, payload)
    assert isinstance(out, Batch)
    assert out.data.equals(rb)


def test_control_frame_roundtrips() -> None:
    kind, payload = split(encode_frame(Barrier(42)))
    assert kind == Kind.CONTROL
    assert decode(kind, payload) == Barrier(42)


def test_credit_frame_roundtrips() -> None:
    kind, payload = split(encode_credit(17))
    assert kind == Kind.CREDIT
    assert decode(kind, payload) == 17
