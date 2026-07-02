"""Wire framing for the cross-process transport.

Each message on a socket is ``[1-byte kind][4-byte big-endian length][payload]``:

- ``DATA``: a :class:`~nautilus.core.records.Batch`; its ``pa.RecordBatch`` is written as an Arrow IPC
  stream (canonical extension types such as ``fixed_shape_tensor`` survive the round trip).
- ``CONTROL``: a control frame (:class:`EOS`, :class:`Barrier`), msgpack-encoded as a small dict.
- ``CREDIT``: an integer credit count (msgpack), returned by the consumer to the producer.
"""

from __future__ import annotations

import asyncio
from enum import IntEnum

import msgpack
import pyarrow as pa

from nautilus.core.records import EOS, EOS_FRAME, Barrier, Batch, Frame

_HEADER = 5  # 1 kind byte + 4 length bytes
_MAX_FRAME_BYTES = (
    256 * 1024 * 1024
)  # reject an absurd length before allocating (garbage/corruption)


class Kind(IntEnum):
    DATA = 1
    CONTROL = 2
    CREDIT = 3


def encode_frame(frame: Frame) -> bytes:
    """Encode a data or control frame as a full wire message."""
    if isinstance(frame, Batch):
        return _pack(Kind.DATA, _batch_to_ipc(frame.data))
    return _pack(Kind.CONTROL, _control_to_bytes(frame))


def encode_credit(count: int) -> bytes:
    """Encode a credit-return message carrying ``count`` credits."""
    return _pack(Kind.CREDIT, msgpack.packb(count))


def decode(kind: Kind, payload: bytes) -> Frame | int:
    """Decode one message payload into a frame, or an int credit count for ``CREDIT``."""
    if kind == Kind.DATA:
        return Batch(_ipc_to_batch(payload))
    if kind == Kind.CONTROL:
        return _bytes_to_control(payload)
    if kind == Kind.CREDIT:
        return int(msgpack.unpackb(payload))
    raise ValueError(
        f"unknown frame kind {kind!r}"
    )  # never silently treat an unknown kind as credit


def split(message: bytes) -> tuple[Kind, bytes]:
    """Parse one complete in-memory wire message into ``(kind, payload)`` — the synchronous counterpart
    to :func:`read_message` for a message already fully in a buffer (e.g. the output of
    :func:`encode_frame` / :func:`encode_credit`), without a stream. ``payload`` then decodes via
    :func:`decode`."""
    kind = Kind(message[0])
    length = int.from_bytes(message[1:_HEADER], "big")
    return kind, message[_HEADER : _HEADER + length]


async def read_message(reader: asyncio.StreamReader) -> tuple[Kind, bytes]:
    """Read one ``[kind][length][payload]`` message from a stream."""
    header = await reader.readexactly(_HEADER)
    kind = Kind(header[0])
    length = int.from_bytes(header[1:], "big")
    if (
        length > _MAX_FRAME_BYTES
    ):  # bound the allocation, mirroring the handshake's _MAX_PAYLOAD guard
        raise ValueError(f"frame length {length} exceeds max {_MAX_FRAME_BYTES}")
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


def _pack(kind: Kind, payload: bytes) -> bytes:
    return bytes([int(kind)]) + len(payload).to_bytes(4, "big") + payload


def _batch_to_ipc(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return bytes(sink.getvalue().to_pybytes())


def _ipc_to_batch(payload: bytes) -> pa.RecordBatch:
    with pa.ipc.open_stream(pa.py_buffer(payload)) as reader:
        return reader.read_next_batch()


def _control_to_bytes(frame: Frame) -> bytes:
    if isinstance(frame, Barrier):
        return bytes(msgpack.packb({"k": "barrier", "id": frame.checkpoint_id}))
    if isinstance(frame, EOS):
        return bytes(msgpack.packb({"k": "eos"}))
    raise ValueError(f"cannot encode frame: {frame!r}")


def _bytes_to_control(payload: bytes) -> Frame:
    fields = msgpack.unpackb(payload, raw=False)
    tag = fields["k"]
    if tag == "barrier":
        return Barrier(int(fields["id"]))
    if tag == "eos":
        return EOS_FRAME
    raise ValueError(f"unknown control tag: {tag!r}")
