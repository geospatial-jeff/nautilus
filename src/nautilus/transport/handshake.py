"""The edge handshake: a self-identifying preamble a producer writes before any frame.

When a producer dials a worker's :class:`~nautilus.transport.listener.EdgeListener`, the accepting side
has no other way to know *which* edge this socket is — TCP gives it only an anonymous connection. So the
producer writes one preamble naming the :class:`~nautilus.runtime.connector.ChannelId`, and the listener
reads it to route the socket to the right consumer. The preamble is the only thing that says where a
connection belongs, so accept order never matters.

The preamble must be read off the raw :class:`asyncio.StreamReader` *before* a
:class:`~nautilus.transport.socket_channel.SocketChannel` is built on that reader: the channel starts a
background read loop that would otherwise consume the preamble as if it were a frame. After the preamble
the stream is byte-identical to a socketpair channel, so the same credit/framing code runs over a dialed
TCP edge unchanged.

The wire format is ``[4-byte magic][4-byte big-endian length][msgpack payload]`` — distinct from the
frame format in :mod:`nautilus.transport.framing` (whose messages start with a 1-byte kind), so a
non-nautilus connection or a truncated preamble is rejected immediately instead of being misread as a
frame.
"""

from __future__ import annotations

import asyncio

import msgpack

from nautilus.runtime.connector import ChannelId

_MAGIC = b"NAUT"  # marks a nautilus edge preamble; a mismatch is a foreign/garbage connection
_PREFIX = 8  # 4 magic + 4 length bytes
_MAX_PAYLOAD = 4096  # a ChannelId is tiny; a larger length is garbage, not a real handshake


class HandshakeError(RuntimeError):
    """The preamble was absent, truncated, or not a valid nautilus edge handshake."""


def encode_handshake(channel_id: ChannelId) -> bytes:
    """Serialize a :class:`ChannelId` as a complete preamble."""
    payload: bytes = msgpack.packb(
        [
            channel_id.src_operator_id,
            channel_id.src_subtask,
            channel_id.dst_operator_id,
            channel_id.dst_subtask,
        ],
        use_bin_type=True,
    )
    return _MAGIC + len(payload).to_bytes(4, "big") + payload


def decode_handshake(prefix: bytes, payload: bytes) -> ChannelId:
    """Parse a preamble's ``prefix`` (magic + length) and ``payload`` into a :class:`ChannelId`."""
    if prefix[:4] != _MAGIC:
        raise HandshakeError(f"bad handshake magic {prefix[:4]!r}")
    try:
        fields = msgpack.unpackb(payload, raw=False)
        src_op, src_sub, dst_op, dst_sub = fields
        return ChannelId(str(src_op), int(src_sub), str(dst_op), int(dst_sub))
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise HandshakeError(f"malformed handshake payload: {exc}") from exc


async def write_handshake(writer: asyncio.StreamWriter, channel_id: ChannelId) -> None:
    """Producer side: write and flush the preamble, before the first frame goes on the wire."""
    writer.write(encode_handshake(channel_id))
    await writer.drain()


async def read_handshake(reader: asyncio.StreamReader) -> ChannelId:
    """Consumer side: read one preamble off the raw stream. Raises :class:`HandshakeError` (bad magic or
    oversized length) or :class:`asyncio.IncompleteReadError` (a peer that sends a short preamble then
    disconnects). A peer that connects and then *stalls* is not bounded here — the
    :class:`~nautilus.transport.listener.EdgeListener` wraps this read in its handshake timeout."""
    prefix = await reader.readexactly(_PREFIX)
    length = int.from_bytes(prefix[4:_PREFIX], "big")
    if prefix[:4] != _MAGIC or length > _MAX_PAYLOAD:
        raise HandshakeError(f"not a nautilus handshake (magic {prefix[:4]!r}, length {length})")
    payload = await reader.readexactly(length)
    return decode_handshake(prefix, payload)
