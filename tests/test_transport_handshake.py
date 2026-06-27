"""Stage 2c: the edge handshake codec round-trips a ChannelId and rejects anything else cleanly.

The handshake is the only thing that tells an accepting listener which edge a socket is, so the codec
must round-trip every field exactly, must leave the post-preamble bytes untouched for the SocketChannel
that follows, and must fail (never hang) on a foreign or truncated connection.
"""

from __future__ import annotations

import asyncio

import msgpack
import pytest

from nautilus.runtime.connector import ChannelId
from nautilus.transport.handshake import (
    _MAGIC,
    HandshakeError,
    encode_handshake,
    read_handshake,
)


def _reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def test_round_trips_every_field() -> None:
    for cid in [
        ChannelId("source", 0, "op0", 0),
        ChannelId("op1", 3, "sink", 0),
        ChannelId("op_a", 7, "op_b", 5),
    ]:
        assert await read_handshake(_reader(encode_handshake(cid))) == cid


async def test_post_handshake_bytes_are_left_for_the_channel() -> None:
    # The SocketChannel built next reads frames off the same stream, so the bytes after the preamble
    # must remain — this is the read-preamble-before-SocketChannel invariant.
    cid = ChannelId("source", 0, "op0", 0)
    reader = _reader(encode_handshake(cid) + b"FRAME-BYTES")
    assert await read_handshake(reader) == cid
    assert await reader.read() == b"FRAME-BYTES"


async def test_bad_magic_rejected() -> None:
    with pytest.raises(HandshakeError):
        await read_handshake(_reader(b"XXXX" + (0).to_bytes(4, "big")))


async def test_oversized_length_rejected_without_reading_payload() -> None:
    # A huge length is garbage; reject on the prefix alone rather than trying to allocate/read it.
    with pytest.raises(HandshakeError):
        await read_handshake(_reader(_MAGIC + (10**6).to_bytes(4, "big")))


async def test_garbage_payload_rejected() -> None:
    payload: bytes = msgpack.packb({"not": "a channel id"}, use_bin_type=True)
    data = _MAGIC + len(payload).to_bytes(4, "big") + payload
    with pytest.raises(HandshakeError):
        await read_handshake(_reader(data))


async def test_short_preamble_raises_incomplete() -> None:
    # A peer that connects and disconnects mid-preamble must surface as a read error, never a hang.
    with pytest.raises(asyncio.IncompleteReadError):
        await read_handshake(_reader(b"NA"))
    full = encode_handshake(ChannelId("source", 0, "op0", 0))
    with pytest.raises(asyncio.IncompleteReadError):
        await read_handshake(_reader(full[:-2]))  # length promises more payload than arrives
