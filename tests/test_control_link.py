"""Stage 4.2: the control-link wire round-trips every message and rejects garbage.

The framed wire is the remote replacement for the control ``mp.Queue``s, so it must carry every control
message faithfully — including a ``Done`` with its sink result and a ``Launch`` with the cloudpickled plan
— and reassemble messages across arbitrary TCP segment boundaries (a readable event may hold a partial
frame or several). A bad magic or an oversized declared length is rejected *before* allocating, so a
foreign or truncated stream fails fast rather than sizing a huge buffer.
"""

from __future__ import annotations

import asyncio

import pytest

from nautilus.cluster.control_link import (
    Abort,
    ControlLinkError,
    Launch,
    encode,
    read_message,
    take_message,
)
from nautilus.cluster.protocol import Done, Failed, Register
from nautilus.telemetry import TelemetryConfig, Tier


def test_take_message_round_trips_simple_messages() -> None:
    for message in [
        Abort(),
        Register(1, "worker-1", 9000),
        Failed(2, "a traceback"),
        Done(0, [], b"sink-ipc-bytes"),
    ]:
        assert take_message(bytearray(encode(message))) == message


def test_take_message_reassembles_several_frames_and_a_partial_tail() -> None:
    first, second = encode(Register(0, "h", 1)), encode(Register(1, "h", 2))
    buffer = bytearray(first + second[:3])  # one whole frame plus a partial second
    assert take_message(buffer) == Register(0, "h", 1)
    assert take_message(buffer) is None  # the second frame has not fully arrived
    buffer += second[3:]
    assert take_message(buffer) == Register(1, "h", 2)
    assert not buffer  # fully consumed


def test_launch_round_trips_its_fields() -> None:
    launch = Launch(0, b"plan-bytes", {("op0", 0): 0}, 16, TelemetryConfig(tier=Tier.COUNTERS))
    got = take_message(bytearray(encode(launch)))
    assert (got.worker_id, got.plan_bytes, got.capacity) == (0, b"plan-bytes", 16)
    assert got.placement == {("op0", 0): 0}
    assert got.config.tier == Tier.COUNTERS


def test_bad_magic_is_rejected() -> None:
    with pytest.raises(ControlLinkError):
        take_message(bytearray(b"XXXX" + (0).to_bytes(4, "big")))


def test_oversized_length_is_rejected_before_allocating() -> None:
    framed_garbage = b"NCTL" + (2**31).to_bytes(4, "big")  # declared length beyond the max
    with pytest.raises(ControlLinkError):
        take_message(bytearray(framed_garbage))


async def test_read_message_async_round_trip_then_eof() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(encode(Register(3, "h", 7)))
    reader.feed_eof()
    assert await read_message(reader) == Register(3, "h", 7)
    with pytest.raises(asyncio.IncompleteReadError):  # EOF = the peer closed
        await read_message(reader)


async def test_read_message_async_rejects_bad_magic() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"XXXX" + (0).to_bytes(4, "big"))
    reader.feed_eof()
    with pytest.raises(ControlLinkError):
        await read_message(reader)
