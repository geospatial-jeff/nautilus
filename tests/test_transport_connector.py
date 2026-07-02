"""Stage 2c: the connect/accept + handshake seam, proven in isolation over real loopback TCP.

No compiler, spawn, or coordinator — a test harness owns the listeners. The seam must (1) route each
socket to the edge its handshake names regardless of accept order, (2) carry frames identically to the
socketpair path, (3) not deadlock a capacity-1 edge, (4) reject a bad/unknown connection without dying,
and (5) leave no tasks or sockets behind. A mis-route would silently split a key downstream, so the
arrival-order fuzz test is the load-bearing one.
"""

from __future__ import annotations

import asyncio
import random
import socket
from contextlib import suppress

import pytest

from nautilus.core.records import EOS_FRAME, Barrier, Batch, Frame
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.runtime.connector import ChannelId
from nautilus.testing import data
from nautilus.transport.connector import SocketConnector
from nautilus.transport.handshake import encode_handshake, write_handshake
from nautilus.transport.listener import EdgeListener
from nautilus.transport.socket_channel import SocketChannel

_LOOPBACK = "127.0.0.1"


# --- harness -----------------------------------------------------------------------------------


async def _setup(
    cids: list[ChannelId], *, capacity: int = DEFAULT_CAPACITY
) -> tuple[SocketConnector, SocketConnector, EdgeListener, EdgeListener]:
    """One consumer worker (its listener expects ``cids``) and one producer worker dialing it."""
    consumer_listener = EdgeListener(_LOOPBACK, 0, cids)
    await consumer_listener.start()
    producer_listener = EdgeListener(_LOOPBACK, 0, [])  # nothing flows into the producer worker
    await producer_listener.start()
    consumer = SocketConnector(consumer_listener, lambda c: ("", 0), capacity=capacity)
    producer = SocketConnector(
        producer_listener, lambda c: consumer_listener.address, capacity=capacity
    )
    return producer, consumer, consumer_listener, producer_listener


async def _drain_and_close(
    producer: SocketConnector, consumer: SocketConnector, *listeners: EdgeListener
) -> None:
    # Symmetric teardown: the producer drains its outbound edges while the consumer closes its inbound
    # ones, so finish() sees the peer's FIN promptly instead of eating the full drain timeout.
    await asyncio.gather(producer.finish(), consumer.close())
    await producer.close()
    for listener in listeners:
        await listener.close()


async def _send_all(channel: SocketChannel, frames: list[Frame]) -> None:
    for frame in frames:
        await channel.send(frame)


async def _recv_n(channel: SocketChannel, n: int) -> list[Frame]:
    return [await channel.recv() for _ in range(n)]


def _frame_eq(a: Frame, b: Frame) -> bool:
    if isinstance(a, Batch) and isinstance(b, Batch):
        return a.data.equals(b.data)
    return a == b


# --- routing, ordering, equivalence ------------------------------------------------------------


async def test_connect_handshake_delivers_frames() -> None:
    cid = ChannelId("op0", 0, "op1", 0)
    producer, consumer, cl, pl = await _setup([cid])
    try:
        send = await producer.outbound(cid)
        recv = await consumer.inbound(cid)
        frames: list[Frame] = [data(x=[1, 2, 3]), Barrier(10), data(x=[4, 5]), EOS_FRAME]
        _, got = await asyncio.gather(_send_all(send, frames), _recv_n(recv, len(frames)))
        assert all(_frame_eq(a, b) for a, b in zip(frames, got, strict=True))
    finally:
        await _drain_and_close(producer, consumer, cl, pl)


async def test_frames_are_identical_to_the_socketpair_path() -> None:
    frames: list[Frame] = [
        data(x=[1, 2, 3]),
        Barrier(7),
        data(y=["a", "b"]),
        Barrier(8),
        Barrier(9),
        EOS_FRAME,
    ]

    # socketpair path (no handshake): the Stage 1 baseline.
    a, b = socket.socketpair(socket.AF_UNIX)
    ra, wa = await asyncio.open_connection(sock=a)
    rb, wb = await asyncio.open_connection(sock=b)
    sp_send, sp_recv = SocketChannel(ra, wa), SocketChannel(rb, wb)
    _, sp_got = await asyncio.gather(_send_all(sp_send, frames), _recv_n(sp_recv, len(frames)))
    await asyncio.gather(sp_send.finish(), sp_recv.close())
    await sp_send.close()

    # connect + handshake path.
    cid = ChannelId("op0", 0, "op1", 0)
    producer, consumer, cl, pl = await _setup([cid])
    try:
        send = await producer.outbound(cid)
        recv = await consumer.inbound(cid)
        _, hs_got = await asyncio.gather(_send_all(send, frames), _recv_n(recv, len(frames)))
    finally:
        await _drain_and_close(producer, consumer, cl, pl)

    # The handshake is read off before the SocketChannel, so both paths deliver the same frames.
    assert all(_frame_eq(x, y) for x, y in zip(frames, sp_got, strict=True))
    assert all(_frame_eq(x, y) for x, y in zip(frames, hs_got, strict=True))


async def test_producer_parked_until_consumer_accepts() -> None:
    # The producer connects and sends before the consumer calls inbound — the socket is parked and
    # resolves when the consumer asks for it.
    cid = ChannelId("op0", 0, "op1", 0)
    producer, consumer, cl, pl = await _setup([cid])
    try:
        send = await producer.outbound(cid)
        await send.send(data(x=[1]))
        await send.send(EOS_FRAME)
        recv = await consumer.inbound(cid)  # claims the already-connected (parked) socket
        got = await _recv_n(recv, 2)
        assert _frame_eq(got[0], data(x=[1])) and _frame_eq(got[1], EOS_FRAME)
    finally:
        await _drain_and_close(producer, consumer, cl, pl)


async def test_random_arrival_order_routes_without_crosswiring() -> None:
    # The load-bearing test: producers connect in a shuffled order; each consumer must receive the
    # frame tagged with its own upstream index. A crossed wire would surface as a wrong src.
    rng = random.Random(5)
    cids = [ChannelId(f"op{u}", u, "sink", 0) for u in range(6)]
    producer, consumer, cl, pl = await _setup(cids)
    try:
        order = list(range(len(cids)))
        rng.shuffle(order)
        sends = {}
        for u in order:
            channel = await producer.outbound(cids[u])
            await channel.send(data(src=[u]))
            await channel.send(EOS_FRAME)
            sends[u] = channel
        for u in range(len(cids)):
            recv = await consumer.inbound(cids[u])
            got = await _recv_n(recv, 2)
            assert got[0].data.column("src").to_pylist() == [u], u
    finally:
        await _drain_and_close(producer, consumer, cl, pl)


async def test_capacity_one_single_edge_does_not_deadlock() -> None:
    cid = ChannelId("op0", 0, "op1", 0)
    producer, consumer, cl, pl = await _setup([cid], capacity=1)
    try:
        send = await producer.outbound(cid)
        recv = await consumer.inbound(cid)
        n = 20
        frames: list[Frame] = [data(i=[k]) for k in range(n)] + [EOS_FRAME]
        _, got = await asyncio.wait_for(
            asyncio.gather(_send_all(send, frames), _recv_n(recv, len(frames))), timeout=10
        )
        assert [f.data.column("i").to_pylist()[0] for f in got[:-1]] == list(range(n))
        assert _frame_eq(got[-1], EOS_FRAME)
    finally:
        await _drain_and_close(producer, consumer, cl, pl)


# --- clean rejection: the listener survives a bad connection -----------------------------------

_GOOD = ChannelId("op0", 0, "op1", 0)


async def _dial_then_close(listener: EdgeListener, preamble: bytes) -> None:
    """Open a raw connection to ``listener``, send ``preamble``, and confirm the listener closes it
    (its read() returns EOF) — the clean-rejection signal — then close our end."""
    reader, writer = await asyncio.open_connection(*listener.address)
    writer.write(preamble)
    await writer.drain()
    assert await asyncio.wait_for(reader.read(), timeout=5) == b""
    writer.close()
    with suppress(OSError, ConnectionError):
        await writer.wait_closed()


async def _expected_edge_still_delivers(listener: EdgeListener) -> None:
    """Prove the listener still routes its expected edge after a bad connection was rejected."""
    producer_listener = EdgeListener(_LOOPBACK, 0, [])
    await producer_listener.start()
    producer = SocketConnector(producer_listener, lambda c: listener.address)
    consumer = SocketConnector(listener, lambda c: ("", 0))
    send = await producer.outbound(_GOOD)
    recv = await consumer.inbound(_GOOD)
    await send.send(EOS_FRAME)
    assert _frame_eq((await _recv_n(recv, 1))[0], EOS_FRAME)
    await _drain_and_close(producer, consumer, producer_listener)


async def test_unknown_channel_id_is_rejected_and_listener_survives() -> None:
    listener = EdgeListener(_LOOPBACK, 0, [_GOOD])
    await listener.start()
    try:
        # A well-formed handshake for an edge the listener does not expect is closed cleanly.
        await _dial_then_close(listener, encode_handshake(ChannelId("ghost", 0, "op1", 0)))
        assert listener.rejected == 1
        await _expected_edge_still_delivers(listener)
    finally:
        await listener.close()


async def test_malformed_handshake_is_rejected_and_listener_survives() -> None:
    listener = EdgeListener(_LOOPBACK, 0, [_GOOD])
    await listener.start()
    try:
        await _dial_then_close(listener, b"GARBAGE-not-a-nautilus-handshake")
        assert listener.rejected == 1
        await _expected_edge_still_delivers(listener)
    finally:
        await listener.close()


# --- listener contract -------------------------------------------------------------------------


async def test_accept_of_an_unexpected_edge_raises() -> None:
    async with EdgeListener(_LOOPBACK, 0, [_GOOD]) as listener:
        with pytest.raises(KeyError):
            await listener.accept(ChannelId("not", 0, "expected", 0))


async def test_address_before_start_raises() -> None:
    listener = EdgeListener(_LOOPBACK, 0, [])
    with pytest.raises(RuntimeError):
        _ = listener.address


async def test_duplicate_connection_for_an_edge_is_rejected() -> None:
    # An edge connects once. A second connection naming an already-taken edge is rejected, and the
    # first edge keeps working.
    listener = EdgeListener(_LOOPBACK, 0, [_GOOD])
    await listener.start()
    producer_listener = EdgeListener(_LOOPBACK, 0, [])
    await producer_listener.start()
    producer = SocketConnector(producer_listener, lambda c: listener.address)
    consumer = SocketConnector(listener, lambda c: ("", 0))
    try:
        send = await producer.outbound(_GOOD)
        recv = await consumer.inbound(_GOOD)  # claims the edge
        await _dial_then_close(listener, encode_handshake(_GOOD))  # second connect → rejected
        assert listener.rejected == 1
        await send.send(EOS_FRAME)
        assert _frame_eq((await _recv_n(recv, 1))[0], EOS_FRAME)  # first edge unaffected
        await _drain_and_close(producer, consumer, producer_listener)
    finally:
        await listener.close()


# --- teardown ----------------------------------------------------------------------------------


async def test_teardown_leaks_no_tasks() -> None:
    baseline = len(asyncio.all_tasks())
    cid = ChannelId("op0", 0, "op1", 0)
    producer, consumer, cl, pl = await _setup([cid])
    send = await producer.outbound(cid)
    recv = await consumer.inbound(cid)
    await send.send(EOS_FRAME)
    await _recv_n(recv, 1)
    await _drain_and_close(producer, consumer, cl, pl)
    await asyncio.sleep(0)  # let cancelled read-loops settle
    assert len(asyncio.all_tasks()) <= baseline


async def test_close_returns_with_a_parked_unclaimed_connection() -> None:
    # A producer connected (valid handshake) but the consumer never accepted: close() must close the
    # parked socket before awaiting the server, so it returns promptly instead of hanging on wait_closed.
    listener = EdgeListener(_LOOPBACK, 0, [_GOOD])
    await listener.start()
    reader, writer = await asyncio.open_connection(*listener.address)
    await write_handshake(writer, _GOOD)
    await asyncio.sleep(0)  # let _on_accept park the slot
    await asyncio.wait_for(
        listener.close(), timeout=2
    )  # < default close_timeout, so via real close
    writer.close()
    with suppress(OSError, ConnectionError):
        await writer.wait_closed()


async def test_stalled_handshake_is_timed_out_and_rejected() -> None:
    # A peer that connects and sends nothing is timed out and rejected, never parking accept forever.
    listener = EdgeListener(_LOOPBACK, 0, [_GOOD], handshake_timeout=0.3)
    await listener.start()
    try:
        reader, writer = await asyncio.open_connection(*listener.address)  # connect, send nothing
        assert await asyncio.wait_for(reader.read(), timeout=5) == b""  # timed out → closed
        assert listener.rejected == 1
        writer.close()
        with suppress(OSError, ConnectionError):
            await writer.wait_closed()
    finally:
        await asyncio.wait_for(listener.close(), timeout=2)


async def test_close_cancels_an_in_flight_handshake() -> None:
    # A handshake still being read when close() runs is cancelled (its socket released), so close()
    # neither hangs nor leaks the accept task.
    listener = EdgeListener(_LOOPBACK, 0, [_GOOD], handshake_timeout=30)
    await listener.start()
    reader, writer = await asyncio.open_connection(
        *listener.address
    )  # connect, leave handshake pending
    for _ in range(
        200
    ):  # wait until the server's accept task is in flight, then cancel it via close()
        if listener._accept_tasks:
            break
        await asyncio.sleep(0.01)
    assert listener._accept_tasks, "the accept handler never started"
    await asyncio.wait_for(listener.close(), timeout=2)
    writer.close()
    with suppress(OSError, ConnectionError):
        await writer.wait_closed()


async def test_close_is_bounded_with_an_open_claimed_channel() -> None:
    # If a caller closes the listener before the consumer's SocketChannel (the wrong order), close()
    # falls back to its bounded wait instead of hanging forever.
    listener = EdgeListener(_LOOPBACK, 0, [_GOOD], close_timeout=0.4)
    await listener.start()
    producer_listener = EdgeListener(_LOOPBACK, 0, [])
    await producer_listener.start()
    producer = SocketConnector(producer_listener, lambda c: listener.address)
    consumer = SocketConnector(listener, lambda c: ("", 0))
    await producer.outbound(_GOOD)
    await consumer.inbound(_GOOD)  # claimed; its SocketChannel is left open
    await asyncio.wait_for(listener.close(), timeout=3)  # bounded by close_timeout, does not hang
    await asyncio.gather(producer.finish(), consumer.close())
    await producer.close()
    await producer_listener.close()


async def test_bidirectional_teardown_completes_promptly() -> None:
    # Two workers, each producing one edge to the other and consuming one from the other — the layout
    # where sequential finish-then-close circular-waits and eats the full per-channel drain timeout.
    # SocketConnector.finish() drains outbound and closes inbound in one gather, so a gather over both
    # workers' finish() completes well under the 5s drain timeout.
    a_to_b = ChannelId("a", 0, "b", 0)
    b_to_a = ChannelId("b", 0, "a", 0)
    a_listener = EdgeListener(_LOOPBACK, 0, [b_to_a])
    b_listener = EdgeListener(_LOOPBACK, 0, [a_to_b])
    await a_listener.start()
    await b_listener.start()
    a = SocketConnector(a_listener, lambda c: b_listener.address)
    b = SocketConnector(b_listener, lambda c: a_listener.address)
    try:
        a_out = await a.outbound(
            a_to_b
        )  # dial-all-outbound before accept-all-inbound (no deadlock)
        b_out = await b.outbound(b_to_a)
        a_in = await a.inbound(b_to_a)
        b_in = await b.inbound(a_to_b)
        await a_out.send(EOS_FRAME)
        await b_out.send(EOS_FRAME)
        await a_in.recv()
        await b_in.recv()
        # If finish() were outbound-only, each worker's drain would wait on a peer that never closed its
        # inbound, eating the 5s timeout; the symmetric finish() resolves it under the wait_for guard.
        await asyncio.wait_for(asyncio.gather(a.finish(), b.finish()), timeout=4)
        await asyncio.gather(a.close(), b.close())
    finally:
        await asyncio.gather(a_listener.close(), b_listener.close())
