"""Credit-based flow control over a real UDS connection (both ends in one process)."""

from __future__ import annotations

import asyncio
import random
import socket

import pytest

from nautilus.core.records import EOS, EOS_FRAME, Barrier, Batch
from nautilus.runtime.mailbox import Mailbox
from nautilus.testing import batch
from nautilus.transport.socket_channel import SocketChannel


async def _connected_pair(window: int) -> tuple[SocketChannel, SocketChannel]:
    """A producer/consumer SocketChannel pair over a connected UDS socket pair."""
    a, b = socket.socketpair(socket.AF_UNIX)
    ra, wa = await asyncio.open_connection(sock=a)
    rb, wb = await asyncio.open_connection(sock=b)
    return SocketChannel(ra, wa, capacity=window), SocketChannel(rb, wb, capacity=window)


async def test_credit_conservation_fast_producer_slow_consumer() -> None:
    window, total = 4, 50
    send_ch, recv_ch = await _connected_pair(window)
    sent = received = max_in_flight = 0

    async def producer() -> None:
        nonlocal sent
        for i in range(total):
            await send_ch.send(Batch(batch(i=[i])))
            sent += 1
        await send_ch.send(EOS_FRAME)

    async def consumer() -> None:
        nonlocal received, max_in_flight
        rng = random.Random(0)
        while True:
            frame = await recv_ch.recv()
            if isinstance(frame, EOS):
                break
            received += 1
            max_in_flight = max(max_in_flight, sent - received)
            await asyncio.sleep(rng.random() * 0.002)  # a deliberately slow consumer

    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer())
        tg.create_task(consumer())

    assert received == total
    assert max_in_flight <= window  # the window was never exceeded
    await send_ch.close()
    await recv_ch.close()


async def test_control_not_blocked_by_saturated_data() -> None:
    window = 2
    send_ch, recv_ch = await _connected_pair(window)
    for i in range(window):  # fill the data window
        await send_ch.send(Batch(batch(i=[i])))

    # a further DATA send blocks: no credit until the consumer reads
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(send_ch.send(Batch(batch(i=[99]))), timeout=0.2)

    # a CONTROL send is not gated by credit and completes immediately
    await asyncio.wait_for(send_ch.send(Barrier(99)), timeout=0.5)

    frames = [await asyncio.wait_for(recv_ch.recv(), timeout=0.5) for _ in range(window + 1)]
    assert isinstance(frames[-1], Barrier) and frames[-1].checkpoint_id == 99
    await send_ch.close()
    await recv_ch.close()


async def test_diamond_two_inputs_no_deadlock() -> None:
    window = 2
    s1, r1 = await _connected_pair(window)
    s2, r2 = await _connected_pair(window)
    mailbox = Mailbox([r1, r2])
    received = 0

    async def producer(ch: SocketChannel, n: int) -> None:
        for i in range(n):
            await ch.send(Batch(batch(i=[i])))
        await ch.send(EOS_FRAME)

    async def consumer() -> None:
        nonlocal received
        while not mailbox.exhausted:
            idx, frame = await mailbox.get()
            if isinstance(frame, EOS):
                mailbox.close_input(idx)
            else:
                received += 1

    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(s1, 10))
        tg.create_task(producer(s2, 10))
        tg.create_task(consumer())

    assert received == 20
    for ch in (s1, r1, s2, r2):
        await ch.close()


async def test_stray_credit_rejected() -> None:
    window = 2
    send_ch, recv_ch = await _connected_pair(window)
    with pytest.raises(RuntimeError):
        await send_ch._grant_credits(window + 1)  # would push credits past the window
    await send_ch.close()
    await recv_ch.close()
