"""Edge-case robustness of the credit transport: peer death, child errors, no silent hangs."""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from nautilus.core.records import EOS, EOS_FRAME, Batch
from nautilus.operators import InMemorySource, KeyedCount
from nautilus.testing import batch, data
from nautilus.transport import run_two_process
from nautilus.transport.socket_channel import SocketChannel, TransportClosed


async def _pair(window: int) -> tuple[SocketChannel, SocketChannel]:
    a, b = socket.socketpair(socket.AF_UNIX)
    ra, wa = await asyncio.open_connection(sock=a)
    rb, wb = await asyncio.open_connection(sock=b)
    return SocketChannel(ra, wa, capacity=window), SocketChannel(rb, wb, capacity=window)


async def test_recv_raises_on_peer_disconnect_without_eos() -> None:
    send_ch, recv_ch = await _pair(4)
    await send_ch.send(Batch(batch(i=[0])))
    await send_ch.close()  # producer goes away WITHOUT sending EOS

    first = await asyncio.wait_for(recv_ch.recv(), timeout=2.0)  # the buffered frame still arrives
    assert isinstance(first, Batch)
    with pytest.raises(TransportClosed):  # must raise, not hang
        await asyncio.wait_for(recv_ch.recv(), timeout=2.0)
    await recv_ch.close()


async def test_send_raises_when_consumer_disappears() -> None:
    send_ch, recv_ch = await _pair(1)
    await send_ch.send(Batch(batch(i=[0])))  # spend the one credit
    await recv_ch.close()  # consumer goes away; no more credit will ever come back

    with pytest.raises((TransportClosed, ConnectionError, OSError)):  # must raise, not hang forever
        await asyncio.wait_for(send_ch.send(Batch(batch(i=[1]))), timeout=2.0)
    await send_ch.close()


async def test_clean_eos_close_is_not_an_error() -> None:
    send_ch, recv_ch = await _pair(4)
    await send_ch.send(Batch(batch(i=[0])))
    await send_ch.send(EOS_FRAME)
    await send_ch.close()

    assert isinstance(await asyncio.wait_for(recv_ch.recv(), timeout=2.0), Batch)
    assert isinstance(await asyncio.wait_for(recv_ch.recv(), timeout=2.0), EOS)
    await recv_ch.close()


def test_child_operator_error_propagates_without_hanging() -> None:
    # The child's KeyedCount asks for a "word" column the source never produces, so the child's
    # pipeline raises. The parent must surface that quickly, not wait out a timeout.
    source = InMemorySource([data(wrong=["a", "b"]), EOS_FRAME])
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="worker failed"):
        run_two_process(source, [KeyedCount("word")], capacity=4)
    assert time.monotonic() - start < 20.0
