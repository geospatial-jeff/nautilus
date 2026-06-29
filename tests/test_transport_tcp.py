"""Graceful shutdown over a real TCP loopback connection: close must not drop unread data."""

from __future__ import annotations

import asyncio

from nautilus.core.records import EOS, EOS_FRAME, Batch
from nautilus.testing import batch
from nautilus.transport.socket_channel import SocketChannel


async def _tcp_pair(
    window: int,
) -> tuple[asyncio.AbstractServer, SocketChannel, SocketChannel]:
    """A producer/consumer ``SocketChannel`` pair over a real TCP connection on the loopback."""
    accepted: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = asyncio.Queue()

    async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await accepted.put((reader, writer))

    server = await asyncio.start_server(on_connect, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    rc, wc = await asyncio.open_connection("127.0.0.1", port)
    rs, ws = await accepted.get()
    return server, SocketChannel(rc, wc, capacity=window), SocketChannel(rs, ws, capacity=window)


async def test_finish_drains_before_close_no_data_loss() -> None:
    # A slow consumer leaves the last data frames and EOS still in flight when the producer finishes.
    # finish() drains to the consumer's end before close(), so an RST never discards them.
    server, producer, consumer = await _tcp_pair(window=4)
    total = 200
    received: list[Batch] = []

    async def produce() -> None:
        for i in range(total):
            await producer.send(Batch(batch(i=[i])))
        await producer.send(EOS_FRAME)
        await producer.finish()  # without this, close() can RST away the in-flight tail
        await producer.close()

    async def consume() -> None:
        while True:
            frame = await consumer.recv()
            if isinstance(frame, EOS):
                break
            received.append(frame)
            await asyncio.sleep(0.0005)  # slow enough that frames are still in flight at finish
        await consumer.close()  # consumer closes right after EOS, releasing the producer's drain

    async with asyncio.TaskGroup() as tg:
        tg.create_task(produce())
        tg.create_task(consume())

    assert len(received) == total  # every data frame survived the shutdown
    server.close()
    await server.wait_closed()
