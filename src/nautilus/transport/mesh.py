"""A :class:`~nautilus.runtime.parallel.ChannelFactory` whose channels are real socket pairs.

Swapping :class:`~nautilus.runtime.parallel.InProcFactory` for :class:`SocketPairFactory` runs the
*identical* parallel mesh over the cross-process :class:`~nautilus.transport.socket_channel.SocketChannel`
— the same credit/framing path a node-to-node TCP edge uses — instead of in-process queues. That is the
"same graph whether in-process or a TCP SocketChannel" guarantee, exercised in one process.

``socket.socketpair`` returns two already-connected sockets, so there is no ``start_server`` accept
queue, no connect/accept ordering, and no handshake to drive — only the same ``SocketChannel`` code.
Teardown plain-closes both ends *after* the run's :class:`~asyncio.TaskGroup` has joined: at that point
every frame has been delivered, so there is no unread data to drop and no reason to call
:meth:`~nautilus.transport.socket_channel.SocketChannel.finish` (which would block on its drain timeout
per pair).
"""

from __future__ import annotations

import asyncio
import socket

from nautilus.runtime.channel import Channel
from nautilus.runtime.parallel import ChannelFactory
from nautilus.transport.socket_channel import SocketChannel


class SocketPairFactory(ChannelFactory):
    """Builds each mesh channel as the two ends of a connected UNIX ``socketpair``."""

    def __init__(self) -> None:
        self._created: list[SocketChannel] = []

    async def pair(self, capacity: int) -> tuple[Channel, Channel]:
        a, b = socket.socketpair(socket.AF_UNIX)
        ra, wa = await asyncio.open_connection(sock=a)
        rb, wb = await asyncio.open_connection(sock=b)
        send_end = SocketChannel(ra, wa, capacity=capacity)
        recv_end = SocketChannel(rb, wb, capacity=capacity)
        self._created += [send_end, recv_end]
        return send_end, recv_end

    async def close_all(self) -> None:
        # close() cancels and awaits each end's read-loop task, so the per-pair readers leave nothing
        # running. (Why close() and not finish() is in the module docstring.)
        for ch in self._created:
            await ch.close()
