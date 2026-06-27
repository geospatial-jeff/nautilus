"""``SocketConnector``: the cross-worker implementation of the runtime
:class:`~nautilus.runtime.connector.Connector`.

The executor wires actors against the :class:`~nautilus.runtime.connector.Connector` interface without
knowing the transport. The in-process connector returns one queue for both ends of an edge; this one
returns the two ends of a TCP connection — and that is the only difference the executor ever sees, so
the same plan slice runs single-process or across the network with identical wiring code.

Each end's role mirrors the connect/accept split:

* :meth:`outbound` (the producer) dials the node hosting the edge's destination, writes the handshake,
  and wraps the connection as the send end;
* :meth:`inbound` (the consumer) takes the socket its :class:`~nautilus.transport.listener.EdgeListener`
  routed for that edge and wraps it as the recv end.

It is given a plain ``resolve`` callable mapping a :class:`ChannelId` to the destination address, not a
cluster address book — keeping ``transport`` free of any ``cluster`` import (the control plane supplies a
resolver in Stage 2d). Teardown drains then closes only the channels it created; the listener it accepts
from is a bootstrap resource its owner binds and closes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from nautilus.runtime.channel import DEFAULT_CAPACITY, Channel
from nautilus.runtime.connector import ChannelId, Connector
from nautilus.transport.handshake import write_handshake
from nautilus.transport.listener import EdgeListener
from nautilus.transport.socket_channel import SocketChannel


class SocketConnector(Connector):
    """A worker's cross-worker connector: producers dial, consumers accept via the worker's listener."""

    def __init__(
        self,
        listener: EdgeListener,
        resolve: Callable[[ChannelId], tuple[str, int]],
        *,
        capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        self._listener = listener
        self._resolve = resolve
        self._capacity = capacity
        self._outbound: list[SocketChannel] = []
        self._inbound: list[SocketChannel] = []

    async def outbound(self, channel_id: ChannelId) -> Channel:
        host, port = self._resolve(channel_id)
        reader, writer = await asyncio.open_connection(host, port)
        try:
            # Announce the edge before the SocketChannel starts its read loop, so the bytes after the
            # preamble are exactly the frame stream a socketpair channel would carry.
            await write_handshake(writer, channel_id)
        except Exception:
            writer.close()  # no SocketChannel wraps it yet, so close the raw socket ourselves
            raise
        channel = SocketChannel(reader, writer, capacity=self._capacity)
        self._outbound.append(channel)
        return channel

    async def inbound(self, channel_id: ChannelId) -> Channel:
        reader, writer = await self._listener.accept(channel_id)
        channel = SocketChannel(reader, writer, capacity=self._capacity)
        self._inbound.append(channel)
        return channel

    async def finish(self) -> None:
        """Graceful symmetric teardown: drain every outbound edge (half-close, await the consumer's
        drain) and close every inbound edge (send FIN to our producers), all in one gather. Closing
        inbound concurrently with draining outbound is what lets a bidirectional mesh tear down without
        each worker's drain waiting on a peer that is itself still draining."""
        await asyncio.gather(
            *(channel.finish() for channel in self._outbound),
            *(channel.close() for channel in self._inbound),
        )

    async def close(self) -> None:
        """Close every channel this connector created (inbound and outbound), abortively and
        idempotently. The listener is closed by its owner."""
        await asyncio.gather(*(channel.close() for channel in self._inbound + self._outbound))
