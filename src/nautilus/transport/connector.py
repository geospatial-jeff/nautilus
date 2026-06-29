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
import socket
from collections.abc import Callable

from nautilus.runtime.channel import DEFAULT_CAPACITY, Channel
from nautilus.runtime.connector import ChannelId, Connector
from nautilus.transport.handshake import write_handshake
from nautilus.transport.listener import EdgeListener
from nautilus.transport.socket_channel import SocketChannel

# Bound a data dial. asyncio.open_connection has no timeout, so a misadvertised peer would block the OS
# TCP connect for minutes while crash detection never fires; this is the cap. Generous — it covers
# cross-host RTT, SYN-retransmit backoff, and a listen backlog draining when many producers dial one
# listener during a wide shuffle — but the caller keeps it below the bootstrap timeout.
DEFAULT_CONNECT_TIMEOUT = 30.0
_DIAL_ATTEMPTS = 3  # retry only a transient getaddrinfo miss (a peer container not yet in DNS)
_DIAL_BACKOFF = 0.5  # seconds between dial attempts, scaled by the attempt number

# A silent partition (a dead host, not a clean close) sends no FIN, so a SocketChannel's read would block
# forever and the unbounded completion wait would never return. Keepalive bounds detection to ~25s: idle
# 10s, then 3 probes 5s apart, after which a blocked recv raises a ConnectionError.
_KEEPALIVE_IDLE = 10
_KEEPALIVE_INTVL = 5
_KEEPALIVE_CNT = 3


def _enable_keepalive(writer: asyncio.StreamWriter) -> None:
    """Turn on TCP keepalive for a cross-host edge so a silent partition becomes a bounded
    ``ConnectionError`` instead of an indefinite block on ``recv``. The ``TCP_*`` tunables are
    Linux-specific and skipped where the platform lacks them; ``SO_KEEPALIVE`` alone still applies the OS
    default. Best-effort: a rejected sockopt never fails an edge."""
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (
            ("TCP_KEEPIDLE", _KEEPALIVE_IDLE),
            ("TCP_KEEPINTVL", _KEEPALIVE_INTVL),
            ("TCP_KEEPCNT", _KEEPALIVE_CNT),
        ):
            if hasattr(socket, name):
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
    except OSError:
        pass


class SocketConnector(Connector):
    """A worker's cross-worker connector: producers dial, consumers accept via the worker's listener."""

    def __init__(
        self,
        listener: EdgeListener,
        resolve: Callable[[ChannelId], tuple[str, int]],
        *,
        capacity: int = DEFAULT_CAPACITY,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ) -> None:
        self._listener = listener
        self._resolve = resolve
        self._capacity = capacity
        self._connect_timeout = connect_timeout
        self._outbound: list[SocketChannel] = []
        self._inbound: list[SocketChannel] = []

    async def _dial(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a TCP connection, bounded by ``connect_timeout``. Retries *only* a transient
        ``getaddrinfo`` miss (a peer whose DNS is not yet warm); a refused or timed-out connect is not
        retried — the two-phase bootstrap guarantees the destination listener is bound before any producer
        dials, so those are real failures, not startup races."""
        last: Exception | None = None
        for attempt in range(_DIAL_ATTEMPTS):
            try:
                return await asyncio.wait_for(
                    asyncio.open_connection(host, port), self._connect_timeout
                )
            except socket.gaierror as exc:
                last = exc
                await asyncio.sleep(_DIAL_BACKOFF * (attempt + 1))
        assert last is not None  # only reached after the loop exhausts on gaierror
        raise last

    async def outbound(self, channel_id: ChannelId) -> Channel:
        host, port = self._resolve(channel_id)
        reader, writer = await self._dial(host, port)
        try:
            _enable_keepalive(writer)
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
        _enable_keepalive(writer)
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
