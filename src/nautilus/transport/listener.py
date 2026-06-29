"""``EdgeListener``: one accept point per worker that routes each socket to the edge it names.

A worker runs a single TCP server. Every producer on another worker dials it and announces a
:class:`~nautilus.runtime.connector.ChannelId` (the :mod:`~nautilus.transport.handshake` preamble); the
listener reads that preamble and hands the socket to the consumer awaiting exactly that edge. Routing by
the handshake — not by accept order — is what lets all producers connect concurrently without the wires
crossing. Crossed wires would silently split a key's stream across the wrong downstream instances, so a
mis-route must be impossible, not merely unlikely.

The set of inbound edges a worker expects is fixed at bind time, before any producer connects, so the
two orderings both resolve correctly through one per-edge slot:

* a producer that connects *before* its consumer calls :meth:`accept` is **parked** — the slot holds
  the socket until the consumer asks for it;
* a consumer that calls :meth:`accept` *before* the producer connects awaits the slot until it arrives.

A connection whose handshake names an edge not in the expected set — or a second connection for an edge
already taken, or a truncated/foreign preamble — is rejected (its socket closed) and the listener stays
up for everyone else. Only an absent edge is an error; an out-of-order-but-valid one is normal.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from contextlib import suppress

from nautilus.runtime.connector import ChannelId
from nautilus.transport.handshake import read_handshake

_Conn = tuple[asyncio.StreamReader, asyncio.StreamWriter]

_HANDSHAKE_TIMEOUT = 10.0  # seconds to wait for a dialed peer's preamble before rejecting it
_CLOSE_TIMEOUT = 5.0  # bound on wait_closed() so a still-open claimed socket can't wedge teardown


class EdgeListener:
    """A worker's inbound accept point. Construct with the full set of inbound :class:`ChannelId`\\ s it
    will receive, :meth:`start` to bind, then :meth:`accept` each edge; :meth:`close` tears it down. Also
    usable as an async context manager."""

    def __init__(
        self,
        host: str,
        port: int,
        expected: Iterable[ChannelId],
        *,
        handshake_timeout: float = _HANDSHAKE_TIMEOUT,
        close_timeout: float = _CLOSE_TIMEOUT,
    ) -> None:
        self._host = host
        self._port = port
        self._expected = frozenset(expected)
        self._handshake_timeout = handshake_timeout
        self._close_timeout = close_timeout
        self._slots: dict[ChannelId, asyncio.Future[_Conn]] = {}
        self._claimed: set[ChannelId] = set()
        self._accept_tasks: set[asyncio.Task[None]] = set()
        self._server: asyncio.Server | None = None
        #: Connections closed for a foreign/unknown/duplicate/stalled handshake — surfaced for tests.
        self.rejected = 0

    async def start(self) -> None:
        """Bind the server and create one slot per expected edge. Call once, before any producer dials."""
        loop = asyncio.get_running_loop()
        self._slots = {cid: loop.create_future() for cid in self._expected}
        self._server = await asyncio.start_server(self._on_accept, self._host, self._port)

    @property
    def address(self) -> tuple[str, int]:
        """The bound ``(host, port)`` from ``getsockname()`` (the port is concrete even if 0 was
        requested). This is the *bind* address: when a worker binds all interfaces (``0.0.0.0``) the host
        is not itself dialable, so a cross-host worker advertises a separate routable host and takes only
        the concrete port from here."""
        if self._server is None:
            raise RuntimeError("EdgeListener.address read before start()")
        host, port = self._server.sockets[0].getsockname()[:2]
        return host, port

    async def accept(self, channel_id: ChannelId) -> _Conn:
        """Await the ``(reader, writer)`` of the socket whose handshake named ``channel_id``. Resolves
        immediately if the producer already connected (the socket was parked)."""
        if channel_id not in self._slots:
            raise KeyError(f"{channel_id} is not in this listener's expected inbound set")
        conn = await self._slots[channel_id]
        self._claimed.add(channel_id)
        return conn

    async def _on_accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # One accepted connection, run as its own task. Read its handshake off the raw stream (before any
        # SocketChannel) and resolve the named slot. A peer that stalls is bounded by the handshake
        # timeout; any failure closes only this connection — never the listener. The task is tracked so
        # close() can cancel a handshake still in flight at teardown.
        task = asyncio.current_task()
        if task is not None:
            self._accept_tasks.add(task)
        try:
            channel_id = await asyncio.wait_for(read_handshake(reader), self._handshake_timeout)
            slot = self._slots.get(channel_id)
            if slot is None or slot.done():  # unknown edge, or a second connection for one taken
                await self._reject(writer)
                return
            slot.set_result((reader, writer))
        except asyncio.CancelledError:
            writer.close()  # close() cancelled this in-flight handshake — release the socket
            raise
        except Exception:  # bad magic / timeout / truncated / foreign preamble — reject, stay up
            await self._reject(writer)
        finally:
            if task is not None:
                self._accept_tasks.discard(task)

    async def _reject(self, writer: asyncio.StreamWriter) -> None:
        self.rejected += 1
        writer.close()
        with suppress(OSError, ConnectionError, asyncio.CancelledError):
            await writer.wait_closed()

    async def close(self) -> None:
        """Tear the listener down without hanging. Stop accepting, cancel any handshake still in flight
        and any slot never connected, and close any parked-but-unclaimed socket — all *before* awaiting
        the server, because on Python 3.12 ``wait_closed()`` blocks until every accepted connection is
        closed, so a half-open or unclaimed socket would otherwise wedge it.

        A *claimed* slot's socket belongs to its :class:`SocketChannel`, so the caller must close every
        SocketChannel built from this listener (e.g. via ``SocketConnector.close()``) before calling
        this. If one is still open, ``wait_closed()`` is bounded by ``close_timeout`` rather than
        blocking forever."""
        if self._server is not None:
            self._server.close()
        # Cancel handshakes still reading, releasing their sockets so they don't leak or block the wait.
        for task in list(self._accept_tasks):
            task.cancel()
        for task in list(self._accept_tasks):
            with suppress(asyncio.CancelledError):
                await task
        # Cancel slots never connected; close parked-but-unclaimed sockets (a claimed slot's socket is
        # owned by its SocketChannel). Done before wait_closed() so neither can block it.
        for channel_id, slot in self._slots.items():
            if not slot.done():
                slot.cancel()
            elif (
                channel_id not in self._claimed
                and not slot.cancelled()
                and slot.exception() is None
            ):
                _, writer = slot.result()
                writer.close()
                with suppress(OSError, ConnectionError, asyncio.CancelledError):
                    await writer.wait_closed()
        if self._server is not None:
            with suppress(OSError, asyncio.TimeoutError):
                await asyncio.wait_for(self._server.wait_closed(), self._close_timeout)
        self._slots = {}
        self._accept_tasks = set()

    async def __aenter__(self) -> EdgeListener:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
