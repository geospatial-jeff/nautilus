"""Channels: one-way, backpressured, FIFO streams of frames between two operator instances.

Stage 0 provides :class:`InProcChannel`, a bounded ``asyncio.Queue``. The bound *is* the
backpressure: a full queue suspends the sender until the consumer drains it, so a slow operator
propagates backpressure to its upstream. Frames (data and control) share one ordered FIFO within a
process — there is no head-of-line hazard because the single consumer drains strictly in order.

The cross-process :class:`~nautilus.transport.socket_channel.SocketChannel` (Stage 1) keeps the same
``send``/``recv`` FIFO contract, but its backpressure is a credit window rather than a queue bound. It
runs over TCP — loopback between two local processes today, a node-to-node connection between machines
with no interface change.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from nautilus.core.records import Frame

DEFAULT_CAPACITY = 16


class Channel(ABC):
    """One-way backpressured FIFO of :class:`~nautilus.core.records.Frame`."""

    @abstractmethod
    async def send(self, frame: Frame) -> None:
        """Enqueue a frame, awaiting if the channel is full (backpressure)."""

    @abstractmethod
    async def recv(self) -> Frame:
        """Dequeue the next frame in FIFO order, awaiting if empty."""

    def depth(self) -> int | None:
        """Current queued frame count, or ``None`` if this channel cannot report it. Part of the
        interface (not a reflection probe) so a cross-process channel that cannot cheaply report
        depth — the ``SocketChannel`` — returns ``None`` explicitly rather than silently dropping
        the queue-depth fact.
        """
        return None

    def bytes_written(self) -> int | None:
        """Cumulative bytes this end has written to the wire, or ``None`` for an in-process channel
        (which moves no bytes). The sending :class:`~nautilus.runtime.actor.Output` records the delta as
        ``transport.bytes_sent``. Cumulative — not per-send — so a missed read never loses bytes."""
        return None

    def credit_wait_micros(self) -> int | None:
        """Cumulative microseconds a producer has blocked here awaiting flow-control credit, or ``None``
        for an in-process channel (whose backpressure is the queue bound, timed as ``edge.send_wait``).
        The sending :class:`~nautilus.runtime.actor.Output` records the delta as
        ``edge.credit_wait_micros``."""
        return None

    def encode_micros(self) -> int | None:
        """Cumulative microseconds this end spent serializing frames to the wire, or ``None`` for an
        in-process channel (which serializes nothing). The sending
        :class:`~nautilus.runtime.actor.Output` records the delta as ``transport.encode_micros``."""
        return None

    def decode_micros(self) -> int | None:
        """Cumulative microseconds this end spent deserializing frames from the wire, or ``None`` for an
        in-process channel. Accumulated in the background read loop; the receiving actor totals it as
        ``transport.decode_micros`` at close."""
        return None


class InProcChannel(Channel):
    """A bounded in-process channel backed by ``asyncio.Queue``."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._q: asyncio.Queue[Frame] = asyncio.Queue(maxsize=capacity)

    async def send(self, frame: Frame) -> None:
        await self._q.put(frame)

    async def recv(self) -> Frame:
        return await self._q.get()

    def depth(self) -> int | None:
        return self._q.qsize()
