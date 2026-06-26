"""Channels: one-way, backpressured, FIFO streams of frames between two operator instances.

Stage 0 provides :class:`InProcChannel`, a bounded ``asyncio.Queue``. The bound *is* the
backpressure: a full queue suspends the sender until the consumer drains it, so a slow operator
propagates backpressure to its upstream. Frames (data and control) share one ordered FIFO within a
process — there is no head-of-line hazard because the single consumer drains strictly in order.

The cross-process :class:`SocketChannel` (Stage 1) keeps the same ``send``/``recv`` interface but
gates data frames with credit while control frames stay credit-exempt. It runs over a TCP socket: the
loopback interface between two local processes today, a node-to-node connection between machines with
no interface change.
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
