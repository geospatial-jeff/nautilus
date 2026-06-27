"""The connector: how the executor obtains the channels for one worker's slice of the mesh.

The executor wires actors without knowing whether an edge is in-process or crosses to another worker.
A :class:`Connector` hides that: given a :class:`ChannelId` — a directed instance-to-instance edge —
it returns the send end (the producer's :class:`~nautilus.runtime.channel.Channel`) or the recv end
(the consumer's). The in-process connector here returns one :class:`~nautilus.runtime.channel.InProcChannel`
for both ends of an id, so a single process runs the whole plan; the socket connector (Stage 2c) returns
the two ends of a TCP edge with the *same* signature, which is what lets one plan slice run unchanged in
one process or across the network.

The connector also owns teardown, because only it knows which edges are cross-worker: :meth:`finish`
drains outbound socket edges on a clean stop, :meth:`close` tears everything down. Both are no-ops
in-process — an :class:`~nautilus.runtime.channel.InProcChannel` has nothing to drain or close — so the
executor drives teardown the same way regardless of transport.

A :class:`Deployment` is the plain-data placement the executor reads: which node it is (so its hardware
telemetry is attributed to the right worker) and which operator instances it hosts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from nautilus.runtime.channel import DEFAULT_CAPACITY, Channel, InProcChannel


@dataclass(frozen=True, slots=True)
class ChannelId:
    """One directed channel: from subtask ``src_subtask`` of ``src_operator_id`` to subtask
    ``dst_subtask`` of ``dst_operator_id``. Hashable and transport-neutral — it is also the preamble a
    socket edge announces itself with (Stage 2c), so a producer and the consumer that awaits it name
    the same channel from opposite ends."""

    src_operator_id: str
    src_subtask: int
    dst_operator_id: str
    dst_subtask: int


class Connector(ABC):
    """Resolves a :class:`ChannelId` to its send/recv :class:`~nautilus.runtime.channel.Channel` and
    owns mesh teardown. One implementation per transport."""

    @abstractmethod
    async def outbound(self, channel_id: ChannelId) -> Channel:
        """The producer's send end of ``channel_id``."""

    @abstractmethod
    async def inbound(self, channel_id: ChannelId) -> Channel:
        """The consumer's recv end of ``channel_id``."""

    @abstractmethod
    async def finish(self) -> None:
        """Graceful symmetric teardown: drain this connector's outbound edges and close its inbound
        edges *concurrently* (one gather), so every worker emits its FIN at once and a bidirectional
        mesh cannot circular-wait on each peer's drain. A no-op in-process."""

    @abstractmethod
    async def close(self) -> None:
        """Tear every channel down, abortively. A no-op in-process; over sockets it cancels read loops
        so a peer's ``recv()`` raises promptly. Idempotent — safe to call after :meth:`finish`."""


@dataclass(frozen=True, slots=True)
class Deployment:
    """The placement a worker reads: its ``node`` label and which ``(operator_id, subtask_index)``
    instances it hosts (``None`` = host every instance, the single-process case)."""

    node: str
    hosted: frozenset[tuple[str, int]] | None = None

    @staticmethod
    def single_worker(node: str = "local") -> Deployment:
        """One worker hosting the whole plan; its hardware telemetry is attributed to ``node`` (pinned
        to ``"local"`` so a single-process report is identical to the legacy single-process run)."""
        return Deployment(node=node, hosted=None)

    def hosts(self, operator_id: str, subtask_index: int) -> bool:
        return self.hosted is None or (operator_id, subtask_index) in self.hosted


class InProcessConnector(Connector):
    """Every edge is an in-process :class:`~nautilus.runtime.channel.InProcChannel`: one channel object
    is both the send and the recv end of an id (as in the legacy in-process mesh), created lazily the
    first time either end is asked for. Teardown is a genuine no-op — an in-process channel has no
    socket to drain or close."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._channels: dict[ChannelId, InProcChannel] = {}

    def _channel(self, channel_id: ChannelId) -> InProcChannel:
        channel = self._channels.get(channel_id)
        if channel is None:
            channel = InProcChannel(self._capacity)
            self._channels[channel_id] = channel
        return channel

    async def outbound(self, channel_id: ChannelId) -> Channel:
        return self._channel(channel_id)

    async def inbound(self, channel_id: ChannelId) -> Channel:
        return self._channel(channel_id)

    async def finish(self) -> None:
        return None

    async def close(self) -> None:
        return None
