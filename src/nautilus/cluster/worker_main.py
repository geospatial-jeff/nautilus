"""The spawned worker entrypoint and its hybrid connector.

A worker process runs one slice of a :class:`~nautilus.compile.plan.PhysicalPlan` — the instances
placement assigned to it. It is pure data plane: it binds a listener, wires its edges, runs its actors,
and reports the outcome to the coordinator. It never reads another worker except over data edges, and
never the coordinator except through the control messages in :mod:`nautilus.cluster.protocol`.

Its connector is **hybrid**: an edge whose endpoints are both on this worker is a free in-process
channel; an edge that crosses to another worker is a socket. ``execute`` wires both with identical code —
only the connector chooses per edge — which is the mixed in-process/socket mesh the cluster runs on.

Startup is the worker's half of the two-phase bootstrap: bind the listener and report its address, then
block (off the event loop) for the address book the coordinator broadcasts once *every* worker has
bound. Receiving it is the signal that all destination listeners exist, so dialing cannot fail for a
not-yet-bound peer.

The listener binds ``bind_host`` (all interfaces, ``0.0.0.0``, in a container) but the worker registers a
separate ``advertise_host`` — the routable address peers actually dial — because ``getsockname()`` on a
``0.0.0.0`` bind returns ``0.0.0.0``, which no peer can reach. Only the concrete bound port is taken from
the listener. On a single-machine run the two are equal (both loopback), so registration is unchanged.
"""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Callable
from contextlib import suppress
from typing import Any, cast

import cloudpickle

from nautilus.cluster.membership import AddressBook, edge_resolver
from nautilus.cluster.protocol import Done, Failed, Register, encode_batches
from nautilus.compile.plan import PhysicalPlan
from nautilus.runtime.channel import Channel
from nautilus.runtime.connector import ChannelId, Connector, Deployment, InProcessConnector
from nautilus.runtime.execute import execute
from nautilus.telemetry import TelemetryConfig
from nautilus.transport.connector import SocketConnector
from nautilus.transport.listener import EdgeListener


class HybridConnector(Connector):
    """Routes each edge to the in-process connector when ``is_local`` (both endpoints on this worker) or
    the socket connector when it crosses workers. Teardown fans out to both."""

    def __init__(
        self, is_local: Callable[[ChannelId], bool], local: Connector, remote: Connector
    ) -> None:
        self._is_local = is_local
        self._local = local
        self._remote = remote

    async def outbound(self, channel_id: ChannelId) -> Channel:
        connector = self._local if self._is_local(channel_id) else self._remote
        return await connector.outbound(channel_id)

    async def inbound(self, channel_id: ChannelId) -> Channel:
        connector = self._local if self._is_local(channel_id) else self._remote
        return await connector.inbound(channel_id)

    async def finish(self) -> None:
        await asyncio.gather(self._local.finish(), self._remote.finish())

    async def close(self) -> None:
        await asyncio.gather(self._local.close(), self._remote.close())


def cross_worker_inbound(
    plan: PhysicalPlan, placement: dict[tuple[str, int], int], worker_id: int
) -> set[ChannelId]:
    """The set of cross-worker edges whose destination instance this worker hosts — exactly what its
    :class:`EdgeListener` must expect at bind time. A control frame (e.g. EOS) is broadcast to every
    downstream channel, so every ``(u, d)`` of a connection is a real edge, co-located or remote."""
    width = {op.operator_id: op.parallelism for op in plan.operators}
    expected: set[ChannelId] = set()
    for edge in plan.edges:
        for u in range(width[edge.src_operator_id]):
            for d in range(width[edge.dst_operator_id]):
                src, dst = (edge.src_operator_id, u), (edge.dst_operator_id, d)
                if placement[dst] == worker_id and placement[src] != worker_id:
                    expected.add(ChannelId(edge.src_operator_id, u, edge.dst_operator_id, d))
    return expected


def worker_main(
    worker_id: int,
    plan_bytes: bytes,
    placement: dict[tuple[str, int], int],
    bind_host: str,
    advertise_host: str,
    capacity: int,
    config: TelemetryConfig,
    events: Any,
    commands: Any,
) -> None:
    """Spawn entrypoint (must be importable in the child): run this worker's slice and report the
    outcome. Reachable only via :func:`~nautilus.cluster.launcher.spawn_workers`, never imported by the
    data path."""
    asyncio.run(
        _run_worker(
            worker_id,
            plan_bytes,
            placement,
            bind_host,
            advertise_host,
            capacity,
            config,
            events,
            commands,
        )
    )


async def _run_worker(
    worker_id: int,
    plan_bytes: bytes,
    placement: dict[tuple[str, int], int],
    bind_host: str,
    advertise_host: str,
    capacity: int,
    config: TelemetryConfig,
    events: Any,
    commands: Any,
) -> None:
    listener: EdgeListener | None = None
    try:
        plan = cast(PhysicalPlan, cloudpickle.loads(plan_bytes))
        listener = EdgeListener(bind_host, 0, cross_worker_inbound(plan, placement, worker_id))
        await listener.start()
        # Register the routable advertised host with the concrete bound port — never the bind host, which
        # is 0.0.0.0 (undialable) when binding all interfaces.
        events.put(Register(worker_id, advertise_host, listener.address[1]))
        # Block for the address book off the event loop. The coordinator sends it only once every worker
        # has registered (bound), so by now every destination listener exists and dialing can't miss one.
        address_book = cast(
            AddressBook, await asyncio.get_running_loop().run_in_executor(None, commands.get)
        )

        def is_local(channel_id: ChannelId) -> bool:
            src = (channel_id.src_operator_id, channel_id.src_subtask)
            dst = (channel_id.dst_operator_id, channel_id.dst_subtask)
            return placement[src] == placement[dst]

        connector = HybridConnector(
            is_local,
            InProcessConnector(capacity),
            SocketConnector(listener, edge_resolver(placement, address_book), capacity=capacity),
        )
        deployment = Deployment(
            node=f"worker-{worker_id}",
            hosted=frozenset(instance for instance, w in placement.items() if w == worker_id),
        )
        # Each worker samples its own process, attributed to its node (worker-<id>), so the report has
        # one process row per worker. The config is the coordinator's, minus its clock (see deploy()).
        result = await execute(plan, connector, deployment, capacity=capacity, config=config)
        events.put(Done(worker_id, result.snapshots, encode_batches(result.sink_batches)))
    except Exception:
        events.put(Failed(worker_id, traceback.format_exc()))
    finally:
        if listener is not None:
            # After execute() closed the connector (its SocketChannels); the listener is closed last,
            # per the teardown order EdgeListener.close() documents.
            with suppress(Exception):
                await listener.close()
