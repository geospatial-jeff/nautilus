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
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, cast

import cloudpickle

from nautilus.cluster.membership import AddressBook, edge_resolver
from nautilus.cluster.protocol import Done, Failed, Heartbeat, Register, encode_batches
from nautilus.compile.plan import PhysicalPlan
from nautilus.runtime.channel import Channel
from nautilus.runtime.connector import ChannelId, Connector, Deployment, InProcessConnector
from nautilus.runtime.execute import execute
from nautilus.telemetry import TelemetryConfig
from nautilus.telemetry.model import InstanceSnapshot
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


async def run_worker_slice(
    worker_id: int,
    plan_bytes: bytes,
    placement: dict[tuple[str, int], int],
    bind_host: str,
    advertise_host: str,
    capacity: int,
    config: TelemetryConfig,
    send_event: Callable[[Any], None],
    recv_address_book: Callable[[], Awaitable[AddressBook]],
    node_host: str | None = None,
) -> None:
    """Run this worker's data-plane slice, independent of how its control messages travel. Binds the
    listener, registers via ``send_event``, awaits the address book via ``recv_address_book``, runs
    :func:`~nautilus.runtime.execute.execute`, and reports ``Done`` or ``Failed`` via ``send_event``. The
    local spawn path backs the callables with ``multiprocessing`` queues; the remote daemon backs them
    with its control socket — so both run the exact same slice, the firewall against the two paths
    drifting. A failure is caught and reported as ``Failed`` (never raised) so the caller's transport
    always sees one terminal event; a *cancellation* (an abort) is deliberately not caught, so it unwinds
    to the daemon's teardown.

    ``node_host`` is the worker's physical-host identity for telemetry attribution. ``None`` (the local
    spawn path) keeps the node label ``worker-<id>``, so single-machine reports are unchanged; a daemon
    passes its advertised host, so a multi-node report tags each row ``worker-<id>@<host>`` and a reader
    can tell *which container* an operator ran on — not just its logical worker id."""
    listener: EdgeListener | None = None
    try:
        plan = cast(PhysicalPlan, cloudpickle.loads(plan_bytes))
        listener = EdgeListener(bind_host, 0, cross_worker_inbound(plan, placement, worker_id))
        await listener.start()
        # Register the routable advertised host with the concrete bound port — never the bind host, which
        # is 0.0.0.0 (undialable) when binding all interfaces.
        send_event(Register(worker_id, advertise_host, listener.address[1]))
        # Await the address book. The coordinator sends it only once every worker has registered (bound),
        # so by now every destination listener exists and dialing can't miss one.
        address_book = await recv_address_book()

        def is_local(channel_id: ChannelId) -> bool:
            src = (channel_id.src_operator_id, channel_id.src_subtask)
            dst = (channel_id.dst_operator_id, channel_id.dst_subtask)
            return placement[src] == placement[dst]

        connector = HybridConnector(
            is_local,
            InProcessConnector(capacity),
            SocketConnector(listener, edge_resolver(placement, address_book), capacity=capacity),
        )
        node = f"worker-{worker_id}" if node_host is None else f"worker-{worker_id}@{node_host}"
        deployment = Deployment(
            node=node,
            hosted=frozenset(instance for instance, w in placement.items() if w == worker_id),
        )

        # Each worker samples its own process, attributed to its node, so the report has one process row
        # per worker; across machines the node carries the host so a reader sees which container it ran on.
        # The config is the coordinator's, minus its clock (see deploy()).
        #
        # When the coordinator attached a dashboard it set config.heartbeat_interval_micros; push this
        # worker's snapshot on that cadence over the same send_event the terminal Done uses. execute owns
        # the timer and the registry — here we only wrap each snapshot as a Heartbeat and ship it.
        def send_heartbeat(snaps: list[InstanceSnapshot]) -> None:
            send_event(Heartbeat(worker_id, snaps))

        interval = config.heartbeat_interval_micros
        result = await execute(
            plan,
            connector,
            deployment,
            capacity=capacity,
            config=config,
            heartbeat=send_heartbeat if interval is not None else None,
            heartbeat_interval_micros=interval or 500_000,
        )
        send_event(Done(worker_id, result.snapshots, encode_batches(result.sink_batches)))
    except Exception:
        send_event(Failed(worker_id, traceback.format_exc()))
    finally:
        if listener is not None:
            # After execute() closed the connector (its SocketChannels); the listener is closed last,
            # per the teardown order EdgeListener.close() documents.
            with suppress(Exception):
                await listener.close()


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
    """The local spawn path: back :func:`run_worker_slice`'s control callables with the worker's
    ``multiprocessing`` queues — ``events.put`` to report, a blocking ``commands.get`` (off the event
    loop) for the address book."""
    loop = asyncio.get_running_loop()
    await run_worker_slice(
        worker_id,
        plan_bytes,
        placement,
        bind_host,
        advertise_host,
        capacity,
        config,
        send_event=events.put,
        recv_address_book=lambda: loop.run_in_executor(None, commands.get),
    )
