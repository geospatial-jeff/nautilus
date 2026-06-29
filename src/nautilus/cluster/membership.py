"""Membership and addressing: who is in the job, and where to dial each edge.

The :class:`AddressBook` is the static per-job map from a worker id to the ``(host, port)`` its
:class:`~nautilus.transport.listener.EdgeListener` bound — built once by the coordinator after every
worker has bound, and broadcast unchanged. It does not change during the run (membership is fixed; a
rescale is a new job).

:func:`edge_resolver` turns placement + the address book into the resolver a
:class:`~nautilus.transport.connector.SocketConnector` needs: a producer dialing an edge connects to the
listener of the worker that hosts the edge's *destination* instance. Keeping this here — plain data and a
closure over it — is why ``transport`` never imports ``cluster``: the control plane hands the connector a
resolver, not an address book.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nautilus.runtime.connector import ChannelId


@dataclass(frozen=True)
class AddressBook:
    """Maps each worker id to the ``(host, port)`` of its :class:`EdgeListener`."""

    addresses: dict[int, tuple[str, int]]

    def address_of(self, worker_id: int) -> tuple[str, int]:
        return self.addresses[worker_id]


def edge_resolver(
    placement: dict[tuple[str, int], int], address_book: AddressBook
) -> Callable[[ChannelId], tuple[str, int]]:
    """A resolver for :meth:`SocketConnector.outbound`: the address a producer dials for an edge is the
    listener of the worker hosting the edge's destination instance."""

    def resolve(channel_id: ChannelId) -> tuple[str, int]:
        dst = (channel_id.dst_operator_id, channel_id.dst_subtask)
        return address_book.address_of(placement[dst])

    return resolve
