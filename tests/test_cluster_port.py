"""Characterization tests for the cluster control plane — the pure/near-pure pieces a Rust port must
reproduce byte-for-byte.

These pin the *physical half of co-partitioning* (placement) and the two consumers that read it
(``edge_resolver`` for dialing, ``cross_worker_inbound`` for the listener's expected set), the control-wire
framer (``take_message``), the sink-result codec (``encode_batches``/``decode_batches``), and the one
argument check ``deploy`` makes before it touches a socket. Everything that needs a live coordinator or
worker socket, a subprocess, or a real bind is out of scope (see the module's ``itemsSkipped``): those are
covered by the socket-driven suites and are not hermetic here.

Every golden below was produced by running the real code once and pinning its actual output — no value is
hand-derived.
"""

from __future__ import annotations

import numpy as np
import pytest

from nautilus.api import LogicalVertex, linear_graph
from nautilus.cluster.control_link import encode, take_message
from nautilus.cluster.coordinator import deploy
from nautilus.cluster.membership import AddressBook, edge_resolver
from nautilus.cluster.placement import place
from nautilus.cluster.protocol import Register, decode_batches, encode_batches
from nautilus.cluster.worker_main import cross_worker_inbound
from nautilus.compile import compile_graph
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.runtime.connector import ChannelId
from nautilus.tensors import is_tensor
from nautilus.testing import batch


def _plan(parallelism: int):
    """A fixed source -> Tokenize -> KeyedCount(parallelism) -> sink graph. KeyedCount is the only wide
    operator, so its second instance is the one edge that crosses workers under round-robin placement.
    """
    return compile_graph(
        linear_graph(
            lambda: InMemorySource([]),
            [
                LogicalVertex("op0", lambda: Tokenize("line", "word"), "one_input"),
                LogicalVertex(
                    "op1", lambda: KeyedCount("word"), "one_input", parallelism, ("word",)
                ),
            ],
        )
    )


# --- (a) placement: the golden {vertex -> worker} map for a fixed graph + roster -----------------


def test_placement_golden_map_two_workers() -> None:
    """The whole placement map for the fixed 2-worker deploy. Everything single-instance lands on worker
    0 (co-located, so those edges stay in-process); only ``op1[1]`` crosses to worker 1. This is the
    physical half of co-partitioning that ``edge_resolver`` and ``cross_worker_inbound`` both read.
    """
    placement = place(_plan(2), [0, 1])
    assert placement == {
        ("source", 0): 0,
        ("op0", 0): 0,
        ("op1", 0): 0,
        ("op1", 1): 1,
        ("sink", 0): 0,
    }


# --- (b) edge_resolver + address_of --------------------------------------------------------------


def test_edge_resolver_dials_the_destination_workers_listener() -> None:
    """A producer dialing a cross-worker edge connects to the listener of the worker hosting the edge's
    *destination* instance, not its source. ``op0[0]`` is on worker 0 but ``op1[1]`` is on worker 1, so
    the resolved address is worker 1's."""
    placement = place(_plan(2), [0, 1])
    address_book = AddressBook({0: ("10.0.0.1", 5000), 1: ("10.0.0.2", 6000)})
    resolve = edge_resolver(placement, address_book)
    edge = ChannelId("op0", 0, "op1", 1)  # src on worker 0, dst on worker 1
    assert placement[("op1", 1)] == 1  # the destination instance
    assert resolve(edge) == ("10.0.0.2", 6000)  # worker 1's listener, not worker 0's


def test_address_of_raises_keyerror_for_unknown_worker() -> None:
    address_book = AddressBook({0: ("10.0.0.1", 5000)})
    assert address_book.address_of(0) == ("10.0.0.1", 5000)
    with pytest.raises(KeyError):
        address_book.address_of(99)


# --- (c) cross_worker_inbound over the full src x dst product ------------------------------------


def test_cross_worker_inbound_is_dst_here_and_src_elsewhere() -> None:
    """The listener's expected set for a worker is exactly the ``(src, dst)`` edges whose destination it
    hosts and whose source it does not — the remote-only slice of the full src x dst product. For the
    fixed 2-worker plan each worker expects exactly one such edge."""
    plan = _plan(2)
    placement = place(plan, [0, 1])

    # Worker 0 hosts sink[0]; the only remote producer into it is op1[1] on worker 1.
    assert cross_worker_inbound(plan, placement, 0) == {ChannelId("op1", 1, "sink", 0)}
    # Worker 1 hosts op1[1]; its only remote producer is op0[0] on worker 0.
    assert cross_worker_inbound(plan, placement, 1) == {ChannelId("op0", 0, "op1", 1)}


def test_cross_worker_inbound_excludes_co_located_edges() -> None:
    """An edge whose src and dst share this worker is in-process, never in the listener's set. Every
    edge into worker 0 that stays on worker 0 (source->op0, op0->op1[0], op1[0]->sink) is absent, and the
    two sets never overlap — an edge is expected on exactly the worker that hosts its destination.
    """
    plan = _plan(2)
    placement = place(plan, [0, 1])
    inbound0 = cross_worker_inbound(plan, placement, 0)
    inbound1 = cross_worker_inbound(plan, placement, 1)
    assert ChannelId("op0", 0, "op1", 0) not in inbound0  # co-located, in-process
    assert ChannelId("source", 0, "op0", 0) not in inbound0  # co-located, in-process
    assert inbound0.isdisjoint(inbound1)


# --- (d) control_link take_message: whole frames FIFO, trailing partial -------------------------


def test_take_message_pops_whole_frames_fifo_then_none_on_partial() -> None:
    """``take_message`` pops complete frames off the front FIFO, consuming them in place, and returns
    ``None`` once only a partial frame remains. Buffer = encode(m1)+encode(m2)+encode(m3)[:5] pops m1,
    then m2, then ``None`` (m3's frame is truncated to just its header prefix)."""
    m1, m2, m3 = Register(0, "a", 1), Register(1, "b", 2), Register(2, "c", 3)
    buffer = bytearray(encode(m1) + encode(m2) + encode(m3)[:5])
    assert take_message(buffer) == m1
    assert take_message(buffer) == m2
    assert take_message(buffer) is None  # only m3's 5-byte prefix remains
    assert bytes(buffer) == encode(m3)[:5]  # the partial tail is left in place, unconsumed


# --- (e) protocol encode_batches / decode_batches -----------------------------------------------


def test_fixed_shape_tensor_batch_survives_encode_decode() -> None:
    """A fixed_shape_tensor column round-trips through the Arrow-IPC sink codec — schema and values
    both — which is why the codec exists (a pickled RecordBatch would drop the canonical extension type).
    """
    imgs = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rb = batch(idx=[10, 20], image=imgs)
    assert is_tensor(rb.column("image").type)  # a real fixed_shape_tensor column

    decoded = decode_batches(encode_batches([rb]))
    assert len(decoded) == 1
    assert decoded[0].schema.equals(rb.schema)  # extension type preserved
    assert decoded[0].equals(rb)  # values preserved


def test_empty_batches_encode_and_decode_to_nothing() -> None:
    """No batches is the sink-less worker's case: an empty list encodes to empty bytes, and empty bytes
    decode back to an empty list — no Arrow stream is ever written or read."""
    assert encode_batches([]) == b""
    assert decode_batches(b"") == []


# --- (f) deploy argument check ------------------------------------------------------------------


def test_deploy_rejects_empty_daemon_roster() -> None:
    """``deploy`` with an empty ``daemons`` roster fails before it touches a socket: the roster is the
    worker count in the remote path, so an empty one has no worker to run on."""
    graph = linear_graph(
        lambda: InMemorySource([]),
        [LogicalVertex("op0", lambda: Tokenize("line", "word"), "one_input")],
    )
    with pytest.raises(ValueError, match="roster is empty"):
        deploy(graph, daemons=[])
