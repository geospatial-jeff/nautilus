"""Stage 4.1: bind-vs-advertise, a rejected non-routable advertise, and a bounded data dial.

A worker binds all interfaces but registers a separate routable advertised host; a ``0.0.0.0``/empty
advertise is rejected at the bind barrier with a clear message; and dialing an unreachable address fails
with a bounded ``TimeoutError`` instead of hanging — the three changes that let the same plan cross a real
network. The first runs real spawned workers (bind ``0.0.0.0``, advertise loopback — the container
scenario, hermetically); the others are unit-level.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter

import pytest

from nautilus.cluster import deploy
from nautilus.cluster.protocol import Register
from nautilus.cluster.rendezvous import bind_barrier
from nautilus.core.records import EOS_FRAME
from nautilus.driver.local import run_local_chain
from nautilus.driver.result import RunResult
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.runtime.connector import ChannelId
from nautilus.testing import data, staged_graph
from nautilus.transport.connector import SocketConnector


def _frames() -> list:
    return [data(line=["the cat the dog the fox a cat a dog"]), EOS_FRAME]


def _wc(result: RunResult) -> Counter:
    return Counter((row["word"], row["count"]) for row in result.to_pylist())


def test_bind_all_interfaces_advertise_loopback_matches_serial(monkeypatch) -> None:
    # The real container scenario, hermetically: bind 0.0.0.0 (all interfaces) but advertise a routable
    # host (127.0.0.1). Peers dial the advertised host and reach the all-interfaces listener, so the keyed
    # shuffle crosses a socket and the result matches the single-process run. A non-loopback bind now
    # requires the cluster secret (fail-closed), so this also drives the data-plane auth handshake end to
    # end — every cross-worker edge authenticates before it carries a frame.
    monkeypatch.setenv("NAUTILUS_CLUSTER_SECRET", "test-cluster-secret-long-enough-0123456789")
    serial = asyncio.run(
        run_local_chain(InMemorySource(_frames()), [Tokenize("line", "word"), KeyedCount("word")])
    )
    graph = staged_graph(
        InMemorySource(_frames()),
        [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), 2, ("word",))],
    )
    result = deploy(graph, num_workers=2, host="0.0.0.0", advertise_host="127.0.0.1")
    assert _wc(result) == _wc(serial)
    nodes = {o.node for o in result.telemetry.operators if o.operator_id == "op1"}
    assert nodes == {"worker-0", "worker-1"}  # the shuffle genuinely crossed the advertised edge


class _CannedCohort:
    """A cohort that yields one canned Register, for driving bind_barrier without real workers."""

    def __init__(self, register: Register) -> None:
        self._register = register
        self.sent: list[tuple[int, object]] = []

    def next_event(self, timeout: float | None, watch: set[int] | None = None) -> object:
        return self._register

    def send(self, worker_id: int, message: object) -> None:
        self.sent.append((worker_id, message))

    def reap(self) -> None: ...


@pytest.mark.parametrize("bad", ["0.0.0.0", "", "::"])
def test_non_routable_advertise_is_rejected(bad: str) -> None:
    cohort = _CannedCohort(Register(0, bad, 5000))
    with pytest.raises(ValueError, match="routable"):
        bind_barrier(cohort, 1, 1.0)  # type: ignore[arg-type]


def test_routable_advertise_passes_the_bind_barrier() -> None:
    cohort = _CannedCohort(Register(0, "worker-0", 5000))
    bind_barrier(cohort, 1, 1.0)  # type: ignore[arg-type]
    assert cohort.sent and cohort.sent[0][0] == 0  # the address book was broadcast to the worker


async def test_dial_to_a_black_hole_is_bounded() -> None:
    # 192.0.2.1 is TEST-NET-1 (RFC 5737): globally unrouted, so SYNs vanish and connect would hang —
    # which the connect timeout must turn into a bounded error rather than an indefinite wait.
    connector = SocketConnector(None, lambda c: ("192.0.2.1", 9), connect_timeout=0.5)  # type: ignore[arg-type]
    start = time.monotonic()
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await connector.outbound(ChannelId("a", 0, "b", 0))
    assert time.monotonic() - start < 5.0  # bounded by connect_timeout, not an indefinite hang
