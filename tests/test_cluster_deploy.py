"""Stage 2d: the smallest real distributed run — spawned workers over cross-worker socket edges.

``deploy`` must produce the same result as the serial run while genuinely crossing process boundaries.
A keyed shuffle at parallelism 2 forces real cross-worker edges, and results are compared as a multiset
(cross-worker interleave is nondeterministic). The bidirectional case (each worker holding both a
producer and a consumer end of the same peer) is the layout where a non-symmetric teardown would
deadlock. Failures must re-raise the child's traceback and reap every worker.

These tests spawn processes, so ``deploy`` is synchronous. Operators it ships must be reconstructable in
the child: the built-ins by import, and the raising operators below by value (cloudpickle is told to
pickle this module by value, so the child needs no test import).
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
from collections import Counter

import cloudpickle
import pyarrow as pa
import pytest

from nautilus.cluster import WorkerError, deploy
from nautilus.core.operator import Collector, OneInputOperator, OperatorContext
from nautilus.core.records import EOS_FRAME
from nautilus.driver.local import run_local_chain
from nautilus.driver.result import RunResult
from nautilus.driver.run import run_plan
from nautilus.operators import InMemorySource, KeyedCount, MapBatch, Tokenize
from nautilus.testing import data, op_counter, staged_graph

# Ship this module's operator classes by value, so a spawned worker reconstructs them without importing
# the test module (which it has no path to).
cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _RaiseOnOpen(OneInputOperator):
    def open(self, ctx: OperatorContext) -> None:
        raise RuntimeError("boom at open")

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        out.emit(batch)


class _RaiseOnProcess(OneInputOperator):
    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        raise RuntimeError("boom at process")


def _identity(batch: pa.RecordBatch) -> pa.RecordBatch:
    return batch  # a keyless stateless stage, so its inbound edge routes by position, not key


def _source() -> InMemorySource:
    return InMemorySource(
        [
            data(line=["the cat sat the dog ran the fox"]),
            data(line=["a fox and a cat and a dog and the"]),
            EOS_FRAME,
        ]
    )


def _wc(result: RunResult) -> Counter:
    return Counter((row["word"], row["count"]) for row in result.to_pylist())


# --- correctness over a genuine cross-worker edge ----------------------------------------------


def test_two_worker_keyed_wordcount_matches_serial() -> None:
    serial = asyncio.run(run_local_chain(_source(), [Tokenize("line", "word"), KeyedCount("word")]))
    graph = staged_graph(
        _source(),
        [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), 2, ("word",))],
    )
    result = deploy(graph, num_workers=2)
    assert _wc(result) == _wc(serial)
    # The keyed operator genuinely ran on both workers — proof the edge crossed a socket, not collapsed.
    nodes = {o.node for o in result.telemetry.operators if o.operator_id == "op1"}
    assert nodes == {"worker-0", "worker-1"}


def test_bidirectional_shuffle_matches_serial_and_completes_promptly() -> None:
    # tokenize at parallelism 2 -> keyedCount at parallelism 2 is a two-into-two shuffle: each
    # worker holds both an outbound producer end and an inbound consumer end of the other — the
    # layout where finish-then-close would deadlock.
    # Symmetric teardown drains it, so the whole run finishes well under the 5s per-channel drain timeout.
    serial = asyncio.run(run_local_chain(_source(), [Tokenize("line", "word"), KeyedCount("word")]))
    graph = staged_graph(
        _source(),
        [(Tokenize("line", "word"), 2, None), (KeyedCount("word"), 2, ("word",))],
    )
    result = deploy(graph, num_workers=2)
    assert _wc(result) == _wc(serial)
    # Each of the 2 keyed instances fanned in BOTH map instances, so its mailbox had 2 inputs and saw
    # EOS on each — the full local+remote input set was wired before the actor started.
    assert op_counter(result.telemetry, "op1", "eos.received") == 4


def test_same_plan_runs_under_inproc_and_socket_connectors() -> None:
    # HARD CONSTRAINT: one graph runs identically single-process (InProcessConnector) and across workers
    # (SocketConnector), by multiset.
    graph = staged_graph(
        _source(),
        [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), 2, ("word",))],
    )
    in_process = asyncio.run(run_plan(graph))
    distributed = deploy(graph, num_workers=2)
    assert _wc(in_process) == _wc(distributed)


# --- cross-process telemetry: the formerly-dead transport/placement metrics now populate ----------


def test_cross_worker_run_records_transport_and_placement() -> None:
    from nautilus.telemetry.catalog import Tier
    from nautilus.telemetry.recorder import TelemetryConfig

    graph = staged_graph(
        _source(),
        [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), 2, ("word",))],
    )
    rep = deploy(graph, num_workers=2, telemetry=TelemetryConfig(tier=Tier.FULL)).telemetry

    # transport.bytes_sent (FULL tier) is nonzero on the edge that genuinely crossed a socket.
    edge_bytes = {
        (dict(p.labels)["edge_src"], dict(p.labels)["edge_dst"]): p.value
        for o in rep.operators
        for p in o.counters
        if p.name == "transport.bytes_sent"
    }
    assert edge_bytes, "no transport.bytes_sent recorded over a cross-worker edge"
    assert any(v > 0 for v in edge_bytes.values())

    # edge.credit_wait_micros is recorded (>= 0) on those same socket edges, never on in-process ones.
    assert any(p.name == "edge.credit_wait_micros" for o in rep.operators for p in o.counters)

    # Arrow IPC serialization is timed on both ends: encode on the producer's edge, decode on the
    # receiving instance. Both only appear because data genuinely crossed a socket.
    assert any(p.name == "transport.encode_micros" for o in rep.operators for p in o.counters)
    assert any(p.name == "transport.decode_micros" for o in rep.operators for p in o.counters)

    # placement.instances_per_worker is recorded once per worker node and sums to the instance count.
    placements = {
        o.node: g.last
        for o in rep.operators
        for g in o.gauges
        if g.name == "placement.instances_per_worker"
    }
    assert set(placements) == {"worker-0", "worker-1"}
    instances = {(o.operator_id, o.subtask_index) for o in rep.operators if o.kind != "process"}
    assert sum(placements.values()) == len(
        instances
    )  # every instance is placed on exactly one worker


def test_equal_width_keyless_edge_co_locates_and_crosses_no_socket() -> None:
    # The regression guard for the forward-edge routing (data locality): source(1) -> op0(2) -> op1(2),
    # both maps keyless. op0 -> op1 is a 2 -> 2 keyless edge, so it forwards (sender i -> instance i);
    # with same-index placement across two workers op0[i] and op1[i] share a worker, so no DATA batch
    # crosses that edge's sockets. The source's 1 -> 2 fan-out has no 1:1 mapping and does cross. A revert
    # to round-robin on the middle edge would push half the data across a socket — caught here. (The
    # structural digest can't guard this: forward and round-robin give the same per-instance row counts on
    # a uniform stream, so only the bytes crossing a socket tell them apart.)
    #
    # transport.bytes_sent counts every frame over a socket, including the EOS each instance broadcasts to
    # its remote peers — so a co-located data edge still shows a few bytes of control traffic. The signal
    # is DATA, orders of magnitude larger: eight 4096-row int64 batches are ~32 KB each, so a data-carrying
    # socket edge is >> 1 KB while a control-only one is a handful of bytes.
    from nautilus.telemetry.catalog import Tier
    from nautilus.telemetry.recorder import TelemetryConfig

    src = InMemorySource(
        [data(n=list(range(i * 4096, (i + 1) * 4096))) for i in range(8)] + [EOS_FRAME]
    )
    graph = staged_graph(src, [(MapBatch(_identity), 2, None), (MapBatch(_identity), 2, None)])
    rep = deploy(graph, num_workers=2, telemetry=TelemetryConfig(tier=Tier.FULL)).telemetry

    edge_bytes = {
        (dict(p.labels)["edge_src"], dict(p.labels)["edge_dst"]): p.value
        for o in rep.operators
        for p in o.counters
        if p.name == "transport.bytes_sent"
    }
    # The forward edge carried only control frames across its sockets — no data batch (which would be tens
    # of KB) ever crossed. Round-robin here would push ~half the data over, blowing past this.
    assert edge_bytes.get(("op0", "op1"), 0) < 1024
    # Sanity that the run genuinely spans workers (so the check above means co-location, not disabled
    # sockets): the source's 1 -> 2 fan-out did send real data over a socket.
    assert edge_bytes.get(("source", "op0"), 0) > 10_000


# --- fail-fast: re-raise the child traceback and reap every worker -----------------------------


def test_failure_at_process_reraises_child_traceback_and_reaps() -> None:
    graph = staged_graph(
        _source(),
        [(Tokenize("line", "word"), 1, None), (_RaiseOnProcess(), 2, ("word",))],
    )
    with pytest.raises(WorkerError) as exc:
        deploy(graph, num_workers=2)
    assert "boom at process" in exc.value.child_traceback
    assert multiprocessing.active_children() == []  # every worker reaped


def test_failure_at_open_reraises_child_traceback_and_reaps() -> None:
    graph = staged_graph(
        _source(),
        [(Tokenize("line", "word"), 1, None), (_RaiseOnOpen(), 2, ("word",))],
    )
    with pytest.raises(WorkerError) as exc:
        deploy(graph, num_workers=2)
    assert "boom at open" in exc.value.child_traceback
    assert multiprocessing.active_children() == []
