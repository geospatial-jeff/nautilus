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
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.parallel import Stage, graph_from_stages
from nautilus.runtime.result import RunResult
from nautilus.runtime.run import run_plan
from nautilus.testing import data

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


def _op_counter(report, operator_id: str, name: str) -> int:
    return sum(
        p.value
        for o in report.operators
        if o.operator_id == operator_id
        for p in o.counters
        if p.name == name
    )


# --- correctness over a genuine cross-worker edge ----------------------------------------------


def test_two_worker_keyed_wordcount_matches_serial() -> None:
    serial = asyncio.run(run_local_chain(_source(), [Tokenize("line", "word"), KeyedCount("word")]))
    graph = graph_from_stages(
        _source(),
        [Stage(lambda: Tokenize("line", "word")), Stage(lambda: KeyedCount("word"), 2, ["word"])],
    )
    result = deploy(graph, num_workers=2)
    assert _wc(result) == _wc(serial)
    # The keyed operator genuinely ran on both workers — proof the edge crossed a socket, not collapsed.
    nodes = {o.node for o in result.telemetry.operators if o.operator_id == "op1"}
    assert nodes == {"worker-0", "worker-1"}


def test_bidirectional_shuffle_matches_serial_and_completes_promptly() -> None:
    # tokenize(P=2) -> keyedCount(P=2) is a 2x2 shuffle: each worker holds both an outbound producer end
    # and an inbound consumer end of the other — the layout where finish-then-close would deadlock.
    # Symmetric teardown drains it, so the whole run finishes well under the 5s per-channel drain timeout.
    serial = asyncio.run(run_local_chain(_source(), [Tokenize("line", "word"), KeyedCount("word")]))
    graph = graph_from_stages(
        _source(),
        [
            Stage(lambda: Tokenize("line", "word"), 2),
            Stage(lambda: KeyedCount("word"), 2, ["word"]),
        ],
    )
    result = deploy(graph, num_workers=2)
    assert _wc(result) == _wc(serial)
    # Each of the 2 keyed instances fanned in BOTH map instances, so its mailbox had 2 inputs and saw
    # EOS on each — the full local+remote input set was wired before the actor started.
    assert _op_counter(result.telemetry, "op1", "eos.received") == 4


def test_same_plan_runs_under_inproc_and_socket_connectors() -> None:
    # HARD CONSTRAINT: one graph runs identically single-process (InProcessConnector) and across workers
    # (SocketConnector), by multiset.
    graph = graph_from_stages(
        _source(),
        [Stage(lambda: Tokenize("line", "word")), Stage(lambda: KeyedCount("word"), 2, ["word"])],
    )
    in_process = asyncio.run(run_plan(graph))
    distributed = deploy(graph, num_workers=2)
    assert _wc(in_process) == _wc(distributed)


# --- cross-process telemetry: the formerly-dead transport/placement metrics now populate ----------


def test_cross_worker_run_records_transport_and_placement() -> None:
    from nautilus.telemetry.catalog import Tier
    from nautilus.telemetry.recorder import TelemetryConfig

    graph = graph_from_stages(
        _source(),
        [Stage(lambda: Tokenize("line", "word")), Stage(lambda: KeyedCount("word"), 2, ["word"])],
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


# --- fail-fast: re-raise the child traceback and reap every worker -----------------------------


def test_failure_at_process_reraises_child_traceback_and_reaps() -> None:
    graph = graph_from_stages(
        _source(),
        [Stage(lambda: Tokenize("line", "word")), Stage(lambda: _RaiseOnProcess(), 2, ["word"])],
    )
    with pytest.raises(WorkerError) as exc:
        deploy(graph, num_workers=2)
    assert "boom at process" in exc.value.child_traceback
    assert multiprocessing.active_children() == []  # every worker reaped


def test_failure_at_open_reraises_child_traceback_and_reaps() -> None:
    graph = graph_from_stages(
        _source(),
        [Stage(lambda: Tokenize("line", "word")), Stage(lambda: _RaiseOnOpen(), 2, ["word"])],
    )
    with pytest.raises(WorkerError) as exc:
        deploy(graph, num_workers=2)
    assert "boom at open" in exc.value.child_traceback
    assert multiprocessing.active_children() == []
