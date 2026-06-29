"""Stage 4.2: the multi-node control path, exercised hermetically over loopback.

The remote analogue of ``test_cluster_deploy``: the coordinator *dials* long-lived ``nautilus worker``
daemons (real subprocesses, distinct control ports on 127.0.0.1) instead of spawning children, and the
distributed result must match the single-process run while a keyed shuffle genuinely crosses a socket. A
daemon stays up and serves more than one job; a worker failure re-raises the child's traceback; and a
roster longer than the plan's parallelism leaves the surplus daemon untouched. This needs no Docker, so it
runs in the default suite and keeps the multi-node control path green without containers.

The raising operator below is shipped by value (cloudpickle is told to pickle this module by value), so a
daemon subprocess reconstructs it without importing the test module.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import closing, contextmanager

import cloudpickle
import pyarrow as pa
import pytest

from nautilus.cluster import WorkerError, deploy
from nautilus.cluster.daemon import healthcheck
from nautilus.core.operator import Collector, OneInputOperator
from nautilus.core.records import EOS_FRAME
from nautilus.driver.local import run_local_chain
from nautilus.driver.result import RunResult
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.testing import data, staged_graph

cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _RaiseOnProcess(OneInputOperator):
    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        raise RuntimeError("boom at process")


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def _daemons(n: int) -> Iterator[list[tuple[str, int]]]:
    """Launch ``n`` worker daemons on loopback as subprocesses, wait until each accepts, yield the roster,
    and terminate them on exit."""
    ports = [_free_port() for _ in range(n)]
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "nautilus",
                "worker",
                "--listen",
                f"127.0.0.1:{port}",
                "--advertise",
                "127.0.0.1",
                "--bind",
                "127.0.0.1",
            ]
        )
        for port in ports
    ]
    try:
        deadline = time.monotonic() + 30
        for port in ports:
            while not healthcheck("127.0.0.1", port):
                if time.monotonic() > deadline:
                    raise RuntimeError(f"daemon on port {port} never came up")
                time.sleep(0.1)
        yield [("127.0.0.1", port) for port in ports]
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _source() -> InMemorySource:
    return InMemorySource(
        [
            data(line=["the cat sat the dog ran the fox"]),
            data(line=["a fox and a cat and a dog and the"]),
            EOS_FRAME,
        ]
    )


def _wordcount_graph() -> object:
    return staged_graph(
        _source(),
        [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), 2, ("word",))],
    )


def _wc(result: RunResult) -> Counter:
    return Counter((row["word"], row["count"]) for row in result.to_pylist())


def _serial() -> RunResult:
    return asyncio.run(run_local_chain(_source(), [Tokenize("line", "word"), KeyedCount("word")]))


@pytest.fixture(scope="module")
def roster() -> Iterator[list[tuple[str, int]]]:
    with _daemons(2) as addresses:
        yield addresses


def test_remote_deploy_matches_serial(roster: list[tuple[str, int]]) -> None:
    result = deploy(_wordcount_graph(), daemons=roster)
    assert _wc(result) == _wc(_serial())
    # The keyed operator genuinely ran on both daemons — proof the shuffle crossed a real socket. The node
    # label carries the daemon's advertised host (worker-<id>@<host>), so the report shows which container.
    nodes = {o.node for o in result.telemetry.operators if o.operator_id == "op1"}
    assert nodes == {"worker-0@127.0.0.1", "worker-1@127.0.0.1"}


def test_daemon_serves_a_second_job(roster: list[tuple[str, int]]) -> None:
    # The same daemons run another job after the first — they stay up (the run completing doesn't stop
    # them), and the result matches the single-process run again.
    result = deploy(_wordcount_graph(), daemons=roster)
    assert _wc(result) == _wc(_serial())
    digest_again = deploy(_wordcount_graph(), daemons=roster).telemetry.structural_digest()
    assert digest_again == result.telemetry.structural_digest()


def test_remote_worker_failure_reraises_child_traceback() -> None:
    graph = staged_graph(
        _source(),
        [(Tokenize("line", "word"), 1, None), (_RaiseOnProcess(), 2, ("word",))],
    )
    with _daemons(2) as addresses:
        with pytest.raises(WorkerError) as exc:
            deploy(graph, daemons=addresses)
    assert "boom at process" in exc.value.child_traceback


def test_daemons_recover_after_a_failed_job() -> None:
    # A job whose operator raises aborts the run (reap closes the control connections); the daemons must
    # return to idle rather than wedge or orphan their work, so a normal job on the SAME daemons right
    # afterward still succeeds. This exercises the abort/no-orphan path and recovery in one shot.
    failing = staged_graph(
        _source(),
        [(Tokenize("line", "word"), 1, None), (_RaiseOnProcess(), 2, ("word",))],
    )
    with _daemons(2) as roster:
        with pytest.raises(WorkerError):
            deploy(failing, daemons=roster)
        result = deploy(_wordcount_graph(), daemons=roster)
        assert _wc(result) == _wc(_serial())
        nodes = {o.node for o in result.telemetry.operators if o.operator_id == "op1"}
        assert nodes == {"worker-0@127.0.0.1", "worker-1@127.0.0.1"}


def test_roster_longer_than_parallelism_leaves_surplus_idle() -> None:
    # The plan's max parallelism is 2, so a 3-daemon roster dials only the first two; the surplus daemon
    # is never dialed (no Launch, not awaited, not treated as crashed) and stays healthy.
    with _daemons(3) as addresses:
        result = deploy(_wordcount_graph(), daemons=addresses)
        assert _wc(result) == _wc(_serial())
        surplus_host, surplus_port = addresses[2]
        assert healthcheck(surplus_host, surplus_port)  # untouched, still accepting
