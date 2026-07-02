"""Stage 2 of the cluster dashboard: the coordinator aggregates worker heartbeats into a live report.

This spawns real worker processes (``deploy`` is synchronous), so a keyed shuffle at parallelism 2 is a
genuine two-node run whose live reports must cover both workers. Built-in operators only, so a spawned
child reconstructs them by import. The live path is additive — the returned final report is unchanged —
which the multiset-vs-serial and convergence assertions pin.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from nautilus.cluster import deploy
from nautilus.core.records import EOS_FRAME
from nautilus.driver.local import run_local_chain
from nautilus.driver.result import RunResult
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.telemetry.report import RunReport
from nautilus.testing import data, staged_graph


def _source() -> InMemorySource:
    return InMemorySource(
        [
            data(line=["the cat sat the dog ran the fox"]),
            data(line=["a fox and a cat and a dog and the"]),
            EOS_FRAME,
        ]
    )


def _graph():
    return staged_graph(
        _source(),
        [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), 2, ("word",))],
    )


def _wc(result: RunResult) -> Counter:
    return Counter((row["word"], row["count"]) for row in result.to_pylist())


def test_coordinator_aggregates_worker_heartbeats_live() -> None:
    serial = asyncio.run(run_local_chain(_source(), [Tokenize("line", "word"), KeyedCount("word")]))
    reports: list[RunReport] = []
    result = deploy(
        _graph(), num_workers=2, on_report=reports.append, heartbeat_interval_micros=5_000
    )

    # Additive: the live path does not disturb the final result — it is exactly the serial multiset.
    assert _wc(result) == _wc(serial)

    # Heartbeats arrived, and across them both worker nodes contributed a process row (proof each worker
    # snapshotted itself and the coordinator merged by node).
    assert reports, "no live report was published"
    assert {o.node for rep in reports for o in rep.operators if o.kind == "process"} == {
        "worker-0",
        "worker-1",
    }

    # The last live report is built on the final Done, so it has converged to the authoritative totals.
    assert result.telemetry.summary.total_rows_out > 0
    assert reports[-1].summary.total_rows_out == result.telemetry.summary.total_rows_out
