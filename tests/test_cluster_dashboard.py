"""Stages 2–3 of the cluster dashboard: the coordinator aggregates worker heartbeats into a live report
(Stage 2), and ``serve_cluster`` serves that report over HTTP (Stage 3).

These spawn real worker processes (``deploy`` is synchronous), so a keyed shuffle at parallelism 2 is a
genuine two-node run whose live reports must cover both workers. Built-in operators only, so a spawned
child reconstructs them by import. The live path is additive — the returned final report is unchanged —
which the multiset-vs-serial and convergence assertions pin.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

from nautilus.cluster import deploy, serve_cluster
from nautilus.core.records import EOS_FRAME
from nautilus.core.time import TestClock
from nautilus.driver.local import run_local_chain
from nautilus.driver.result import RunResult
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.telemetry.live import LiveAggregator, load_dashboard_html
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


# --- Stage 3: the aggregator serves the report, and serve_cluster wires it end to end ------------


def test_live_aggregator_serves_report_with_status() -> None:
    agg = LiveAggregator()
    # Before any worker reports: a minimal live body the page can poll without erroring.
    doc0 = json.loads(agg.render_json())
    assert doc0["status"] == "live"
    assert "sampled_at_micros" in doc0
    assert "operators" not in doc0

    # A real report is served verbatim plus a status; mark_completed flips only the status.
    rep = asyncio.run(
        run_local_chain(_source(), [Tokenize("line", "word")], clock=TestClock())
    ).telemetry
    agg.update(rep)
    live = json.loads(agg.render_json())
    assert live["status"] == "live"
    assert live["schema_version"] == rep.schema_version
    assert any(o["kind"] == "process" for o in live["operators"])
    agg.mark_completed()
    assert json.loads(agg.render_json())["status"] == "completed"


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception:
        return 0, b""  # server torn down between polls, etc. — treated as "nothing to read"


def test_serve_cluster_serves_aggregated_dashboard() -> None:
    # serve_cluster runs the two-worker run in this (main) thread, spawning workers as the other deploy
    # tests do; a background poller reads the HTTP endpoint throughout and records what it served. Bounded
    # run + max_seconds linger, so it exits on its own.
    ready = threading.Event()
    stop = threading.Event()
    holder: dict[str, str] = {}
    procs_seen = 0
    statuses_seen: set[str] = set()

    def on_ready(url: str) -> None:
        holder["url"] = url
        ready.set()

    def poll() -> None:
        nonlocal procs_seen
        if not ready.wait(15):
            return
        url = holder["url"] + "api/telemetry.json"
        while not stop.is_set():
            status, body = _get(url)
            if status == 200:
                doc = json.loads(body)
                procs = [o for o in doc.get("operators", []) if o.get("kind") == "process"]
                procs_seen = max(procs_seen, len(procs))
                status_str = doc.get("status")
                if isinstance(status_str, str):
                    statuses_seen.add(status_str)
            time.sleep(0.03)

    poller = threading.Thread(target=poll)
    poller.start()
    result = serve_cluster(
        _graph(),
        num_workers=2,
        host="127.0.0.1",
        port=0,
        heartbeat_interval_micros=5_000,
        linger=True,
        max_seconds=2.0,
        on_ready=on_ready,
    )
    stop.set()
    poller.join(10)

    assert isinstance(result, RunResult)
    assert procs_seen == 2  # the dashboard served both workers' process rows
    assert (
        "completed" in statuses_seen
    )  # and flipped to completed while lingering on the final report


# --- Stage 4: the dashboard page renders every worker, not just the first -----------------------


def test_dashboard_html_renders_per_worker_hardware() -> None:
    html = load_dashboard_html().decode()
    assert "worker-label" in html  # one hardware group per worker node
    # It reads every process row, not just the first — the single-process assumption this stage removes.
    assert '.find(o=>o.kind==="process")' not in html
    assert 'filter(o=>o.kind==="process")' in html
