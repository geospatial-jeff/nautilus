"""Stage 1 of the cluster dashboard: a worker pushes its recorder snapshot on an interval.

The heartbeat carries the same ``InstanceSnapshot``s as the terminal ``Done``, so the coordinator can
rebuild the cross-worker report mid-run; it is a telemetry channel, not a liveness one. These tests pin
the payload and the guarantee that turning it on never changes a run's identity. The end-to-end path
(coordinator aggregating heartbeats into a live report) is exercised in the deploy tests as later stages
land.
"""

from __future__ import annotations

from nautilus.compile import compile_graph
from nautilus.core.records import EOS_FRAME
from nautilus.core.time import TestClock
from nautilus.driver.local import run_local_chain
from nautilus.driver.pipeline import graph_from_pipeline
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.runtime.connector import Deployment, InProcessConnector
from nautilus.runtime.execute import execute
from nautilus.telemetry.model import InstanceSnapshot
from nautilus.telemetry.recorder import TelemetryConfig
from nautilus.testing import data


def _wordcount_plan():
    return compile_graph(
        graph_from_pipeline(
            InMemorySource([data(line=["the cat sat"]), data(line=["the dog ran"]), EOS_FRAME]),
            [Tokenize("line", "word"), KeyedCount("word")],
            1,
        )
    )


async def test_execute_delivers_heartbeats_carrying_snapshots() -> None:
    # execute creates the heartbeat task before the actor tasks and fires it once immediately, so a
    # finished run always delivered at least one — this asserts the payload shape, not a cadence.
    seen: list[list[InstanceSnapshot]] = []
    await execute(
        _wordcount_plan(),
        InProcessConnector(DEFAULT_CAPACITY),
        Deployment.single_worker(),
        clock=TestClock(),
        heartbeat=seen.append,
        heartbeat_interval_micros=1_000,
    )
    assert seen, "expected at least one heartbeat"
    assert all(isinstance(snaps, list) for snaps in seen)
    assert all(isinstance(s, InstanceSnapshot) for snaps in seen for s in snaps)
    assert len(seen[0]) > 0  # the recorders Phase A/B registered are present by the first fire


async def test_execute_emits_no_heartbeat_when_unset() -> None:
    # No callback → the periodic task is never created; the run completes exactly as before.
    result = await execute(
        _wordcount_plan(),
        InProcessConnector(DEFAULT_CAPACITY),
        Deployment.single_worker(),
        clock=TestClock(),
    )
    assert result.snapshots  # a normal run, unaffected


def test_heartbeat_off_by_default() -> None:
    assert TelemetryConfig().heartbeat_interval_micros is None


async def test_heartbeat_interval_excluded_from_run_identity() -> None:
    # The interval changes only how often telemetry ships, never a recorded value — so two runs that
    # differ only in it carry the same structural digest AND the same config digest. Guards against the
    # field ever being folded into either.
    frames = [data(line=["the cat sat", "the dog ran"]), EOS_FRAME]
    transforms = [Tokenize("line", "word"), KeyedCount("word")]
    with_hb = await run_local_chain(
        InMemorySource(list(frames)),
        list(transforms),
        clock=TestClock(),
        telemetry=TelemetryConfig(clock=TestClock(), heartbeat_interval_micros=1_000),
    )
    without = await run_local_chain(
        InMemorySource(list(frames)),
        list(transforms),
        clock=TestClock(),
        telemetry=TelemetryConfig(clock=TestClock()),
    )
    assert with_hb.telemetry.structural_digest() == without.telemetry.structural_digest()
    assert with_hb.telemetry.meta.config_digest == without.telemetry.meta.config_digest
