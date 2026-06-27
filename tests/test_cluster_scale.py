"""Stage 2e: scaling deploy to N workers — correctness, placement invariance, capping, telemetry.

Distributed runs must equal the serial run by multiset across any worker count and parallelism, and the
structural digest must be placement-invariant (the same plan routes the same way no matter how many
workers host it). Hardware telemetry is attributed per worker (exactly W process rows). These tests
spawn processes, so they use only importable (module-level) operators.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter

from nautilus.cluster import deploy
from nautilus.core.records import EOS_FRAME
from nautilus.operators import InMemorySource, KeyedCount, KeyedTumblingSum, Tokenize
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.parallel import Stage, graph_from_stages
from nautilus.runtime.result import RunResult
from nautilus.telemetry import TelemetryConfig, Tier
from nautilus.testing import data, wm
from nautilus.windows import TumblingEventTimeWindows

_WORDS = ["the", "cat", "sat", "dog", "ran", "a", "fox", "x", "y", "z"]


def _multiset(result: RunResult) -> Counter:
    return Counter(tuple(sorted(row.items())) for row in result.to_pylist())


def _wordcount_frames(rng: random.Random) -> list:
    lines = [
        " ".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 5)))
        for _ in range(rng.randint(1, 4))
    ]
    return [data(line=[ln]) for ln in lines] + [EOS_FRAME]


def _windowed_frames(rng: random.Random) -> list:
    frames: list = []
    clock_t = 0
    for _ in range(rng.randint(2, 4)):
        m = rng.randint(1, 6)
        frames.append(
            data(
                key=[rng.choice(["a", "b", "c", "d"]) for _ in range(m)],
                val=[rng.randint(1, 5) for _ in range(m)],
                ts=[clock_t + rng.randint(0, 8) for _ in range(m)],
            )
        )
        clock_t += 10
        frames.append(wm(clock_t))
    frames.append(EOS_FRAME)
    return frames


def _windowed_stage(p: int) -> Stage:
    return Stage(
        lambda: KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10)), p, ["key"]
    )


# --- correctness across random W x parallelism --------------------------------------------------


def test_distributed_matches_serial_over_random_workers_and_parallelism() -> None:
    rng = random.Random(2024)
    for trial in range(6):
        workers = rng.choice([1, 2, 3])
        parallelism = rng.choice([1, 2, 3])
        if rng.random() < 0.5:
            frames = _wordcount_frames(rng)
            serial = asyncio.run(
                run_local_chain(
                    InMemorySource(frames), [Tokenize("line", "word"), KeyedCount("word")]
                )
            )
            graph = graph_from_stages(
                InMemorySource(frames),
                [
                    Stage(lambda: Tokenize("line", "word")),
                    Stage(lambda: KeyedCount("word"), parallelism, ["word"]),
                ],
            )
        else:
            frames = _windowed_frames(rng)
            serial = asyncio.run(
                run_local_chain(
                    InMemorySource(frames),
                    [KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))],
                )
            )
            graph = graph_from_stages(InMemorySource(frames), [_windowed_stage(parallelism)])
        result = deploy(graph, num_workers=workers)
        assert _multiset(result) == _multiset(serial), (trial, workers, parallelism)


def test_single_key_skew_terminates_across_workers() -> None:
    # Every row shares one key: the shuffle sends all data to one instance; the others receive only the
    # broadcast EOS over their sockets, advance event time, and forward EOS. It must still terminate.
    frames = [
        data(key=["a"] * 8, val=list(range(8)), ts=list(range(8))),
        wm(10),
        EOS_FRAME,
    ]
    serial = asyncio.run(
        run_local_chain(
            InMemorySource(frames),
            [KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))],
        )
    )
    result = deploy(graph_from_stages(InMemorySource(frames), [_windowed_stage(4)]), num_workers=4)
    assert _multiset(result) == _multiset(serial)


# --- placement invariance + capping -------------------------------------------------------------


def _wordcount_graph():
    return graph_from_stages(
        InMemorySource([data(line=["the cat the dog the fox a cat a dog"]), EOS_FRAME]),
        [Stage(lambda: Tokenize("line", "word")), Stage(lambda: KeyedCount("word"), 3, ["word"])],
    )


def test_structural_digest_is_placement_invariant_across_worker_counts() -> None:
    # The same plan at fixed parallelism routes the same way regardless of how many workers host it, so
    # the structural digest is identical across W in {1, 2, 3} (different addresses, same routing).
    digests = {
        deploy(_wordcount_graph(), num_workers=w).telemetry.structural_digest() for w in (1, 2, 3)
    }
    assert len(digests) == 1, digests


def test_workers_capped_at_max_parallelism() -> None:
    # max parallelism here is 3 (KeyedCount); asking for 8 workers caps to 3 — no empty worker, no hang.
    result = deploy(_wordcount_graph(), num_workers=8)
    nodes = {o.node for o in result.telemetry.operators if o.kind != "process"}
    assert nodes == {"worker-0", "worker-1", "worker-2"}


def test_host_is_a_parameter_and_routing_is_address_independent() -> None:
    # Workers bind distinct loopback addresses (distinct ports) and route by the address book, so an
    # explicit host gives the same result + digest as the default — the cross-host seam on one host.
    explicit = deploy(_wordcount_graph(), num_workers=2, host="127.0.0.1")
    default = deploy(_wordcount_graph(), num_workers=2)
    assert _multiset(explicit) == _multiset(default)
    assert explicit.telemetry.structural_digest() == default.telemetry.structural_digest()


# --- telemetry aggregation at the boundary ------------------------------------------------------


def test_cli_key_columns_bridge_matches_serial() -> None:
    # The CLI builder->IR bridge: graph_from_pipeline reads each operator's self-declared key_columns()
    # and replicates instances at parallelism > 1, so a keyed pipeline driven only by --parallelism
    # shuffles by key (never RoundRobin-splits) and equals the serial run. Exercises cli._run's
    # single-process (run_plan) and distributed (deploy) routing through the bridge.
    from nautilus.cli import _run
    from nautilus.pipelines import wordcount

    serial = asyncio.run(run_local_chain(*wordcount()))
    in_process = _run("wordcount", Tier.COUNTERS, 16, workers=1, parallelism=3)
    distributed = _run("wordcount", Tier.COUNTERS, 16, workers=2, parallelism=2)
    assert _multiset(in_process) == _multiset(serial)
    assert _multiset(distributed) == _multiset(serial)


def test_one_process_row_per_worker_and_rows_conserved() -> None:
    serial = asyncio.run(
        run_local_chain(
            InMemorySource([data(line=["the cat the dog the fox a cat a dog"]), EOS_FRAME]),
            [Tokenize("line", "word"), KeyedCount("word")],
        )
    )
    result = deploy(
        _wordcount_graph(),
        num_workers=3,
        telemetry=TelemetryConfig(tier=Tier.COUNTERS, sample_system=True),
    )
    process_nodes = {o.node for o in result.telemetry.operators if o.kind == "process"}
    assert process_nodes == {"worker-0", "worker-1", "worker-2"}  # exactly W process rows
    # Rows are conserved across the worker boundary: the summed output equals the serial run's.
    assert result.telemetry.summary.total_rows_out == serial.telemetry.summary.total_rows_out
