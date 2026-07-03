"""Stage 2e: scaling deploy to multiple workers — correctness, placement invariance, capping, telemetry.

Distributed runs must equal the serial run by multiset across any worker count and parallelism, and the
structural digest must be placement-invariant (the same plan routes the same way no matter how many
workers host it). Hardware telemetry is attributed per worker (exactly one process row each). These
tests spawn processes, so they use only importable (module-level) operators.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from nautilus.cluster import deploy
from nautilus.core.operator import OneInputOperator
from nautilus.core.records import EOS_FRAME
from nautilus.driver.local import run_local_chain
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.telemetry import TelemetryConfig, Tier
from nautilus.testing import data, multiset, staged_graph

_Spec = tuple[OneInputOperator, int, "tuple[str, ...] | None"]

_WORDS = ["the", "cat", "sat", "dog", "ran", "a", "fox", "x", "y", "z"]


def _wordcount_frames(rng: random.Random) -> list:
    lines = [
        " ".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 5)))
        for _ in range(rng.randint(1, 4))
    ]
    return [data(line=[ln]) for ln in lines] + [EOS_FRAME]


def _keyed_frames(rng: random.Random) -> list:
    frames: list = []
    for _ in range(rng.randint(2, 4)):
        m = rng.randint(1, 6)
        frames.append(data(key=[rng.choice(["a", "b", "c", "d"]) for _ in range(m)]))
    frames.append(EOS_FRAME)
    return frames


def _keyed_spec(p: int) -> _Spec:
    return (KeyedCount("key"), p, ("key",))


# --- correctness across random worker-count by parallelism --------------------------------------


@pytest.mark.filterwarnings(
    "ignore:requested"
)  # random combos over-provision on purpose; capping is fine
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
            graph = staged_graph(
                InMemorySource(frames),
                [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), parallelism, ("word",))],
            )
        else:
            frames = _keyed_frames(rng)
            serial = asyncio.run(run_local_chain(InMemorySource(frames), [KeyedCount("key")]))
            graph = staged_graph(InMemorySource(frames), [_keyed_spec(parallelism)])
        result = deploy(graph, num_workers=workers)
        assert multiset(result) == multiset(serial), (trial, workers, parallelism)


def test_single_key_skew_terminates_across_workers() -> None:
    # Every row shares one key: the shuffle sends all data to one instance; the others receive only the
    # broadcast EOS over their sockets and forward it. It must still terminate.
    frames = [
        data(key=["a"] * 8),
        EOS_FRAME,
    ]
    serial = asyncio.run(run_local_chain(InMemorySource(frames), [KeyedCount("key")]))
    result = deploy(staged_graph(InMemorySource(frames), [_keyed_spec(4)]), num_workers=4)
    assert multiset(result) == multiset(serial)


# --- placement invariance + capping -------------------------------------------------------------


def _wordcount_graph():
    return staged_graph(
        InMemorySource([data(line=["the cat the dog the fox a cat a dog"]), EOS_FRAME]),
        [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), 3, ("word",))],
    )


def test_structural_digest_is_placement_invariant_across_worker_counts() -> None:
    # The same plan at fixed parallelism routes the same way regardless of how many workers host it, so
    # the structural digest is identical across 1, 2, and 3 workers
    # (different addresses, same routing).
    digests = {
        deploy(_wordcount_graph(), num_workers=w).telemetry.structural_digest() for w in (1, 2, 3)
    }
    assert len(digests) == 1, digests


def test_workers_capped_at_max_parallelism() -> None:
    # max parallelism here is 3 (KeyedCount); asking for 8 workers caps to 3 — no empty worker, no hang —
    # and says so loudly (a UserWarning), not through a hidden INFO log the user never sees.
    with pytest.warns(UserWarning, match="requested 8 workers"):
        result = deploy(_wordcount_graph(), num_workers=8)
    nodes = {o.node for o in result.telemetry.operators if o.kind != "process"}
    assert nodes == {"worker-0", "worker-1", "worker-2"}


def test_host_is_a_parameter_and_routing_is_address_independent() -> None:
    # Workers bind distinct loopback addresses (distinct ports) and route by the address book, so an
    # explicit host gives the same result + digest as the default — the cross-host seam on one host.
    explicit = deploy(_wordcount_graph(), num_workers=2, host="127.0.0.1")
    default = deploy(_wordcount_graph(), num_workers=2)
    assert multiset(explicit) == multiset(default)
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
    assert multiset(in_process) == multiset(serial)
    assert multiset(distributed) == multiset(serial)


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
    # exactly one process row per worker
    assert process_nodes == {"worker-0", "worker-1", "worker-2"}
    # Rows are conserved across the worker boundary: the summed output equals the serial run's.
    assert result.telemetry.summary.total_rows_out == serial.telemetry.summary.total_rows_out
