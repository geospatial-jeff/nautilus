"""The compiled ``api -> compile -> execute`` path is equivalent to the serial run.

A compiled parallel run must produce the same output **multiset** as the serial single-process run.
Cross-instance sink interleave is nondeterministic, so results are compared as a ``collections.Counter``
of rows, never by order. The stronger claim — that the **structural_digest** is identical no matter how
the plan is placed across workers — is proven compiled-vs-compiled in tests/test_cluster_scale.py.

The digest still anchors serialization here: a cloudpickled-and-reloaded plan must execute to the
identical digest, because a plan that lost or reshaped anything would route differently.
"""

from __future__ import annotations

import random

import cloudpickle

from nautilus.compile import compile_graph
from nautilus.core.operator import OneInputOperator
from nautilus.core.records import EOS_FRAME
from nautilus.driver.local import run_local_chain
from nautilus.driver.pipeline import graph_from_pipeline
from nautilus.driver.result import RunResult
from nautilus.driver.run import run_compiled, run_plan
from nautilus.operators import InMemorySource, KeyedCount, MapBatch, Tokenize
from nautilus.pipelines import wordcount
from nautilus.testing import TestClock, data, multiset, staged_graph

_WORDS = ["the", "cat", "sat", "dog", "ran", "a", "fox", "jumped", "x", "y"]

# (operator, parallelism, key_columns) specs for staged_graph.
_Specs = list[tuple[OneInputOperator, int, "tuple[str, ...] | None"]]


def _digest(result: RunResult) -> str:
    return result.telemetry.structural_digest()


# --- random linear graphs: compiled (parallel) == serial by multiset ----------------------------
# The oracle is now compiled-parallel vs the serial single-process run; placement-invariance of the
# digest (compiled-vs-compiled across worker counts) is proven in tests/test_cluster_scale.py.


def _wordcount_chain(rng: random.Random) -> tuple[list, _Specs, list[OneInputOperator]]:
    lines = [
        " ".join(rng.choice(_WORDS) for _ in range(rng.randint(0, 4)))
        for _ in range(rng.randint(1, 4))
    ]
    frames = [data(line=[ln]) for ln in lines] + [EOS_FRAME]
    specs: _Specs = [
        (Tokenize("line", "word"), rng.choice([1, 2, 3]), None),  # keyless -> RoundRobin
        (KeyedCount("word"), rng.choice([1, 2, 3, 5]), ("word",)),  # keyed shuffle
    ]
    serial = [Tokenize("line", "word"), KeyedCount("word")]
    return frames, specs, serial


def _keyed_chain(rng: random.Random) -> tuple[list, _Specs, list[OneInputOperator]]:
    # A keyless identity map (MapBatch) feeding a keyed aggregate (KeyedCount) — a different edge shape
    # than _wordcount_chain's fan-out Tokenize, so the two builders cover distinct keyless partitioners.
    frames: list = []
    for _ in range(rng.randint(2, 4)):
        m = rng.randint(1, 6)
        frames.append(data(key=[rng.choice(["a", "b", "c", "d"]) for _ in range(m)]))
    frames.append(EOS_FRAME)
    specs: _Specs = [
        (MapBatch(lambda b: b), rng.choice([1, 2]), None),  # keyless rebalance/forward
        (KeyedCount("key"), rng.choice([1, 2, 3]), ("key",)),  # keyed shuffle
    ]
    serial = [MapBatch(lambda b: b), KeyedCount("key")]
    return frames, specs, serial


async def test_compiled_parallel_matches_serial_over_random_linear_graphs() -> None:
    rng = random.Random(2024)
    for trial in range(24):
        builder = rng.choice([_wordcount_chain, _keyed_chain])
        frames, specs, serial_transforms = builder(rng)

        serial = await run_local_chain(
            InMemorySource(list(frames)), serial_transforms, clock=TestClock()
        )
        compiled = await run_plan(
            staged_graph(InMemorySource(list(frames)), specs), clock=TestClock()
        )

        assert multiset(serial) == multiset(compiled), (trial, builder.__name__)


# --- cloudpickle round-trip executes equivalently -----------------------------------------------


async def test_cloudpickle_roundtrip_executes_equivalently() -> None:
    plan = compile_graph(
        staged_graph(
            InMemorySource([data(line=["the quick fox"]), data(line=["the lazy fox"]), EOS_FRAME]),
            [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), 3, ("word",))],
        )
    )
    restored = cloudpickle.loads(cloudpickle.dumps(plan))

    original = await run_compiled(plan, clock=TestClock())
    roundtripped = await run_compiled(restored, clock=TestClock())

    assert multiset(original) == multiset(roundtripped)
    assert _digest(original) == _digest(roundtripped)

    # Structural equivalence of factories — each rebuilds an operator of the same class (object == is
    # impossible for lambda factories, so identity of behavior is checked through the run above).
    def classes(p):
        return [type(o.factory()).__name__ for o in p.operators if o.factory is not None]

    assert classes(restored) == classes(plan) == ["InMemorySource", "Tokenize", "KeyedCount"]


# --- linear_graph migrates both existing shapes -------------------------------------------------


async def test_ops_bridge_matches_run_local_chain() -> None:
    # The run_local_chain shape (source + operator instances, all at parallelism 1) compiles to the
    # serial result.
    frames = [data(line=["the cat the"]), data(line=["dog cat"]), EOS_FRAME]
    transforms = [Tokenize("line", "word"), KeyedCount("word")]

    serial = await run_local_chain(InMemorySource(list(frames)), transforms, clock=TestClock())
    compiled = await run_plan(
        graph_from_pipeline(
            InMemorySource(list(frames)), [Tokenize("line", "word"), KeyedCount("word")], 1
        ),
        clock=TestClock(),
    )
    assert multiset(serial) == multiset(compiled)
    assert _digest(serial) == _digest(compiled)


async def test_real_pipelines_example_round_trips() -> None:
    # A real pipelines.py builder (source + instances) compiles, cloudpickle round-trips, and runs.
    source, transforms = wordcount()
    plan = compile_graph(graph_from_pipeline(source, transforms, 1))
    restored = cloudpickle.loads(cloudpickle.dumps(plan))
    result = await run_compiled(restored, clock=TestClock())
    assert sum(rb.num_rows for rb in result) > 0
    assert {"word", "count"} <= set(result.to_table().column_names)


# --- key groups: at least as many groups as instances preserves the result;
# an equal count matches the direct-hash digest ---


async def test_key_groups_preserve_results_for_g_ge_q() -> None:
    # Routing keyed edges through key groups (for at least as many groups as instances) must not
    # change the result: every key still lands on exactly one instance, so the output multiset
    # equals the serial run. The key-group count is varied over multiples and non-multiples of the
    # parallelism (5 and 7 are not multiples of 3).
    frames = [
        data(line=["the cat sat the dog ran the fox"]),
        data(line=["a fox and a cat and a dog and a the"]),
        EOS_FRAME,
    ]
    serial = await run_local_chain(
        InMemorySource(list(frames)),
        [Tokenize("line", "word"), KeyedCount("word")],
        clock=TestClock(),
    )
    expected = multiset(serial)
    q = 3
    for g in (3, 4, 5, 7, 12):  # all at least the parallelism, mixing multiples and non-multiples
        graph = staged_graph(
            InMemorySource(list(frames)),
            [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), q, ("word",))],
        )
        result = await run_plan(graph, key_groups=g, clock=TestClock())
        assert multiset(result) == expected, g


async def test_digest_matches_direct_hash_exactly_when_q_divides_g() -> None:
    # An instance is the key's hash modulo the key-group count, then modulo the parallelism, which
    # equals the direct hash (the key's hash modulo the parallelism) exactly when the parallelism
    # divides the key-group count. So the structural digest equals the default run (key-group count
    # equal to the parallelism) at every multiple of the parallelism and legitimately differs at a
    # non-multiple — while the result multiset is preserved either way. This pins both halves of the
    # divisibility boundary, so a different (still co-partitioning) group table can't slip through.
    frames = [data(line=["the cat the dog the fox a cat a dog ran sat jumped"]), EOS_FRAME]
    q = 3

    def graph():
        return staged_graph(
            InMemorySource(list(frames)),
            [(Tokenize("line", "word"), 1, None), (KeyedCount("word"), q, ("word",))],
        )

    # key-group count defaults to the parallelism -> identity table
    default = await run_plan(graph(), clock=TestClock())
    for g, q_divides_g in [(3, True), (6, True), (12, True), (4, False), (5, False), (7, False)]:
        run = await run_plan(graph(), key_groups=g, clock=TestClock())
        assert multiset(run) == multiset(default), g  # co-partitioning: result always preserved
        # digest matches iff the parallelism divides the key-group count
        assert (_digest(run) == _digest(default)) is q_divides_g, g
