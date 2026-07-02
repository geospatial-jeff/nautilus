"""The parallel chain moves rows correctly single-process, independent of telemetry.

These exercise the compiled parallel path at parallelism > 1 in one process — via ``run_local_chain``
(uniform parallelism over the transforms) and the fluent ``Stream`` DSL (per-stage parallelism). The
golden multiset-equality tests catch a co-partitioning bug: such a bug conserves rows while silently
splitting a key's state, so only comparing the parallel result against the serial P=1 result as a
multiset reveals it. Cross-instance sink interleave is nondeterministic, so every comparison is over a
``collections.Counter`` of result tuples, never row or batch order.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter

from nautilus.core.records import EOS_FRAME
from nautilus.driver.local import run_local_chain
from nautilus.dsl import source
from nautilus.operators import InMemorySource, KeyedCount, MapBatch
from nautilus.testing import TestClock, data, op_counter

# --- helpers -----------------------------------------------------------------------------------


def _wc_frames(batches: list[list[str]]) -> list:
    return [data(word=w) for w in batches] + [EOS_FRAME]


def _counts(result, key_col: str = "word") -> Counter:
    """The (key, count) multiset of a KeyedCount result — the co-partitioning check compares it between
    the parallel and serial runs, which must agree regardless of cross-instance emit order."""
    return Counter((r[key_col], r["count"]) for r in result.to_pylist())


# --- golden multiset equality (the co-partitioning check) --------------------------------------


async def test_keyed_count_matches_serial_multiset() -> None:
    rng = random.Random(7)
    pool = ["the", "cat", "sat", "dog", "ran", "a", "fox", "jumped", "x", "y"]
    for trial in range(12):
        batches = [
            [rng.choice(pool) for _ in range(rng.randint(0, 8))] for _ in range(rng.randint(1, 5))
        ]
        serial = await run_local_chain(
            InMemorySource(_wc_frames(batches)), [KeyedCount("word")], clock=TestClock()
        )
        for q in (1, 2, 3, 5):
            par = await run_local_chain(
                InMemorySource(_wc_frames(batches)),
                [KeyedCount("word")],
                parallelism=q,
                clock=TestClock(),
            )
            assert _counts(par) == _counts(serial), (trial, q)


async def test_keyed_count_across_batches_matches_serial_multiset() -> None:
    # Per-key counts accumulate across several data batches in one stream; a co-partitioning bug would
    # split a key's running count across instances, so the parallel multiset must still equal serial.
    rng = random.Random(13)
    for trial in range(10):
        frames: list = []
        for _ in range(rng.randint(2, 4)):
            m = rng.randint(1, 6)
            frames.append(data(key=[rng.choice(["a", "b", "c", "d"]) for _ in range(m)]))
        frames.append(EOS_FRAME)

        serial = await run_local_chain(
            InMemorySource(list(frames)), [KeyedCount("key")], clock=TestClock()
        )
        for q in (1, 2, 3, 5):
            par = await run_local_chain(
                InMemorySource(list(frames)),
                [KeyedCount("key")],
                parallelism=q,
                clock=TestClock(),
            )
            assert _counts(par, "key") == _counts(serial, "key"), (trial, q)


async def test_deep_mesh_multi_input_matches_serial() -> None:
    # A two-stage mesh: map(P=2) -> count_by(P=3). Each keyed-count instance fans in BOTH map instances,
    # so its mailbox has two inputs — the multi-input fan-in a single-stage mesh never exercises. A key's
    # rows must still co-partition to one instance, so the parallel multiset matches serial.
    rng = random.Random(21)
    for trial in range(6):
        frames: list = []
        for _ in range(rng.randint(3, 5)):
            m = rng.randint(1, 6)
            frames.append(data(key=[rng.choice(["a", "b", "c", "d", "e"]) for _ in range(m)]))
        frames.append(EOS_FRAME)

        serial = await run_local_chain(
            InMemorySource(list(frames)),
            [MapBatch(lambda b: b), KeyedCount("key")],
            clock=TestClock(),
        )
        # Per-stage parallelism (map at 2, keyed count at 3) — the DSL's job, not uniform run_local_chain.
        par = await asyncio.wait_for(
            source(InMemorySource(list(frames)))
            .map(lambda b: b, parallelism=2)
            .count_by("key", parallelism=3)
            .run_async(clock=TestClock()),
            timeout=20,
        )
        assert _counts(par, "key") == _counts(serial, "key"), trial


async def test_null_keys_co_partition_like_serial() -> None:
    # Regression: a null key cell is counted at P=1 (value_counts includes nulls), so the shuffle must
    # route it and the parallel multiset must still match serial — not abort with a TypeError.
    frames = [
        data(word=["a", None, "a", None, "b", None]),
        data(word=["a", "b", None, "c"]),
        EOS_FRAME,
    ]
    serial = await run_local_chain(
        InMemorySource(list(frames)), [KeyedCount("word")], clock=TestClock()
    )
    for q in (2, 3, 5):
        par = await run_local_chain(
            InMemorySource(list(frames)), [KeyedCount("word")], parallelism=q, clock=TestClock()
        )
        assert _counts(par) == _counts(serial), q


async def test_roundrobin_out_of_parallel_stage_conserves_rows() -> None:
    # map(P=2) -> map(Q=2, keyless -> RoundRobin): the RoundRobin edge is fed by TWO upstream instances,
    # each owning its own cursor. Rows must be conserved over that multi-upstream rebalance.
    rng = random.Random(55)
    frames: list = []
    total = 0
    for _ in range(5):
        m = rng.randint(1, 8)
        frames.append(data(k=[rng.randrange(6) for _ in range(m)]))
        total += m
    frames.append(EOS_FRAME)
    res = await run_local_chain(
        InMemorySource(list(frames)),
        [MapBatch(lambda b: b), MapBatch(lambda b: b)],
        parallelism=2,
        clock=TestClock(),
    )
    assert sum(rb.num_rows for rb in res) == total
    assert op_counter(res.telemetry, "op1", "operator.rows_in") == total


# --- skew / empty-partition liveness -----------------------------------------------------------


async def test_single_key_skew_terminates_and_conserves() -> None:
    # Every row shares one key, so the shuffle routes all data to one of the 4 instances; the other
    # three receive only the broadcast EOS and forward it. The run must still terminate and conserve.
    frames = [
        data(key=["a"] * 6),
        data(key=["a", "a"]),
        EOS_FRAME,
    ]
    serial = await run_local_chain(
        InMemorySource(list(frames)), [KeyedCount("key")], clock=TestClock()
    )
    par = await asyncio.wait_for(
        run_local_chain(
            InMemorySource(list(frames)),
            [KeyedCount("key")],
            parallelism=4,
            clock=TestClock(),
        ),
        timeout=10,
    )
    assert _counts(par, "key") == _counts(serial, "key")
    rep = par.telemetry
    assert op_counter(rep, "op0", "operator.rows_in") == 8  # summed over 4 subtasks
    assert op_counter(rep, "op0", "eos.received") == 4  # every instance got its EOS via broadcast
    assert op_counter(rep, "sink", "eos.received") == 4  # the sink fanned in all 4 instances


# --- stateless row conservation ----------------------------------------------------------------


async def test_stateless_map_conserves_rows() -> None:
    rng = random.Random(99)
    for key_columns in (["k"], None):
        for q in (2, 3):
            frames: list = []
            total = 0
            for _ in range(rng.randint(1, 4)):
                m = rng.randint(1, 10)
                frames.append(data(k=[rng.randrange(5) for _ in range(m)]))
                total += m
            frames.append(EOS_FRAME)
            # A keyless op shuffled on an explicit key (or keyless) — the .apply escape hatch's keying.
            res = await (
                source(InMemorySource(list(frames)))
                .apply(MapBatch(lambda b: b), key_columns=key_columns, parallelism=q)
                .run_async(clock=TestClock())
            )
            assert sum(rb.num_rows for rb in res) == total
            rep = res.telemetry
            assert op_counter(rep, "op0", "operator.rows_in") == total
            assert op_counter(rep, "sink", "operator.rows_in") == total


# --- degenerate (all-serial) -------------------------------------------------------------------


async def test_all_serial_degenerates_to_linear() -> None:
    batches = [["the", "cat", "the"], ["dog"]]
    serial = await run_local_chain(
        InMemorySource(_wc_frames(batches)), [KeyedCount("word")], clock=TestClock()
    )
    par = await run_local_chain(
        InMemorySource(_wc_frames(batches)), [KeyedCount("word")], parallelism=1, clock=TestClock()
    )
    assert _counts(par) == _counts(serial)


async def test_zero_stage_source_to_sink() -> None:
    par = await run_local_chain(
        InMemorySource([data(word=["a", "b"]), EOS_FRAME]), [], clock=TestClock()
    )
    assert sum(rb.num_rows for rb in par) == 2
