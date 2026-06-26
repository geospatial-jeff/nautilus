"""Stage 1.5b: the parallel mesh moves rows correctly, independent of telemetry.

The golden multiset-equality tests are what catch a co-partitioning bug: such a bug conserves rows
while silently splitting a key's state, so only comparing the parallel result against the serial P=1
result as a multiset reveals it. Cross-instance sink interleave is nondeterministic, so every
comparison is over a ``collections.Counter`` of result tuples, never row or batch order.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter

import pytest

from nautilus.core.records import EOS_FRAME, Batch
from nautilus.operators import InMemorySource, KeyedCount, KeyedTumblingSum, MapBatch
from nautilus.runtime.channel import InProcChannel
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.parallel import (
    Stage,
    _check_fanout_partitioner,
    _collect_parallel,
    _partitioner_for,
    run_parallel_chain,
)
from nautilus.runtime.partition import Forward, HashPartitioner, RoundRobin
from nautilus.telemetry.recorder import NULL_RECORDER
from nautilus.testing import TestClock, batch, data, wm
from nautilus.windows import TumblingEventTimeWindows

# --- helpers -----------------------------------------------------------------------------------


def _wc_frames(batches: list[list[str]]) -> list:
    return [data(word=w) for w in batches] + [EOS_FRAME]


def _wc_counts(result) -> Counter:
    return Counter((r["word"], r["count"]) for r in result.to_pylist())


def _ts_counts(result) -> Counter:
    return Counter(
        (r["key"], r["window_start"], r["window_end"], r["sum"]) for r in result.to_pylist()
    )


def _op_counter(rep, op_id: str, name: str) -> int:
    return sum(
        p.value
        for o in rep.operators
        if o.operator_id == op_id
        for p in o.counters
        if p.name == name
    )


# --- the wiring decision -----------------------------------------------------------------------


def test_partitioner_selection() -> None:
    assert isinstance(_partitioner_for(1, None), Forward)
    assert isinstance(_partitioner_for(1, ["k"]), Forward)  # Q==1 is Forward even with keys
    assert isinstance(_partitioner_for(3, ["k"]), HashPartitioner)
    assert isinstance(_partitioner_for(3, None), RoundRobin)  # parallel + no key → rebalance


def test_forward_misuse_raises_at_wiring() -> None:
    with pytest.raises(ValueError, match="Forward"):
        _check_fanout_partitioner(Forward(), 3, "op0")
    _check_fanout_partitioner(HashPartitioner(["k"]), 3, "op0")  # a real shuffle is fine
    _check_fanout_partitioner(Forward(), 1, "sink")  # Q==1 Forward is fine


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
            par = await run_parallel_chain(
                InMemorySource(_wc_frames(batches)),
                [Stage(lambda: KeyedCount("word"), q, ["word"])],
                clock=TestClock(),
            )
            assert _wc_counts(par) == _wc_counts(serial), (trial, q)


async def test_keyed_tumbling_sum_matches_serial_multiset() -> None:
    rng = random.Random(13)
    for trial in range(10):
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

        serial = await run_local_chain(
            InMemorySource(list(frames)),
            [KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))],
            clock=TestClock(),
        )
        for q in (1, 2, 3, 5):
            par = await run_parallel_chain(
                InMemorySource(list(frames)),
                [
                    Stage(
                        lambda: KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10)),
                        q,
                        ["key"],
                    )
                ],
                clock=TestClock(),
            )
            assert _ts_counts(par) == _ts_counts(serial), (trial, q)


async def test_deep_mesh_multi_input_matches_serial() -> None:
    # A two-stage mesh: map(P=2) → keyedSum(P=3). Each keyed-sum instance fans in BOTH map instances,
    # so its mailbox has two inputs and its watermark is the min over two upstreams — the multi-input
    # WatermarkTracker path a single-stage mesh never exercises. Results must still match serial.
    rng = random.Random(21)
    for trial in range(6):
        frames: list = []
        clock_t = 0
        for _ in range(rng.randint(3, 5)):
            m = rng.randint(1, 6)
            frames.append(
                data(
                    key=[rng.choice(["a", "b", "c", "d", "e"]) for _ in range(m)],
                    val=[rng.randint(1, 9) for _ in range(m)],
                    ts=[clock_t + rng.randint(0, 8) for _ in range(m)],
                )
            )
            clock_t += 10
            frames.append(wm(clock_t))
        frames.append(EOS_FRAME)

        serial = await run_local_chain(
            InMemorySource(list(frames)),
            [
                MapBatch(lambda b: b),
                KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10)),
            ],
            clock=TestClock(),
        )
        par = await asyncio.wait_for(
            run_parallel_chain(
                InMemorySource(list(frames)),
                [
                    Stage(lambda: MapBatch(lambda b: b), 2),
                    Stage(
                        lambda: KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10)),
                        3,
                        ["key"],
                    ),
                ],
                clock=TestClock(),
            ),
            timeout=20,
        )
        assert _ts_counts(par) == _ts_counts(serial), trial


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
        par = await run_parallel_chain(
            InMemorySource(list(frames)),
            [Stage(lambda: KeyedCount("word"), q, ["word"])],
            clock=TestClock(),
        )
        assert _wc_counts(par) == _wc_counts(serial), q


async def test_roundrobin_out_of_parallel_stage_conserves_rows() -> None:
    # map(P=2) → map(Q=2, keyless → RoundRobin): the RoundRobin edge is fed by TWO upstream instances,
    # each owning its own cursor. Rows must be conserved over that multi-upstream rebalance.
    rng = random.Random(55)
    frames: list = []
    total = 0
    for _ in range(5):
        m = rng.randint(1, 8)
        frames.append(data(k=[rng.randrange(6) for _ in range(m)]))
        total += m
    frames.append(EOS_FRAME)
    res = await run_parallel_chain(
        InMemorySource(list(frames)),
        [Stage(lambda: MapBatch(lambda b: b), 2), Stage(lambda: MapBatch(lambda b: b), 2)],
        clock=TestClock(),
    )
    assert sum(rb.num_rows for rb in res) == total
    assert _op_counter(res.telemetry, "op1", "operator.rows_in") == total


# --- skew / empty-partition liveness -----------------------------------------------------------


async def test_single_key_skew_terminates_and_conserves() -> None:
    # Every row shares one key, so the shuffle routes all data to one of the 4 instances; the other
    # three receive only the broadcast watermarks + EOS, advance event time, and forward EOS.
    frames = [
        data(key=["a"] * 6, val=[1, 2, 3, 4, 5, 6], ts=[1, 2, 3, 4, 5, 6]),
        wm(10),
        data(key=["a", "a"], val=[10, 20], ts=[12, 13]),
        wm(20),
        EOS_FRAME,
    ]
    serial = await run_local_chain(
        InMemorySource(list(frames)),
        [KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))],
        clock=TestClock(),
    )
    par = await asyncio.wait_for(
        run_parallel_chain(
            InMemorySource(list(frames)),
            [
                Stage(
                    lambda: KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10)),
                    4,
                    ["key"],
                )
            ],
            clock=TestClock(),
        ),
        timeout=10,
    )
    assert _ts_counts(par) == _ts_counts(serial)
    rep = par.telemetry
    assert _op_counter(rep, "op0", "operator.rows_in") == 8  # summed over 4 subtasks
    assert _op_counter(rep, "op0", "eos.received") == 4  # every instance got its EOS via broadcast
    assert _op_counter(rep, "sink", "eos.received") == 4  # the sink fanned in all 4 instances


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
            res = await run_parallel_chain(
                InMemorySource(list(frames)),
                [Stage(lambda: MapBatch(lambda b: b), q, key_columns)],
                clock=TestClock(),
            )
            assert sum(rb.num_rows for rb in res) == total
            rep = res.telemetry
            assert _op_counter(rep, "op0", "operator.rows_in") == total
            assert _op_counter(rep, "sink", "operator.rows_in") == total


# --- EOS-after-all-P (the multi-input sink) ----------------------------------------------------


async def test_collect_parallel_waits_for_all_eos() -> None:
    a, b = InProcChannel(8), InProcChannel(8)
    mb = Mailbox([a, b])
    out: list = []
    task = asyncio.create_task(_collect_parallel(mb, out, NULL_RECORDER))
    await a.send(Batch(batch(x=[1])))
    await a.send(EOS_FRAME)
    # one input closed, one still open → the sink must not terminate
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
    await b.send(EOS_FRAME)
    await asyncio.wait_for(task, timeout=2)
    assert sum(rb.num_rows for rb in out) == 1


# --- degenerate (all-serial) -------------------------------------------------------------------


async def test_all_serial_degenerates_to_linear() -> None:
    batches = [["the", "cat", "the"], ["dog"]]
    serial = await run_local_chain(
        InMemorySource(_wc_frames(batches)), [KeyedCount("word")], clock=TestClock()
    )
    par = await run_parallel_chain(
        InMemorySource(_wc_frames(batches)),
        [Stage(lambda: KeyedCount("word"), 1, ["word"])],
        clock=TestClock(),
    )
    assert _wc_counts(par) == _wc_counts(serial)


async def test_zero_stage_source_to_sink() -> None:
    par = await run_parallel_chain(
        InMemorySource([data(word=["a", "b"]), EOS_FRAME]), [], clock=TestClock()
    )
    assert sum(rb.num_rows for rb in par) == 2
