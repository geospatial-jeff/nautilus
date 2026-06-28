"""HashJoin: the inner symmetric-hash equi-join.

Unit tests drive the operator directly (open / process_left / process_right / close) to pin its
semantics — completeness, order-independence, the column-collision and key-arity guards, and buffer
clearing. Integration tests run a two-source join graph through the real compiler + executor, and
across worker processes, to prove the keyed shuffle co-partitions both sides so the parallel and
distributed results match the serial one.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pyarrow as pa
import pytest

from nautilus.api import LogicalEdge, LogicalGraph, source, two_input
from nautilus.cluster import deploy
from nautilus.core.operator import ListCollector, OperatorContext
from nautilus.core.records import WATERMARK_MAX
from nautilus.driver.run import run_plan
from nautilus.operators import HashJoin, InMemorySource
from nautilus.testing import EOS_FRAME, batch, data, multiset


def _drive(join: HashJoin, steps: list[tuple[str, pa.RecordBatch]]) -> list[pa.RecordBatch]:
    """Open the join, feed a sequence of (side, batch) steps, flush at EOS, close, return emitted batches."""
    coll = ListCollector()
    join.open(OperatorContext("j"))
    for side, b in steps:
        (join.process_left if side == "L" else join.process_right)(b, coll)
    join.on_watermark(
        WATERMARK_MAX, coll
    )  # a no-op for the inner join; here to assert it stays one
    join.close()
    return coll.drain()


def _triples(batches: list[pa.RecordBatch]) -> Counter[tuple[object, object, object]]:
    rows = [r for b in batches for r in b.to_pylist()]
    return Counter((r["id"], r["lval"], r["rval"]) for r in rows)


def test_join_is_complete_and_order_independent() -> None:
    left = batch(id=[1, 1, 2], lval=["a", "b", "c"])
    right = batch(id=[1, 3], rval=[10, 30])
    expected = Counter({(1, "a", 10): 1, (1, "b", 10): 1})  # id 2 has no right, id 3 no left
    assert _triples(_drive(HashJoin("id"), [("L", left), ("R", right)])) == expected
    assert _triples(_drive(HashJoin("id"), [("R", right), ("L", left)])) == expected
    # split across batches and interleaved — each pair still emitted exactly once, when the later arrives
    interleaved = [
        ("L", batch(id=[1], lval=["a"])),
        ("R", batch(id=[1], rval=[10])),
        ("L", batch(id=[1, 2], lval=["b", "c"])),
        ("R", batch(id=[3], rval=[30])),
    ]
    assert _triples(_drive(HashJoin("id"), interleaved)) == expected


def test_join_emits_cross_product_within_a_key() -> None:
    out = _drive(
        HashJoin("id"),
        [("L", batch(id=[5, 5], lval=["a", "b"])), ("R", batch(id=[5, 5], rval=[1, 2]))],
    )
    assert _triples(out) == Counter(
        {(5, "a", 1): 1, (5, "a", 2): 1, (5, "b", 1): 1, (5, "b", 2): 1}
    )


def test_join_emits_nothing_without_matches() -> None:
    out = _drive(
        HashJoin("id"), [("L", batch(id=[1], lval=["a"])), ("R", batch(id=[2], rval=[20]))]
    )
    assert out == []


def test_join_joins_on_differently_named_columns() -> None:
    # left.lid == right.rid; the key appears once (from the left, named lid), right's non-key col carried.
    out = _drive(
        HashJoin("lid", "rid"),
        [("L", batch(lid=[7], lval=["a"])), ("R", batch(rid=[7], rval=[70]))],
    )
    rows = [r for b in out for r in b.to_pylist()]
    assert rows == [{"lid": 7, "lval": "a", "rval": 70}]


def test_join_rejects_colliding_output_column() -> None:
    join = HashJoin("id")
    join.open(OperatorContext("j"))
    join.process_left(batch(id=[1], val=["x"]), ListCollector())
    with pytest.raises(ValueError, match="collision"):  # right's non-key 'val' collides with left's
        join.process_right(batch(id=[1], val=[9]), ListCollector())


def test_join_rejects_unequal_key_arity() -> None:
    with pytest.raises(ValueError, match="same number of columns"):
        HashJoin(["a", "b"], ["c"])


def test_join_clears_buffers_on_close() -> None:
    join = HashJoin("id")
    join.open(OperatorContext("j"))
    join.process_left(batch(id=[1], lval=["a"]), ListCollector())
    join.process_right(batch(id=[1], rval=[10]), ListCollector())
    assert join._left and join._right  # buffered while running
    join.close()
    assert not join._left and not join._right


# --- through the engine: co-partitioning makes the parallel and distributed results match serial -----

_LEFT = [data(id=[1, 1, 2, 3], lval=["a", "b", "c", "d"]), EOS_FRAME]
_RIGHT = [data(id=[1, 2, 2, 4], rval=[10, 20, 21, 40]), EOS_FRAME]
_EXPECTED = Counter(  # id 1: {a,b}x{10}; id 2: {c}x{20,21}; id 3,4 unmatched
    {(1, "a", 10): 1, (1, "b", 10): 1, (2, "c", 20): 1, (2, "c", 21): 1}
)


def _join_graph(parallelism: int) -> LogicalGraph:
    return LogicalGraph(
        vertices=(
            source("L", lambda: InMemorySource(list(_LEFT))),
            source("R", lambda: InMemorySource(list(_RIGHT))),
            two_input("j", lambda: HashJoin("id"), parallelism=parallelism),
        ),
        edges=(LogicalEdge("L", "j", 0, ("id",)), LogicalEdge("R", "j", 1, ("id",))),
    )


async def test_join_runs_through_the_engine() -> None:
    result = await run_plan(_join_graph(1))
    assert Counter((r["id"], r["lval"], r["rval"]) for r in result.to_pylist()) == _EXPECTED


async def test_parallel_join_co_partitions_both_sides() -> None:
    # At P=2 each input is a keyed shuffle on the join value; a key's left and right rows must land on the
    # same instance, so the multiset of matches is identical to the serial run.
    serial = multiset(await run_plan(_join_graph(1)))
    parallel = multiset(await run_plan(_join_graph(2)))
    assert parallel == serial


def test_join_distributed_matches_single_process() -> None:
    # The same plan deployed across two worker processes: a join instance can sit on a different worker
    # from the sources, so this exercises the join over the cross-worker keyed shuffle.
    graph = _join_graph(2)
    serial = asyncio.run(run_plan(graph))
    distributed = deploy(graph, num_workers=2)
    assert multiset(distributed) == multiset(serial)
    assert distributed.telemetry.structural_digest() == serial.telemetry.structural_digest()
