"""The fluent ``Stream`` DSL: it builds the same graph the explicit API does, and runs it.

These pin that the DSL is a faithful builder — the graph it produces compiles to the same structural
digest as the hand-built equivalent — and that its combinators (map/tokenize/count_by/join/apply) and
its terminal (run / collect / run(workers=)) behave, including the join's co-partitioning and self-join
guard.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from nautilus.api import linear_graph, one_input
from nautilus.driver.run import run_plan
from nautilus.dsl import Stream, source
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.testing import EOS_FRAME, data, multiset


def _wordcount_stream() -> Stream:
    src = InMemorySource([data(line=["the cat sat", "the dog ran the"]), EOS_FRAME])
    return source(src).tokenize("line", "word").count_by("word")


def test_dsl_builds_the_expected_dag() -> None:
    graph = _wordcount_stream().to_graph()
    assert [(v.id, v.kind) for v in graph.vertices] == [
        ("v0", "source"),
        ("v1", "one_input"),
        ("v2", "one_input"),
    ]
    assert [(e.src, e.dst, e.dst_input_port, e.key_columns) for e in graph.edges] == [
        ("v0", "v1", 0, None),  # source -> tokenize, keyless
        ("v1", "v2", 0, ("word",)),  # tokenize -> count, shuffled on word
    ]


def test_dsl_compiles_to_the_same_digest_as_the_explicit_api() -> None:
    src = InMemorySource([data(line=["the cat sat", "the dog ran the"]), EOS_FRAME])
    explicit = linear_graph(
        lambda: src,
        [
            one_input("a", lambda: Tokenize("line", "word")),
            one_input("b", lambda: KeyedCount("word"), key_columns=("word",)),
        ],
    )
    from_dsl = asyncio.run(_wordcount_stream().run_async())
    from_api = asyncio.run(run_plan(explicit))
    assert from_dsl.telemetry.structural_digest() == from_api.telemetry.structural_digest()
    assert multiset(from_dsl) == multiset(from_api)


def test_dsl_wordcount_runs() -> None:
    assert Counter((r["word"], r["count"]) for r in _wordcount_stream().collect()) == Counter(
        {("the", 3): 1, ("cat", 1): 1, ("sat", 1): 1, ("dog", 1): 1, ("ran", 1): 1}
    )


def test_dsl_parallelism_override_matches_serial() -> None:
    serial = multiset(_wordcount_stream().run())
    parallel = multiset(_wordcount_stream().run(parallelism=3))
    assert parallel == serial


def _left() -> Stream:
    return source(InMemorySource([data(id=[1, 1, 2], lval=["a", "b", "c"]), EOS_FRAME]))


def _right() -> Stream:
    return source(InMemorySource([data(id=[1, 3], rval=[10, 30]), EOS_FRAME]))


def test_dsl_join_builds_a_two_input_dag_and_runs() -> None:
    joined = _left().join(_right(), on="id")
    graph = joined.to_graph()
    assert [(v.id, v.kind) for v in graph.vertices] == [
        ("v0", "source"),  # left
        ("v1", "source"),  # right, relabelled past the left's ids
        ("v2", "two_input"),
    ]
    ports = sorted((e.dst_input_port, e.src, e.key_columns) for e in graph.edges)
    assert ports == [(0, "v0", ("id",)), (1, "v1", ("id",))]
    assert Counter((r["id"], r["lval"], r["rval"]) for r in joined.collect()) == Counter(
        {(1, "a", 10): 1, (1, "b", 10): 1}
    )


def test_dsl_join_co_partitions_when_parallel() -> None:
    joined = _left().join(_right(), on="id")
    assert multiset(joined.run(parallelism=2)) == multiset(joined.run())


def test_dsl_rejects_a_self_join() -> None:
    s = _left()
    with pytest.raises(ValueError, match="cannot be joined to itself"):
        s.join(s, on="id")


def test_dsl_join_rejects_unequal_key_arity() -> None:
    with pytest.raises(ValueError, match="same number of columns"):
        _left().join(_right(), left_on=["id", "lval"], right_on=["id"])


def test_dsl_apply_keys_by_the_operators_own_declaration() -> None:
    # .apply with a keyed operator picks up its key_columns() so the edge is a keyed shuffle.
    src = InMemorySource([data(word=["a", "a", "b"]), EOS_FRAME])
    graph = source(src).apply(KeyedCount("word")).to_graph()
    assert graph.edges[0].key_columns == ("word",)


def test_dsl_apply_explicit_key_overrides_the_operators_declaration() -> None:
    # An explicit key_columns on .apply wins over the operator's own — the deliberate trust-the-caller
    # behavior the DSL kept when the Stage path (which raised on disagreement) was retired.
    src = InMemorySource([data(word=["a", "a", "b"]), EOS_FRAME])
    graph = source(src).apply(KeyedCount("word"), key_columns="other").to_graph()
    assert graph.edges[0].key_columns == ("other",)  # the override, not KeyedCount's ("word",)


async def test_dsl_join_of_joins_relabels_and_runs() -> None:
    # a.join(b).join(c): the second join relabels an operand that is itself a multi-vertex joined stream,
    # so this exercises the id-remap on a non-trivial subgraph and a three-source DAG end-to-end.
    a = source(InMemorySource([data(id=[1, 2], av=["a1", "a2"]), EOS_FRAME]))
    b = source(InMemorySource([data(id=[1, 2], bv=["b1", "b2"]), EOS_FRAME]))
    c = source(InMemorySource([data(id=[1], cv=["c1"]), EOS_FRAME]))
    joined = a.join(b, on="id").join(c, on="id")
    assert (
        len({v.id for v in joined.to_graph().vertices}) == 5
    )  # 3 sources + 2 joins, all unique ids
    rows = await joined.run_async()
    # a⋈b on id -> (1,a1,b1),(2,a2,b2); ⋈c (only id 1) -> (1,a1,b1,c1)
    assert Counter((r["id"], r["av"], r["bv"], r["cv"]) for r in rows.to_pylist()) == Counter(
        {(1, "a1", "b1", "c1"): 1}
    )


def test_dsl_distributed_run_matches_single_process() -> None:
    joined = _left().join(_right(), on="id")
    distributed = joined.run(workers=2, parallelism=2)
    serial = joined.run()
    assert multiset(distributed) == multiset(serial)
