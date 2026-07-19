"""The fluent ``Stream`` DSL: it builds the same graph the explicit API does, and runs it.

These pin that the DSL is a faithful builder — the graph it produces compiles to the same structural
digest as the hand-built equivalent — and that its combinators (map/tokenize/count_by/join/apply) and
its terminal (run / collect / run(workers=)) behave, including the join's co-partitioning and self-join
guard.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pyarrow.compute as pc
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


@pytest.mark.parametrize(
    "how, extra",
    [
        ("inner", {}),
        ("left", {(2, "c", None): 1}),  # id 2 is left-only
        ("right", {(3, None, 30): 1}),  # id 3 is right-only
        ("outer", {(2, "c", None): 1, (3, None, 30): 1}),
    ],
)
def test_dsl_outer_join_keeps_unmatched_rows(how: str, extra: dict) -> None:
    joined = _left().join(_right(), on="id", how=how)
    expected = Counter({(1, "a", 10): 1, (1, "b", 10): 1, **extra})
    assert Counter((r["id"], r["lval"], r["rval"]) for r in joined.collect()) == expected


def test_dsl_rejects_outer_join_above_parallelism_one() -> None:
    # An outer join is parallelism-1 only (see HashJoin); the DSL rejects a wider one at build time rather
    # than letting it silently drop unmatched rows at run time.
    with pytest.raises(ValueError, match="parallelism 1 only"):
        _left().join(_right(), on="id", how="outer", parallelism=2)


def test_dsl_rejects_a_self_join() -> None:
    s = _left()
    with pytest.raises(ValueError, match="cannot be joined to itself"):
        s.join(s, on="id")


def test_dsl_join_rejects_unequal_key_arity() -> None:
    with pytest.raises(ValueError, match="same number of columns"):
        _left().join(_right(), left_on=["id", "lval"], right_on=["id"])


def test_dsl_rejects_empty_key_names() -> None:
    # an empty/blank key name builds an unusable graph; reject it up front, not at run time
    with pytest.raises(ValueError, match="non-empty column names"):
        _left().join(_right(), on="")
    with pytest.raises(ValueError, match="non-empty column names"):
        source(InMemorySource([data(word=["a"]), EOS_FRAME])).count_by("  ")


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


# --- column reshaping (.select / .drop / .rename / .with_column) --------------------------------


def _reshape_src() -> Stream:
    # three columns so each op has something to keep and something to change
    return source(InMemorySource([data(id=[1, 2], lval=["a", "b"], scratch=[9, 8]), EOS_FRAME]))


def test_dsl_select_keeps_only_named_columns_in_order() -> None:
    rows = _reshape_src().select("lval", "id").collect()
    assert all("scratch" not in r for r in rows)  # unnamed column dropped
    assert list(rows[0].keys()) == ["lval", "id"]  # kept in the given order, not the source's
    assert Counter((r["id"], r["lval"]) for r in rows) == Counter({(1, "a"): 1, (2, "b"): 1})


def test_dsl_drop_removes_named_columns_keeping_the_rest() -> None:
    rows = _reshape_src().drop("scratch").collect()
    assert all("scratch" not in r for r in rows)
    assert Counter((r["id"], r["lval"]) for r in rows) == Counter({(1, "a"): 1, (2, "b"): 1})


def test_dsl_rename_maps_columns_and_leaves_others() -> None:
    rows = _reshape_src().rename({"lval": "label"}).collect()
    assert all("lval" not in r for r in rows)
    assert Counter((r["id"], r["label"], r["scratch"]) for r in rows) == Counter(
        {(1, "a", 9): 1, (2, "b", 8): 1}
    )


def test_dsl_with_column_adds_a_new_column() -> None:
    rows = _reshape_src().with_column("id2", lambda b: pc.multiply(b["id"], 2)).collect()
    assert Counter((r["id"], r["id2"]) for r in rows) == Counter({(1, 2): 1, (2, 4): 1})


def test_dsl_with_column_replaces_an_existing_column_in_place() -> None:
    rows = _reshape_src().with_column("id", lambda b: pc.add(b["id"], 10)).collect()
    assert set(rows[0]) == {"id", "lval", "scratch"}  # replaced, not appended
    assert Counter(r["id"] for r in rows) == Counter({11: 1, 12: 1})


def test_dsl_reshaping_chains_and_matches_parallel() -> None:
    # keyless, stateless -> a fresh instance per subtask, so a uniform run(parallelism) scales them and
    # the output multiset is unchanged.
    s = (
        _reshape_src()
        .select("id", "lval")
        .rename({"lval": "label"})
        .with_column("id2", lambda b: pc.multiply(b["id"], 2))
    )
    assert multiset(s.run(parallelism=3)) == multiset(s.run())


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: _reshape_src().select(), "needs at least one"),
        (lambda: _reshape_src().select("id", "id"), "duplicate column"),
        (lambda: _reshape_src().drop(" "), "must be non-empty"),
        (lambda: _reshape_src().rename({}), "non-empty"),
        (lambda: _reshape_src().rename({"a": "x", "b": "x"}), "same name"),
        (lambda: _reshape_src().with_column("", lambda b: b["id"]), "non-empty"),
    ],
)
def test_dsl_reshaping_rejects_bad_args_at_build_time(build, match) -> None:
    with pytest.raises(ValueError, match=match):
        build()


# --- union (.union — keyless concat, SQL UNION ALL) ---------------------------------------------


def _u_left() -> Stream:
    return source(InMemorySource([data(id=[1, 2], v=["a", "b"]), EOS_FRAME]))


def _u_right() -> Stream:
    return source(InMemorySource([data(id=[2, 3], v=["b", "c"]), EOS_FRAME]))


def test_dsl_union_builds_a_keyless_two_input_dag_and_concatenates() -> None:
    u = _u_left().union(_u_right())
    graph = u.to_graph()
    assert [(vx.id, vx.kind) for vx in graph.vertices] == [
        ("v0", "source"),  # left
        ("v1", "source"),  # right, relabelled past the left's ids
        ("v2", "two_input"),
    ]
    # both edges into the union carry no key — a keyless merge, not a shuffle
    ports = sorted((e.dst_input_port, e.src, e.key_columns) for e in graph.edges)
    assert ports == [(0, "v0", None), (1, "v1", None)]
    # every row from both sides, duplicates kept: (2, "b") is on both -> count 2
    assert Counter((r["id"], r["v"]) for r in u.collect()) == Counter(
        {(1, "a"): 1, (2, "b"): 2, (3, "c"): 1}
    )


def test_dsl_union_matches_serial_when_parallel() -> None:
    u = _u_left().union(_u_right())
    assert multiset(u.run(parallelism=2)) == multiset(u.run())


def test_dsl_union_distributed_matches_single_process() -> None:
    u = _u_left().union(_u_right())
    assert multiset(u.run(workers=2, parallelism=2)) == multiset(u.run())


def test_dsl_union_of_three_streams_chains_and_relabels() -> None:
    # a.union(b).union(c): the second union splices in an operand that is itself a multi-vertex unioned
    # stream, exercising _combine's id-remap on a non-trivial subgraph.
    a = source(InMemorySource([data(id=[1], v=["a"]), EOS_FRAME]))
    b = source(InMemorySource([data(id=[2], v=["b"]), EOS_FRAME]))
    c = source(InMemorySource([data(id=[3], v=["c"]), EOS_FRAME]))
    u = a.union(b).union(c)
    assert len({vx.id for vx in u.to_graph().vertices}) == 5  # 3 sources + 2 unions, all unique
    assert Counter(r["id"] for r in u.collect()) == Counter({1: 1, 2: 1, 3: 1})


def test_dsl_rejects_a_self_union() -> None:
    s = _u_left()
    with pytest.raises(ValueError, match="unioned with itself"):
        s.union(s)
