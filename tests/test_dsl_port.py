"""Port-fidelity characterization of the ``Stream`` DSL's error contracts and build invariants.

These are Tier-2 tests: they pin the *observable* behavior of :mod:`nautilus.dsl` — the exact run-time
error strings, the immutability of a Stream, the join edge/schema shape for differently-named keys, and
the source-vertex parallelism carve-out — so a future Python->Rust rewrite that reproduces them is
provably faithful. Every golden here was read off the real code, not assumed.

Two altitudes of error surface differently and are pinned as such:
  * *build*-time contracts (``source(123)``, both ``on=`` and ``left_on=``, the shared-instance scale-up)
    raise a plain exception at the call/compile site;
  * *run*-time contracts (explode of a non-list column, rename of an absent column) raise inside the
    executor's ``TaskGroup``, so they arrive wrapped in an :class:`ExceptionGroup` and are matched on the
    unwrapped leaf.
"""

from __future__ import annotations

from collections import Counter

import pytest

from nautilus.dsl import source
from nautilus.operators import KeyedCount
from nautilus.testing import batch


def _leaves(exc: BaseException) -> list[BaseException]:
    """Flatten an ``ExceptionGroup`` to its leaf exceptions (the executor wraps a run-time operator error
    in a ``TaskGroup`` group, so the real ``TypeError``/``KeyError`` is one level down)."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


# --- (a) source() forms -------------------------------------------------------------------------


def test_source_rejects_a_bare_int_naming_the_accepted_forms() -> None:
    # The message enumerates every accepted form so a caller who passed the wrong thing knows the menu.
    with pytest.raises(
        TypeError,
        match=(
            r"source\(\) takes a SourceOperator, a pyarrow\.RecordBatch, or a sequence of "
            r"batches/frames, got int"
        ),
    ):
        source(123)


def test_source_of_a_bare_record_batch_collects_its_rows_without_a_hand_built_eos() -> None:
    # A bare pyarrow.RecordBatch is wrapped by from_batches, which appends the terminal EOS for us — the
    # stream runs to completion with no EOS the caller had to build.
    rb = batch(id=[1, 2, 3], v=["a", "b", "c"])
    rows = source(rb).collect()
    assert Counter((r["id"], r["v"]) for r in rows) == Counter(
        {(1, "a"): 1, (2, "b"): 1, (3, "c"): 1}
    )


# --- (b) Stream immutability --------------------------------------------------------------------


def test_two_chains_off_one_source_are_independent() -> None:
    # A Stream is a frozen value: each combinator returns a new Stream, so building one chain off a base
    # never mutates the base or a sibling chain. The base keeps its single source vertex and no edges;
    # each derived chain has exactly one appended vertex, and they do not share graph objects.
    base = source(batch(id=[1, 2], v=["a", "b"]))
    chain_a = base.map(lambda b: b)
    chain_b = base.tokenize("v", "word")

    base_graph = base.to_graph()
    assert len(base_graph.vertices) == 1
    assert len(base_graph.edges) == 0

    assert [(v.id, v.kind) for v in chain_a.to_graph().vertices] == [
        ("v0", "source"),
        ("v1", "one_input"),
    ]
    assert [(v.id, v.kind) for v in chain_b.to_graph().vertices] == [
        ("v0", "source"),
        ("v1", "one_input"),
    ]
    # Distinct graphs: adding to one did not extend the other or the base.
    assert base.to_graph().vertices != chain_a.to_graph().vertices
    assert len(chain_a.to_graph().vertices) == len(chain_b.to_graph().vertices) == 2


# --- (c) join with differently-named keys -------------------------------------------------------


def test_join_left_on_right_on_produces_expected_edge_keys_and_output_schema() -> None:
    # Differently-named keys: the left edge carries left_on, the right edge carries right_on, and the
    # output schema is the left's columns followed by the right's NON-key columns (the right key column,
    # a duplicate of the left key's values, is dropped).
    left = source(batch(lid=[1, 2], lval=["a", "b"]))
    right = source(batch(rid=[1, 3], rval=[10, 30]))
    joined = left.join(right, left_on="lid", right_on="rid")

    graph = joined.to_graph()
    assert [(v.id, v.kind) for v in graph.vertices] == [
        ("v0", "source"),  # left
        ("v1", "source"),  # right, relabelled past the left's ids
        ("v2", "two_input"),
    ]
    ports = sorted((e.dst_input_port, e.src, e.key_columns) for e in graph.edges)
    assert ports == [(0, "v0", ("lid",)), (1, "v1", ("rid",))]

    rows = joined.collect()
    # Only lid==rid==1 matches; output is left's columns + right's non-key column rval, no 'rid'.
    assert [sorted(r.items()) for r in rows] == [[("lid", 1), ("lval", "a"), ("rval", 10)]]
    assert sorted(rows[0].keys()) == ["lid", "lval", "rval"]


def test_join_rejects_both_on_and_left_on_right_on() -> None:
    left = source(batch(lid=[1, 2], lval=["a", "b"]))
    right = source(batch(rid=[1, 3], rval=[10, 30]))
    with pytest.raises(ValueError, match=r"give either on= or left_on=/right_on=, not both"):
        left.join(right, on="lid", left_on="lid", right_on="rid")


# --- (d) scale-up / run-time reshaping errors ---------------------------------------------------


def test_scaling_a_parallelism_one_apply_raises_the_shared_instance_error() -> None:
    # A parallelism-1 .apply hands the executor one shared operator instance. A later uniform
    # to_graph(parallelism=N) cannot replicate it, so the compile step (reached through .run) rejects it.
    # The error is NOT wrapped in an ExceptionGroup — it is raised at compile time, before any TaskGroup.
    scaled = source(batch(word=["a", "a", "b"])).apply(KeyedCount("word"))
    with pytest.raises(
        ValueError,
        match=(
            r"vertex 'v1' has parallelism 3 but its factory returns one shared instance; "
            r"parallelism > 1 needs a factory that builds a fresh operator each call"
        ),
    ):
        scaled.run(parallelism=3)


def test_explode_of_a_non_list_column_raises_at_run_time() -> None:
    # The list-column check is a run-time check against the data (column types are learned from the first
    # batch), so it fires inside the executor and is wrapped in an ExceptionGroup.
    with pytest.raises(ExceptionGroup) as excinfo:
        source(batch(x=[1, 2, 3])).explode("x").collect()
    leaves = _leaves(excinfo.value)
    assert [type(leaf).__name__ for leaf in leaves] == ["TypeError"]
    assert str(leaves[0]) == "explode(): column 'x' is int64, not a list column"


def test_rename_of_an_absent_column_raises_at_run_time() -> None:
    # Column presence resolves at run time against the batch's schema, so renaming a name the data lacks
    # raises a KeyError inside the executor, wrapped in an ExceptionGroup.
    with pytest.raises(ExceptionGroup) as excinfo:
        source(batch(a=[1, 2])).rename({"missing": "new"}).collect()
    leaves = _leaves(excinfo.value)
    assert [type(leaf).__name__ for leaf in leaves] == ["KeyError"]
    # KeyError.__str__ adds its own quoting around the message.
    assert leaves[0].args[0] == "rename(): column(s) ['missing'] not in schema ['a']"


# --- (e) to_graph(parallelism=N) leaves the source at parallelism 1 -----------------------------


def test_to_graph_parallelism_override_leaves_the_source_vertex_at_one() -> None:
    # The uniform scale-up knob overrides every NON-source vertex; the source stays at parallelism 1
    # (a source fans out on its own — widening it here would double-read the data).
    graph = (
        source(batch(id=[1, 2], v=["a", "b"]))
        .map(lambda b: b)
        .map(lambda b: b)
        .to_graph(parallelism=5)
    )
    assert [(v.id, v.kind, v.parallelism) for v in graph.vertices] == [
        ("v0", "source", 1),
        ("v1", "one_input", 5),
        ("v2", "one_input", 5),
    ]
