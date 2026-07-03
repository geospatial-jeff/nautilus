"""The compiler lowers a LogicalGraph to a PhysicalPlan: physical ids by position, a synthesized sink,
a partitioner spec per edge chosen from the downstream operator,
and a fresh-factory check at parallelism above 1.
"""

from __future__ import annotations

import pytest

from nautilus.api import LogicalEdge, LogicalGraph, LogicalVertex, linear_graph, source, two_input
from nautilus.compile import ForwardSpec, KeyGroupSpec, RoundRobinSpec, compile_graph
from nautilus.operators import InMemorySource, KeyedCount, MapBatch, Tokenize


def _graph():
    return linear_graph(
        lambda: InMemorySource([]),
        [
            LogicalVertex("op0", lambda: Tokenize("line", "word"), "one_input"),
            LogicalVertex("op1", lambda: KeyedCount("word"), "one_input", 3, ("word",)),
        ],
    )


def test_names_operators_by_position_with_synthesized_sink() -> None:
    plan = compile_graph(_graph())
    assert [(o.operator_id, o.kind, o.parallelism) for o in plan.operators] == [
        ("source", "source", 1),
        ("op0", "one_input", 1),
        ("op1", "one_input", 3),
        ("sink", "sink", 1),
    ]
    by_id = {o.operator_id: o for o in plan.operators}
    sink = by_id["sink"]
    assert sink.op_class == "CollectSink" and sink.factory is None
    assert by_id["op0"].op_class == "Tokenize"


def test_edge_specs_come_from_the_downstream_operator() -> None:
    plan = compile_graph(_graph())
    specs = {(e.src_operator_id, e.dst_operator_id): e.spec for e in plan.edges}
    assert isinstance(specs[("source", "op0")], ForwardSpec)  # op0 is a single instance
    assert isinstance(specs[("op0", "op1")], KeyGroupSpec)  # op1 is keyed, parallelism 3
    assert specs[("op0", "op1")].key_columns == ("word",)
    # default (key-group count equals parallelism) is the identity table
    assert specs[("op0", "op1")].group_table == (0, 1, 2)
    assert isinstance(specs[("op1", "sink")], ForwardSpec)  # the sink is one instance


def test_keyless_fanout_selects_round_robin() -> None:
    # The source is always one instance, so its edge into a wider stage is a one-to-many width change:
    # no one-to-one mapping, so it rebalances round-robin (the one keyless edge that cannot forward).
    g = linear_graph(
        lambda: InMemorySource([]),
        [LogicalVertex("op0", lambda: MapBatch(lambda b: b), "one_input", 2)],
    )
    plan = compile_graph(g)
    specs = {(e.src_operator_id, e.dst_operator_id): e.spec for e in plan.edges}
    assert isinstance(specs[("source", "op0")], RoundRobinSpec)


def test_equal_width_keyless_edge_forwards_for_locality() -> None:
    # Two keyless stages of the same width: the edge between them forwards straight across (sender i ->
    # instance i), keeping each row on its origin instance instead of shuffling. The source's 1 -> 3
    # fan-out still rebalances (a width change).
    g = linear_graph(
        lambda: InMemorySource([]),
        [
            LogicalVertex("op0", lambda: MapBatch(lambda b: b), "one_input", 3),
            LogicalVertex("op1", lambda: MapBatch(lambda b: b), "one_input", 3),
        ],
    )
    plan = compile_graph(g)
    specs = {(e.src_operator_id, e.dst_operator_id): e.spec for e in plan.edges}
    assert isinstance(specs[("source", "op0")], RoundRobinSpec)  # 1 -> 3 fan-out
    assert isinstance(specs[("op0", "op1")], ForwardSpec)  # 3 -> 3 keyless: co-located forward


def test_keyless_width_change_between_stages_rebalances() -> None:
    # A keyless edge whose two stages differ in width has no one-to-one mapping, so it rebalances even
    # away from the source (here 4 -> 2).
    g = linear_graph(
        lambda: InMemorySource([]),
        [
            LogicalVertex("op0", lambda: MapBatch(lambda b: b), "one_input", 4),
            LogicalVertex("op1", lambda: MapBatch(lambda b: b), "one_input", 2),
        ],
    )
    plan = compile_graph(g)
    specs = {(e.src_operator_id, e.dst_operator_id): e.spec for e in plan.edges}
    assert isinstance(specs[("op0", "op1")], RoundRobinSpec)


def test_parallel_vertex_with_shared_instance_is_rejected() -> None:
    shared = MapBatch(lambda b: b)
    g = linear_graph(
        lambda: InMemorySource([]),
        [LogicalVertex("op0", lambda: shared, "one_input", parallelism=2)],
    )
    with pytest.raises(ValueError, match="shared instance"):
        compile_graph(g)


def test_keyed_operator_at_parallelism_one_forwards() -> None:
    # parallelism 1 is Forward even when keyed — matches the legacy partitioner selection.
    g = linear_graph(
        lambda: InMemorySource([]),
        [LogicalVertex("op0", lambda: KeyedCount("word"), "one_input", 1, ("word",))],
    )
    plan = compile_graph(g)
    assert isinstance(plan.edges[0].spec, ForwardSpec)


# --- key groups: more key groups than instances builds a round-robin table; fewer is rejected ---


def _keyed_graph(parallelism: int):
    return linear_graph(
        lambda: InMemorySource([]),
        [LogicalVertex("op0", lambda: KeyedCount("word"), "one_input", parallelism, ("word",))],
    )


def test_key_groups_above_parallelism_build_a_round_robin_table() -> None:
    plan = compile_graph(_keyed_graph(3), key_groups=5)
    spec = plan.edges[0].spec
    assert isinstance(spec, KeyGroupSpec)
    assert spec.group_table == (0, 1, 2, 0, 1)  # 5 groups round-robin over 3 instances


def test_key_groups_below_parallelism_rejected_at_compile() -> None:
    with pytest.raises(ValueError, match="below the operator parallelism"):
        compile_graph(_keyed_graph(3), key_groups=2)


def test_default_key_groups_is_the_identity_table() -> None:
    spec = compile_graph(_keyed_graph(4)).edges[0].spec
    assert isinstance(spec, KeyGroupSpec)
    # key-group count defaults to the parallelism -> identity
    assert spec.group_table == (0, 1, 2, 3)


# --- a two-input join: two sources shuffle into one two_input vertex on ports 0 and 1 -----------


class _StubJoin:  # a placeholder two-input operator (the real HashJoin lands in a later sub-stage)
    pass


def _join_graph(jid: str = "j", left: str = "a", right: str = "b") -> LogicalGraph:
    return LogicalGraph(
        vertices=(
            source(left, lambda: InMemorySource([])),
            source(right, lambda: InMemorySource([])),
            two_input(jid, lambda: _StubJoin(), parallelism=2),
        ),
        edges=(LogicalEdge(left, jid, 0, ("lk",)), LogicalEdge(right, jid, 1, ("rk",))),
    )


def test_two_source_join_compiles_to_ported_keygroup_edges() -> None:
    plan = compile_graph(_join_graph())
    ops = {(o.operator_id, o.kind, o.parallelism) for o in plan.operators}
    assert {("source0", "source", 1), ("source1", "source", 1), ("op0", "two_input", 2)} <= ops
    join_edges = sorted(
        (e for e in plan.edges if e.dst_operator_id == "op0"), key=lambda e: e.dst_input_port
    )
    assert [e.dst_input_port for e in join_edges] == [0, 1]  # left = port 0, right = port 1
    assert all(isinstance(e.spec, KeyGroupSpec) for e in join_edges)
    # both sides read the join's one parallelism + the run's one key-group count, so the
    # tables match exactly: an equal key co-partitions to the same join instance from the
    # left and the right.
    assert join_edges[0].spec.group_table == join_edges[1].spec.group_table
    assert (join_edges[0].spec.key_columns, join_edges[1].spec.key_columns) == (("lk",), ("rk",))


def _shape(plan: object) -> tuple:
    return (
        [(o.operator_id, o.kind, o.parallelism, o.op_class) for o in plan.operators],  # type: ignore[attr-defined]
        [
            (
                e.src_operator_id,
                e.dst_operator_id,
                e.dst_input_port,
                type(e.spec).__name__,
                getattr(e.spec, "key_columns", None),
                getattr(e.spec, "group_table", None),
            )
            for e in plan.edges  # type: ignore[attr-defined]
        ],
    )


def test_compile_is_invariant_to_vertex_ids() -> None:
    # Physical ids come from topological position, so two join graphs identical but for their vertex ids
    # compile to the same plan — the property the structural digest's reproducibility rests on.
    assert _shape(compile_graph(_join_graph(jid="j", left="a", right="b"))) == _shape(
        compile_graph(_join_graph(jid="JOIN", left="L", right="R"))
    )
