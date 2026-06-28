"""The compiler lowers a LogicalGraph to a PhysicalPlan: physical ids by position, a synthesized sink,
a partitioner spec per edge chosen from the downstream operator, and a fresh-factory check at P>1.
"""

from __future__ import annotations

import pytest

from nautilus.api import LogicalVertex, linear_graph
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
    assert isinstance(specs[("op0", "op1")], KeyGroupSpec)  # op1 is keyed, P=3
    assert specs[("op0", "op1")].key_columns == ("word",)
    assert specs[("op0", "op1")].group_table == (0, 1, 2)  # default G==Q is the identity table
    assert isinstance(specs[("op1", "sink")], ForwardSpec)  # the sink is one instance


def test_keyless_fanout_selects_round_robin() -> None:
    g = linear_graph(
        lambda: InMemorySource([]),
        [LogicalVertex("op0", lambda: MapBatch(lambda b: b), "one_input", 2)],
    )
    plan = compile_graph(g)
    specs = {(e.src_operator_id, e.dst_operator_id): e.spec for e in plan.edges}
    assert isinstance(specs[("source", "op0")], RoundRobinSpec)


def test_parallel_vertex_with_shared_instance_is_rejected() -> None:
    shared = MapBatch(lambda b: b)
    g = linear_graph(
        lambda: InMemorySource([]),
        [LogicalVertex("op0", lambda: shared, "one_input", parallelism=2)],
    )
    with pytest.raises(ValueError, match="shared instance"):
        compile_graph(g)


def test_keyed_operator_at_parallelism_one_forwards() -> None:
    # Q==1 is Forward even when keyed — matches the legacy partitioner selection.
    g = linear_graph(
        lambda: InMemorySource([]),
        [LogicalVertex("op0", lambda: KeyedCount("word"), "one_input", 1, ("word",))],
    )
    plan = compile_graph(g)
    assert isinstance(plan.edges[0].spec, ForwardSpec)


# --- key groups: G > Q builds a round-robin table; G < Q is rejected ----------------------------


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
    assert spec.group_table == (0, 1, 2, 3)  # G defaults to Q -> identity
