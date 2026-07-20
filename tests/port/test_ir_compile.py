"""Port-fidelity (Tier 1) tests pinning the logical IR's validation and the compiler's lowering.

These characterize the *observable* behavior a future Python->Rust rewrite must reproduce: the exact
error messages the IR raises on a malformed vertex/edge/graph, the value-equality contract of the frozen
IR dataclasses, the insertion-index tie-break of the topological order, and the compiler's branches
(the single-leaf rule, the key-group ceiling, keyless-union routing, and a cloudpickle round-trip that
preserves every edge spec's fields). Every golden here was produced by running the real code and pasting
the literal it emitted; see the module tests' notes where actual behavior refined the brief's wording.
"""

from __future__ import annotations

import cloudpickle
import pytest

from nautilus.api import (
    LogicalEdge,
    LogicalGraph,
    LogicalVertex,
    async_one_input,
    async_sink,
    one_input,
    source,
    two_input,
)
from nautilus.api.graph import _topological_order
from nautilus.compile import KeyGroupSpec, RoundRobinSpec, compile_graph
from nautilus.compile.lower import MAX_KEY_GROUPS
from nautilus.compile.plan import ForwardSpec
from nautilus.operators import InMemorySource

# A shared factory used wherever the operator built is irrelevant to the property under test. Reusing one
# callable object matters for the equality tests: the frozen dataclasses compare their factory *by
# identity*, so two vertices are only equal when they share this exact callable.
_F = lambda: object()  # noqa: E731


class _StubJoin:  # a bare two-input operator; a module-level class so cloudpickle can reference it
    pass


# --- (a)-(c) LogicalEdge field validation -------------------------------------------------------


def test_edge_rejects_out_of_range_dst_input_port() -> None:
    with pytest.raises(ValueError, match="dst_input_port must be 0 or 1"):
        LogicalEdge("a", "b", 2)


def test_edge_rejects_empty_key_columns_tuple() -> None:
    # () is neither keyless (None) nor keyed (a non-empty tuple) — the message names "non-empty tuple".
    with pytest.raises(ValueError, match="non-empty tuple"):
        LogicalEdge("a", "b", 0, ())


def test_edge_rejects_blank_key_column_name() -> None:
    with pytest.raises(ValueError, match="non-empty column names"):
        LogicalEdge("a", "b", 0, ("",))


def test_edge_rejects_empty_src() -> None:
    with pytest.raises(ValueError, match="non-empty src and dst"):
        LogicalEdge("", "b")


def test_edge_rejects_empty_dst() -> None:
    with pytest.raises(ValueError, match="non-empty src and dst"):
        LogicalEdge("a", "")


# --- (d) IR value equality: identical fields compare equal; flipping any one field differs -------


def test_identical_vertices_are_equal_and_hash_equal() -> None:
    v1 = LogicalVertex("v", _F, "one_input", 2, ("k",), True)
    v2 = LogicalVertex("v", _F, "one_input", 2, ("k",), True)
    assert v1 == v2
    assert hash(v1) == hash(v2)


def test_flipping_any_vertex_field_makes_it_unequal() -> None:
    base = LogicalVertex("v", _F, "one_input", 2, ("k",), True)
    assert base != LogicalVertex("w", _F, "one_input", 2, ("k",), True)  # id
    assert base != LogicalVertex("v", _F, "one_input", 3, ("k",), True)  # parallelism
    assert base != LogicalVertex("v", _F, "one_input", 2, ("m",), True)  # key_columns
    # kind is bound up with copartitioned's validation, so exercise copartitioned on a two_input, where
    # both values are legal, to prove the flag participates in equality.
    tj = LogicalVertex("j", _F, "two_input", 2, None, True)
    assert tj != LogicalVertex("j", _F, "two_input", 2, None, False)  # copartitioned


def test_identical_edges_are_equal_and_hash_equal() -> None:
    e1 = LogicalEdge("a", "b", 1, ("k",))
    e2 = LogicalEdge("a", "b", 1, ("k",))
    assert e1 == e2
    assert hash(e1) == hash(e2)


def test_flipping_any_edge_field_makes_it_unequal() -> None:
    base = LogicalEdge("a", "b", 1, ("k",))
    assert base != LogicalEdge("c", "b", 1, ("k",))  # src
    assert base != LogicalEdge("a", "c", 1, ("k",))  # dst
    assert base != LogicalEdge("a", "b", 0, ("k",))  # dst_input_port
    assert base != LogicalEdge("a", "b", 1, ("m",))  # key_columns


def test_identical_graphs_are_equal_and_hash_equal() -> None:
    g1 = LogicalGraph(
        (LogicalVertex("source", _F, "source"), LogicalVertex("op0", _F, "one_input"))
    )
    g2 = LogicalGraph(
        (LogicalVertex("source", _F, "source"), LogicalVertex("op0", _F, "one_input"))
    )
    assert g1 == g2
    assert hash(g1) == hash(g2)


# --- (e) topological tie-break tracks vertex INSERTION position, not lexical/dict order ----------


def _diamond(order: tuple[str, ...]) -> tuple:
    """A diamond where the two middle vertices (``zeta`` and ``alpha``) are simultaneously ready after
    the source. ``order`` chooses which of the two comes first in the ``vertices`` tuple; the ids are
    chosen so lexical order ("alpha" < "zeta") disagrees with the requested insertion order, which is
    what distinguishes a position-based tie-break from a lexical one.
    """
    verts = {
        "source": LogicalVertex("source", _F, "source"),
        "zeta": LogicalVertex("zeta", _F, "one_input"),
        "alpha": LogicalVertex("alpha", _F, "one_input"),
        "join": LogicalVertex("join", _F, "two_input", copartitioned=False),
    }
    vertices = (verts["source"], *(verts[k] for k in order), verts["join"])
    edges = (
        LogicalEdge("source", "zeta", 0),
        LogicalEdge("source", "alpha", 0),
        LogicalEdge("zeta", "join", 0),
        LogicalEdge("alpha", "join", 1),
    )
    return vertices, edges


def test_topological_tie_break_follows_insertion_position() -> None:
    # zeta positioned before alpha -> zeta emitted first, even though "alpha" sorts first lexically.
    v, e = _diamond(("zeta", "alpha"))
    assert _topological_order(v, e) == ["source", "zeta", "alpha", "join"]
    # reorder the two ready vertices -> the emitted order follows the new position, proving the tie-break
    # is position-based and not dict/lexical (a lexical tie-break would emit "alpha" first both times).
    v, e = _diamond(("alpha", "zeta"))
    assert _topological_order(v, e) == ["source", "alpha", "zeta", "join"]


# --- (f) DAG / linear-graph validation branches -------------------------------------------------


def test_dag_with_no_source_rejected() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        LogicalGraph(
            vertices=(one_input("x", _F), one_input("y", _F)),
            edges=(LogicalEdge("x", "y"),),
        )


def test_dag_parallel_source_rejected() -> None:
    with pytest.raises(ValueError, match="parallelism 1"):
        LogicalGraph(
            vertices=(LogicalVertex("s", _F, "source", parallelism=2), one_input("y", _F)),
            edges=(LogicalEdge("s", "y"),),
        )


def test_two_input_in_linear_graph_rejected() -> None:
    with pytest.raises(ValueError, match="needs explicit edges for its two inputs"):
        LogicalGraph((source("s", _F), two_input("j", _F)))


def test_async_sink_in_linear_graph_rejected() -> None:
    with pytest.raises(ValueError, match="needs explicit edges"):
        LogicalGraph((source("s", _F), async_sink("k", _F)))


def test_async_one_input_in_linear_graph_rejected() -> None:
    with pytest.raises(ValueError, match="needs explicit edges"):
        LogicalGraph((source("s", _F), async_one_input("a", _F)))


def test_copartitioned_false_on_non_two_input_rejected() -> None:
    with pytest.raises(ValueError, match="only meaningful on a two_input"):
        LogicalVertex("x", _F, "one_input", copartitioned=False)


# --- (g) the compiler's single-leaf rule --------------------------------------------------------


def test_two_leaves_rejected_and_message_lists_both_physical_ids() -> None:
    # source -> x and source -> y leaves both a one_input with no downstream, so the graph has two leaves.
    # The message names the *physical* ids (op0, op1), not the logical ones.
    with pytest.raises(ValueError, match=r"exactly one leaf.*op0.*op1"):
        compile_graph(
            LogicalGraph(
                vertices=(source("s", _F), one_input("x", _F), one_input("y", _F)),
                edges=(LogicalEdge("s", "x"), LogicalEdge("s", "y")),
            )
        )


# --- (h) MAX_KEY_GROUPS ceiling -----------------------------------------------------------------


def _keyed_graph(parallelism: int) -> LogicalGraph:
    return LogicalGraph(
        vertices=(
            source("s", lambda: InMemorySource([])),
            one_input("op", lambda: object(), parallelism=parallelism, key_columns=("k",)),
        ),
        edges=(LogicalEdge("s", "op", 0, ("k",)),),
    )


def test_max_key_groups_constant_is_32768() -> None:
    assert MAX_KEY_GROUPS == 32768


def test_key_groups_at_the_ceiling_compiles() -> None:
    plan = compile_graph(_keyed_graph(2), key_groups=32768)
    keyed = next(e for e in plan.edges if isinstance(e.spec, KeyGroupSpec))
    assert len(keyed.spec.group_table) == 32768


def test_key_groups_above_the_ceiling_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds the maximum"):
        compile_graph(_keyed_graph(2), key_groups=32769)


# --- (i) a keyless union routes with RoundRobin/Forward, never a KeyGroupSpec --------------------


def test_keyless_union_never_produces_a_keygroup_spec() -> None:
    # copartitioned=False with keyless edges into a parallelism-2 union: the compiler must not key-shuffle
    # it. Each source is one instance, so its 1->2 fan-out is a keyless width change (RoundRobin); the
    # union's edge to the single-instance sink forwards.
    plan = compile_graph(
        LogicalGraph(
            vertices=(
                source("a", lambda: InMemorySource([])),
                source("b", lambda: InMemorySource([])),
                two_input("u", lambda: _StubJoin(), parallelism=2, copartitioned=False),
            ),
            edges=(LogicalEdge("a", "u", 0), LogicalEdge("b", "u", 1)),
        )
    )
    assert not any(isinstance(e.spec, KeyGroupSpec) for e in plan.edges)
    assert all(isinstance(e.spec, (RoundRobinSpec, ForwardSpec)) for e in plan.edges)
    inbound = sorted(
        (e for e in plan.edges if e.dst_operator_id == "op0"), key=lambda e: e.dst_input_port
    )
    assert [type(e.spec).__name__ for e in inbound] == ["RoundRobinSpec", "RoundRobinSpec"]


# --- (j) cloudpickle preserves each edge spec's fields exactly ----------------------------------


def test_cloudpickle_round_trip_preserves_edge_spec_fields() -> None:
    # A parallel keyed join: each inbound edge is a KeyGroupSpec (its own key_columns, a shared table),
    # the sink edge a keyless Forward. The plan's operators carry factory callables that cloudpickle
    # recreates as distinct objects, so the *whole plan* does not compare equal after a round-trip; the
    # port-relevant fact is that every edge spec — pure data — survives byte-for-byte, so pin the edges.
    plan = compile_graph(
        LogicalGraph(
            vertices=(
                source("a", lambda: InMemorySource([])),
                source("b", lambda: InMemorySource([])),
                two_input("j", lambda: _StubJoin(), parallelism=2),
            ),
            edges=(LogicalEdge("a", "j", 0, ("lk",)), LogicalEdge("b", "j", 1, ("rk",))),
        )
    )
    restored = cloudpickle.loads(cloudpickle.dumps(plan))
    assert restored.edges == plan.edges  # specs are pure data, so the edges compare exactly

    fields = {
        (e.src_operator_id, e.dst_operator_id, e.dst_input_port): (
            getattr(e.spec, "key_columns", None),
            getattr(e.spec, "group_table", None),
        )
        for e in restored.edges
    }
    assert fields == {
        ("source0", "op0", 0): (("lk",), (0, 1)),
        ("source1", "op0", 1): (("rk",), (0, 1)),
        ("op0", "sink", 0): (None, None),
    }
