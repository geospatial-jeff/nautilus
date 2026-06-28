"""The logical IR is a pure, validated value layer: it constructs linear graphs and explicit-edge DAGs
(joins), and rejects shapes a compiler could not lower (no source, a parallel source, duplicate ids, an
unknown kind, a malformed join, a self-join, a cycle)."""

from __future__ import annotations

import pytest

from nautilus.api import (
    LogicalEdge,
    LogicalGraph,
    LogicalVertex,
    linear_graph,
    one_input,
    source,
    two_input,
)


def _vertex(
    vid: str, parallelism: int = 1, key_columns: tuple[str, ...] | None = None
) -> LogicalVertex:
    return LogicalVertex(
        id=vid,
        factory=lambda: object(),
        kind="one_input",
        parallelism=parallelism,
        key_columns=key_columns,
    )


def test_linear_graph_prepends_a_single_source() -> None:
    g = linear_graph(lambda: object(), [_vertex("op0"), _vertex("op1", 3, ("k",))])
    assert [v.id for v in g.vertices] == ["source", "op0", "op1"]
    assert g.vertices[0].kind == "source" and g.vertices[0].parallelism == 1
    assert g.vertices[2].key_columns == ("k",)


def test_empty_transforms_is_a_valid_source_only_graph() -> None:
    g = linear_graph(lambda: object(), [])
    assert [v.id for v in g.vertices] == ["source"]


def test_duplicate_vertex_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        linear_graph(lambda: object(), [_vertex("op0"), _vertex("op0")])


def test_vertex_validates_kind_and_parallelism() -> None:
    with pytest.raises(ValueError, match="unknown vertex kind"):
        LogicalVertex(id="x", factory=lambda: object(), kind="three_input")
    with pytest.raises(ValueError, match="parallelism must be >= 1"):
        _vertex("op0", parallelism=0)
    with pytest.raises(ValueError, match="non-empty id"):
        LogicalVertex(id="", factory=lambda: object(), kind="one_input")


def test_graph_requires_exactly_one_leading_source() -> None:
    src = LogicalVertex(id="source", factory=lambda: object(), kind="source")
    # two sources
    with pytest.raises(ValueError, match="exactly one source"):
        LogicalGraph((src, LogicalVertex(id="s2", factory=lambda: object(), kind="source")))
    # a transform first (no leading source)
    with pytest.raises(ValueError, match="exactly one source"):
        LogicalGraph((_vertex("op0"),))


def test_parallel_source_is_rejected() -> None:
    src = LogicalVertex(id="source", factory=lambda: object(), kind="source", parallelism=2)
    with pytest.raises(ValueError, match="source vertex must have parallelism 1"):
        LogicalGraph((src,))


def test_key_columns_rejected_on_non_one_input() -> None:
    # keying is a one-input convenience; a source has no input and a join keys per edge.
    with pytest.raises(ValueError, match="only meaningful on a one_input"):
        LogicalVertex(id="j", factory=lambda: object(), kind="two_input", key_columns=("k",))


# --- explicit-edge DAGs: a join is two sources into a two_input vertex on ports 0 and 1 ----------


def _src(vid: str) -> LogicalVertex:
    return source(vid, lambda: object())


def _join(left: str = "a", right: str = "b", jid: str = "j") -> LogicalGraph:
    return LogicalGraph(
        vertices=(_src(left), _src(right), two_input(jid, lambda: object(), parallelism=2)),
        edges=(LogicalEdge(left, jid, 0, ("k",)), LogicalEdge(right, jid, 1, ("k",))),
    )


def test_two_source_join_is_a_valid_dag() -> None:
    g = _join()
    assert {v.id for v in g.vertices} == {"a", "b", "j"}  # multiple sources are allowed in a DAG
    assert [(e.src, e.dst, e.dst_input_port) for e in g.edges] == [("a", "j", 0), ("b", "j", 1)]


def test_edge_to_unknown_vertex_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dst vertex"):
        LogicalGraph(vertices=(_src("a"),), edges=(LogicalEdge("a", "ghost"),))


def test_source_with_inbound_edge_rejected() -> None:
    with pytest.raises(ValueError, match="must have no inbound"):
        LogicalGraph(
            vertices=(_src("a"), one_input("op", lambda: object()), _src("b")),
            edges=(LogicalEdge("a", "op"), LogicalEdge("op", "b")),  # op -> b, but b is a source
        )


def test_two_input_needs_both_ports() -> None:
    with pytest.raises(ValueError, match="needs exactly one inbound edge per input port"):
        LogicalGraph(
            vertices=(_src("a"), _src("b"), two_input("j", lambda: object())),
            edges=(LogicalEdge("a", "j", 0), LogicalEdge("b", "j", 0)),  # both on port 0
        )


def test_one_input_edge_must_be_port_zero() -> None:
    with pytest.raises(ValueError, match="needs exactly one inbound edge per input port"):
        LogicalGraph(
            vertices=(_src("a"), one_input("op", lambda: object())),
            edges=(LogicalEdge("a", "op", 1),),  # a one-input has only port 0
        )


def test_self_join_rejected() -> None:
    with pytest.raises(ValueError, match="self-join"):
        LogicalGraph(
            vertices=(_src("a"), two_input("j", lambda: object())),
            edges=(LogicalEdge("a", "j", 0), LogicalEdge("a", "j", 1)),  # one src into both ports
        )


def test_parallel_join_requires_keyed_inputs() -> None:
    # a keyless edge into a parallel join fans out round-robin, scattering a key's two sides onto
    # different instances so matches vanish silently — reject it at build time.
    with pytest.raises(ValueError, match="keyless"):
        LogicalGraph(
            vertices=(_src("a"), _src("b"), two_input("j", lambda: object(), parallelism=2)),
            edges=(LogicalEdge("a", "j", 0), LogicalEdge("b", "j", 1)),  # no key_columns
        )


def test_serial_join_allows_keyless_inputs() -> None:
    # at parallelism 1 a single instance owns everything, so keyless inputs are harmless
    g = LogicalGraph(
        vertices=(_src("a"), _src("b"), two_input("j", lambda: object())),
        edges=(LogicalEdge("a", "j", 0), LogicalEdge("b", "j", 1)),
    )
    assert len(g.edges) == 2


def test_cycle_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        LogicalGraph(
            vertices=(
                _src("a"),
                one_input("x", lambda: object()),
                one_input("y", lambda: object()),
            ),
            edges=(LogicalEdge("x", "y"), LogicalEdge("y", "x")),
        )
