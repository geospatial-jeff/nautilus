"""The logical IR is a pure, validated value layer: it constructs linear graphs and rejects shapes a
compiler could not lower (no source, a parallel source, duplicate ids, an unknown kind)."""

from __future__ import annotations

import pytest

from nautilus.api import LogicalGraph, LogicalVertex, linear_graph


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
        LogicalVertex(id="x", factory=lambda: object(), kind="two_input")
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
