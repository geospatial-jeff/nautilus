"""Tier 4 property-based conformance tests for dsl-ir.

Two structural invariants of the fluent ``Stream`` DSL and the linear-graph edge synthesis that a
Python->Rust port must reproduce, each asserted over many fixed-seed random inputs rather than a handful
of golden cases. The first pins how :meth:`Stream._combine` (``.join`` / ``.union``) splices an
independent right subgraph in without an id collision; the second pins how the compiler synthesizes a
linear graph's port-0 edges from each vertex's ``key_columns`` convenience. A divergence in either is
invisible in a single example but silently breaks a port, so we drive the whole input space.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nautilus.api import LogicalEdge, LogicalGraph, one_input, source
from nautilus.compile.lower import _logical_edges
from nautilus.dsl import source as stream_source

# The one-input combinators that grow a stream by one vertex + one port-0 edge, so a random pipeline of
# any depth is a random-length draw from these. Each is keyless (the edge carries no key_columns), which
# is what invariant 1 needs: the property is about id/edge *topology*, and keying is invariant 2's job.
_LINEAR_VERBS = (
    lambda s: s.map(lambda b: b),
    lambda s: s.filter(lambda b: b),
    lambda s: s.select("a"),
    lambda s: s.drop("a"),
    lambda s: s.rename({"a": "b"}),
)


def _random_stream(rng: np.random.Generator):
    """A source followed by a random-length (0..4) chain of keyless one-input verbs — an independent
    subgraph whose vertices are ``v0 .. v{n}`` and whose edges are the ``v{i}->v{i+1}`` port-0 chain.
    """
    s = stream_source([])
    for _ in range(int(rng.integers(0, 5))):
        s = _LINEAR_VERBS[int(rng.integers(0, len(_LINEAR_VERBS)))](s)
    return s


def _assert_isolated(stream) -> None:
    """The invariant every ``_combine`` result must satisfy: vertex ids are unique and contiguous
    ``v0..v{N}``, and every edge points at two ids that exist."""
    ids = [v.id for v in stream._vertices]
    assert len(set(ids)) == len(ids), f"vertex ids collided: {ids}"
    nums = sorted(int(v.id[1:]) for v in stream._vertices)
    assert nums == list(range(len(stream._vertices))), f"ids not contiguous v0..vN: {ids}"
    idset = set(ids)
    for e in stream._edges:
        assert e.src in idset, f"edge src {e.src!r} names no vertex"
        assert e.dst in idset, f"edge dst {e.dst!r} names no vertex"


def test_vertex_id_isolation_in_combine() -> None:
    """For any two independent streams, joining/unioning them (and chaining further) yields unique,
    contiguous vertex ids with every edge valid, the right subgraph shifted past the left's count.
    """
    rng = np.random.default_rng(1234)
    for _ in range(500):
        left = _random_stream(rng)
        right = _random_stream(rng)
        shift = len(left._vertices)

        combined = left.join(right, on="a") if rng.integers(0, 2) else left.union(right)
        _assert_isolated(combined)

        # The right subgraph was relabeled by exactly +shift and its edges remapped in lock-step, so it is
        # byte-identical to the original with every id shifted — nothing else about it changed.
        m = len(right._vertices)
        remap = {f"v{i}": f"v{shift + i}" for i in range(m)}
        shifted_right_vertices = tuple(replace(v, id=remap[v.id]) for v in right._vertices)
        assert combined._vertices[shift : shift + m] == shifted_right_vertices
        shifted_right_edges = tuple(
            replace(e, src=remap[e.src], dst=remap[e.dst]) for e in right._edges
        )
        # The combined edge list is: left's edges, the shifted right's edges, then the two new port-0/1
        # edges into the appended two-input vertex.
        assert combined._edges[len(left._edges) : len(left._edges) + len(right._edges)] == (
            shifted_right_edges
        )

        # The appended two-input vertex is the tail; its two inbound edges land on ports 0 (left tail) and
        # 1 (shifted right tail), and its id sits just past both subgraphs.
        assert combined._tail == f"v{shift + m}"
        inbound = [e for e in combined._edges if e.dst == combined._tail]
        assert sorted(e.dst_input_port for e in inbound) == [0, 1]
        assert {e.src for e in inbound} == {left._tail, remap[right._tail]}

        # A chained combine over the result must stay isolated too — the shift is relative each time.
        third = _random_stream(rng)
        chained = combined.join(third, on="a")
        _assert_isolated(chained)

        # The whole thing is a legal DAG the IR accepts (no duplicate ids, every endpoint present).
        assert isinstance(chained.to_graph(), LogicalGraph)


def test_linear_edge_key_columns_inheritance() -> None:
    """For any linear graph, each synthesized port-0 LogicalEdge carries its downstream vertex's
    key_columns; when explicit edges are given instead, the explicit key_columns wins."""
    rng = np.random.default_rng(1234)
    # Draw keyings from the whole space a keyed one-input edge admits: keyless (None), single key, and a
    # multi-column key — plus column names with mixed case/underscores so a port can't assume a shape.
    key_choices: tuple[tuple[str, ...] | None, ...] = (
        None,
        ("k",),
        ("Lat",),
        ("k1", "k2"),
        ("a_col", "b_col", "c_col"),
    )
    for _ in range(500):
        n = int(rng.integers(1, 6))  # 1..5 transforms after the source
        transforms = []
        for i in range(n):
            keys = key_choices[int(rng.integers(0, len(key_choices)))]
            transforms.append(one_input(f"t{i}", lambda: object(), key_columns=keys))
        graph = LogicalGraph((source("s", lambda: object()), *transforms))

        edges = _logical_edges(graph)
        vs = graph.vertices
        # One synthesized edge per adjacent pair, each port 0, each carrying the *downstream* (consuming)
        # vertex's key_columns — the keyed-input convenience the compiler copies onto the edge.
        assert len(edges) == len(vs) - 1
        for i, e in enumerate(edges):
            assert (e.src, e.dst, e.dst_input_port) == (vs[i].id, vs[i + 1].id, 0)
            assert e.key_columns == vs[i + 1].key_columns

        # Explicit edges override synthesis: give the same graph an edge list whose key_columns differ from
        # the vertices' convenience, and _logical_edges must return that list verbatim (the edge wins).
        override = tuple(
            LogicalEdge(
                vs[i].id, vs[i + 1].id, 0, key_choices[int(rng.integers(1, len(key_choices)))]
            )
            for i in range(len(vs) - 1)
        )
        explicit_graph = LogicalGraph(vs, override)
        assert _logical_edges(explicit_graph) == override
