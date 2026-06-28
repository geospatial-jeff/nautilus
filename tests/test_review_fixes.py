"""Regression tests for the Stage-2 review fixes (see CODE_REVIEW.md).

Each test pins a specific behavioral fix so the footgun it closed cannot silently return.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from nautilus import from_batches, run
from nautilus.api import LogicalVertex
from nautilus.compile import compile_graph
from nautilus.compile.plan import KeyGroupSpec, RoundRobinSpec
from nautilus.operators import InMemorySource, KeyedCount
from nautilus.runtime.parallel import Stage, graph_from_stages

# --- C91: from_batches accepts a raw RecordBatch; unknown frames fail loudly --------------------


def test_from_batches_wraps_raw_record_batch():
    rb = pa.record_batch({"x": [1, 2, 3]})
    result = run(from_batches(rb), [])
    assert result.to_pylist() == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_from_batches_rejects_non_frame_non_batch():
    with pytest.raises(TypeError):
        from_batches(123)  # type: ignore[arg-type]


def test_in_memory_source_rejects_non_frame():
    with pytest.raises(TypeError):
        InMemorySource(["not a frame"])  # type: ignore[list-item]


# --- C32: an empty key_columns tuple is rejected, not silently downgraded to keyless ------------


def test_empty_key_columns_rejected():
    with pytest.raises(ValueError, match="key_columns"):
        LogicalVertex(
            id="op0", factory=lambda: KeyedCount("w"), kind="one_input", parallelism=2,
            key_columns=(),
        )


# --- C93: graph_from_stages keys from the operator, never silently round-robins a keyed op ------


def _edge(plan, src, dst):
    return next(e for e in plan.edges if e.src_operator_id == src and e.dst_operator_id == dst)


def test_graph_from_stages_keys_from_operator_declaration():
    # A keyed operator at parallelism 2 with NO explicit Stage.key_columns must still shuffle by key.
    graph = graph_from_stages(from_batches(), [Stage(lambda: KeyedCount("word"), 2)])
    plan = compile_graph(graph)
    assert isinstance(_edge(plan, "source", "op0").spec, KeyGroupSpec)


def test_graph_from_stages_explicit_keys_must_match_operator():
    with pytest.raises(ValueError, match="disagrees"):
        graph_from_stages(from_batches(), [Stage(lambda: KeyedCount("word"), 2, ["other"])])


def test_graph_from_stages_keyless_op_still_round_robins():
    from nautilus.operators import MapBatch

    graph = graph_from_stages(from_batches(), [Stage(lambda: MapBatch(lambda b: b), 2)])
    plan = compile_graph(graph)
    assert isinstance(_edge(plan, "source", "op0").spec, RoundRobinSpec)


# --- C56: a recorder may only write metrics of its own owner -----------------------------------


def test_recorder_owner_gate():
    from nautilus.telemetry import Owner, TelemetryConfig, Tier, make_recorder

    cfg = TelemetryConfig(tier=Tier.COUNTERS)
    engine = make_recorder(operator_id="op0", op_class="X", kind="one_input", config=cfg)
    author = make_recorder(
        operator_id="op0", op_class="X", kind="one_input", config=cfg, owner=Owner.AUTHOR
    )
    engine.counter("operator.rows_out", operator_id="op0", subtask_index=0)  # engine metric: ok
    author.counter("window.fires", operator_id="op0")  # author metric: ok
    with pytest.raises(KeyError):  # author recorder may not write an engine key
        author.counter("operator.rows_out", operator_id="op0", subtask_index=0)
    with pytest.raises(KeyError):  # engine recorder may not write an author key
        engine.counter("window.fires", operator_id="op0")
