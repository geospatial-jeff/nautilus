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
from nautilus.driver.parallel import Stage, graph_from_stages
from nautilus.operators import InMemorySource, KeyedCount, KeyedTumblingSum, Tokenize

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
            id="op0",
            factory=lambda: KeyedCount("w"),
            kind="one_input",
            parallelism=2,
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


# --- C94: run() takes in-process parallelism ---------------------------------------------------


def test_run_with_in_process_parallelism():
    from nautilus import KeyedCount, Tokenize

    src = from_batches(pa.record_batch({"line": ["a b a", "b c a"]}))
    result = run(src, [Tokenize("line", "word"), KeyedCount("word")], parallelism=3)
    counts = {r["word"]: r["count"] for r in result.to_pylist()}
    assert counts == {"a": 3, "b": 2, "c": 1}  # keyed shuffle never splits a key across instances


# --- C95: OperatorContext enumerates/clears keyed state without the raw backend -----------------


def test_operator_context_entries_and_clear():
    from nautilus.core.operator import OperatorContext
    from nautilus.state import KeyContext

    ctx = OperatorContext("op0")
    ctx.value_state("s", KeyContext(("a",))).update(1)
    ctx.value_state("s", KeyContext(("b",), "ns")).update(2)
    assert {(kc.key, kc.namespace): v for kc, v in ctx.entries("s")} == {
        (("a",), None): 1,
        (("b",), "ns"): 2,
    }
    ctx.clear_state("s", KeyContext(("a",)))
    assert {kc.key for kc, _ in ctx.entries("s")} == {("b",)}


# --- C44 / C30: boundary validation ------------------------------------------------------------


def test_deploy_rejects_nonpositive_workers():
    from nautilus.cluster import deploy
    from nautilus.operators import MapBatch

    graph = graph_from_stages(from_batches(), [Stage(lambda: MapBatch(lambda b: b))])
    with pytest.raises(ValueError, match="num_workers"):
        deploy(graph, num_workers=0)


def test_key_groups_without_keyed_edge_rejected():
    from nautilus.operators import MapBatch

    graph = graph_from_stages(from_batches(), [Stage(lambda: MapBatch(lambda b: b), 2)])
    with pytest.raises(ValueError, match="key_groups"):
        compile_graph(graph, key_groups=4)


# --- C31: a spec's partitioner_name is the name of the runtime partitioner it builds ------------


def test_partitioner_name_matches_runtime_class():
    from nautilus.compile.plan import ForwardSpec, KeyGroupSpec, RoundRobinSpec
    from nautilus.runtime.execute import partitioner_from_spec

    for spec in (ForwardSpec(), RoundRobinSpec(), KeyGroupSpec(("k",), (0,))):
        assert spec.partitioner_name == type(partitioner_from_spec(spec)).__name__


# --- C70: the report query helpers return descending-sorted derived ratios ---------------------


def test_report_occupancy_and_rows_per_sec_queries():
    from nautilus import KeyedCount, Tokenize

    rep = run(
        from_batches(pa.record_batch({"line": ["a b", "a"]})),
        [Tokenize("line", "word"), KeyedCount("word")],
    ).telemetry
    for ranked in (rep.by_occupancy(), rep.by_rows_per_sec()):
        assert ranked and all(len(t) == 2 for t in ranked)
        values = [v for _, v in ranked]
        assert values == sorted(values, reverse=True)  # highest first


# --- C112: in-process parallel fail-fast surfaces the error instead of hanging -----------------


async def test_in_process_parallel_fail_fast():
    import asyncio

    from nautilus.core.operator import Collector, OneInputOperator
    from nautilus.driver.run import run_plan

    class Boom(OneInputOperator):
        def process(self, batch, out: Collector) -> None:
            raise RuntimeError("boom")

    graph = graph_from_stages(
        from_batches(pa.record_batch({"k": [1, 2, 3]})), [Stage(lambda: Boom(), 2)]
    )
    with pytest.raises((RuntimeError, ExceptionGroup)):
        await asyncio.wait_for(run_plan(graph), timeout=10)


# ============================================================================================
# Round-2 review fixes
# ============================================================================================


# --- R38/R39: keyed operators preserve the input key/value column types ------------------------


def test_keyed_tumbling_sum_preserves_float_values():
    from nautilus.testing import data, wm
    from nautilus.windows import TumblingEventTimeWindows

    src = from_batches(data(key=["a", "a"], val=[1.5, 2.0], ts=[1, 2]), wm(10))
    res = run(src, [KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))])
    rows = res.to_pylist()
    assert rows[0]["sum"] == 3.5  # not truncated to int 3
    assert pa.types.is_floating(res.to_table().schema.field("sum").type)


def test_keyed_count_preserves_key_type():
    from nautilus.testing import data

    src = from_batches(data(k=pa.array([1, 1, 2], pa.int32())))
    res = run(src, [KeyedCount("k")])
    assert res.to_table().schema.field("k").type == pa.int32()  # not widened to int64


# --- R1: the windowing operator reads timestamp columns as microseconds ------------------------


def test_to_epoch_micros_normalizes_units():
    from nautilus.core.time import to_epoch_micros

    assert to_epoch_micros(pa.array([1000], pa.timestamp("ms"))).to_pylist() == [1_000_000]
    assert to_epoch_micros(pa.array([5], pa.int64())).to_pylist() == [5]


# --- R14 / R17: IR + compiler validation -------------------------------------------------------


def test_source_vertex_rejects_key_columns():
    with pytest.raises(ValueError, match="source"):
        LogicalVertex(id="source", factory=lambda: None, kind="source", key_columns=("k",))


def test_key_groups_upper_bound_rejected():
    graph = graph_from_stages(from_batches(), [Stage(lambda: KeyedCount("w"), 2)])
    with pytest.raises(ValueError, match="exceeds the maximum"):
        compile_graph(graph, key_groups=10**6)


# --- R65 / R70: runtime fail-loud paths --------------------------------------------------------


async def test_get_on_exhausted_mailbox_raises():
    from nautilus.runtime.channel import InProcChannel
    from nautilus.runtime.mailbox import Mailbox

    mb = Mailbox([InProcChannel(4)])
    mb.close_input(0)
    with pytest.raises(RuntimeError):
        await mb.get()


async def test_transform_raises_on_unhandled_frame():
    from nautilus.core.records import Barrier
    from nautilus.operators import MapBatch
    from nautilus.testing import data, run_ops

    # Barrier is a reserved control frame no operator handles — it must fail loudly, not vanish.
    with pytest.raises(BaseException):  # noqa: B017,PT011 (TaskGroup wraps the TypeError)
        await run_ops([data(x=[1]), Barrier(1)], MapBatch(lambda b: b))


# --- R35 / R67: parallel telemetry surface -----------------------------------------------------


def test_json_summary_rows_carry_subtask_and_node():
    import json

    rep = run(
        from_batches(pa.record_batch({"line": ["a b", "a"]})),
        [Tokenize("line", "word"), KeyedCount("word")],
        parallelism=2,
    ).telemetry
    per_op = json.loads(rep.to_json())["summary"]["per_operator"]
    assert per_op and all("subtask_index" in s and "node" in s for s in per_op)


def test_run_parallelism_builds_n_instances():
    rep = run(
        from_batches(pa.record_batch({"line": ["a b c", "a b", "a"]})),
        [Tokenize("line", "word"), KeyedCount("word")],
        parallelism=3,
    ).telemetry
    node = next(n for n in rep.topology.nodes if n.operator_id == "op1")
    assert node.num_subtasks == 3  # the keyed operator actually ran at parallelism 3
