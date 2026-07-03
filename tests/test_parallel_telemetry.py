"""A parallel run's report groups and rolls up by subtask, and both digests diverge between parallelism 1 and a parallel run.

The divergence is structural, not a function edit: a parallel run emits one ``OperatorStats`` per
subtask (``subtask_index`` numbered from 0) and one ``Edge`` row per downstream instance on each
fan-out connection (distinct ``channel_index``),
which the digests already fold in. ``build_report`` groups by ``(operator_id, subtask_index, node)``; in
one process every row shares node ``local``, so the grouping is one row per subtask, as here.
"""

from __future__ import annotations

from nautilus.core.records import EOS_FRAME
from nautilus.driver.local import run_local_chain
from nautilus.operators import InMemorySource, KeyedCount
from nautilus.testing import TestClock, data, op_counter

WORDS = [["the", "cat", "sat"], ["the", "dog", "ran"], ["the", "cat", "the", "cat", "fox"]]


def _src() -> InMemorySource:
    return InMemorySource([data(word=w) for w in WORDS] + [EOS_FRAME])


async def _report(q: int):
    """The telemetry of a single keyed-count stage run at parallelism ``q`` (op id ``op0``)."""
    result = await run_local_chain(_src(), [KeyedCount("word")], parallelism=q, clock=TestClock())
    return result.telemetry


def _subtasks(rep, op_id: str) -> list[int]:
    return sorted(o.subtask_index for o in rep.operators if o.operator_id == op_id)


async def test_digests_diverge_p1_vs_pn() -> None:
    r1 = await _report(1)
    rn = await _report(3)
    assert r1.structural_digest() != rn.structural_digest()
    assert r1.meta.config_digest != rn.meta.config_digest


async def test_structural_digest_stable_across_runs_at_pn() -> None:
    digests = set()
    for _ in range(50):
        digests.add((await _report(3)).structural_digest())
    assert len(digests) == 1, f"parallel structural digest is not deterministic: {digests}"


async def test_per_instance_rollup_sums_to_serial() -> None:
    serial = (await run_local_chain(_src(), [KeyedCount("word")], clock=TestClock())).telemetry
    par = await _report(3)
    # exactly one OperatorStats row per instance of the logical operator, numbered from 0
    assert _subtasks(par, "op0") == [0, 1, 2]
    # the per-subtask structural counters sum to the serial totals
    assert op_counter(par, "op0", "operator.rows_in") == op_counter(
        serial, "op0", "operator.rows_in"
    )
    assert op_counter(par, "op0", "operator.rows_out") == op_counter(
        serial, "op0", "operator.rows_out"
    )
    assert par.summary.total_rows_out == serial.summary.total_rows_out
    assert par.summary.total_rows_in == serial.summary.total_rows_in


async def test_edge_conservation_over_channel_index() -> None:
    par = await _report(3)
    fanout = [e for e in par.edges if e.src_operator_id == "source" and e.dst_operator_id == "op0"]
    # one EdgeStats per downstream channel_index, summing to the conserved row total
    assert sorted(e.channel_index for e in fanout) == [0, 1, 2]
    sent = sum(e.rows_sent_total for e in fanout)
    assert sent == op_counter(par, "source", "operator.rows_out")
    assert sent == op_counter(par, "op0", "operator.rows_in")


async def test_topology_carries_num_subtasks_and_q_edges() -> None:
    par = await _report(3)
    op0 = next(n for n in par.topology.nodes if n.operator_id == "op0")
    assert op0.num_subtasks == 3 and op0.subtask_index == 0
    fanout = [
        e
        for e in par.topology.edges
        if e.src_operator_id == "source" and e.dst_operator_id == "op0"
    ]
    assert len(fanout) == 3
    assert {e.partitioner for e in fanout} == {
        "KeyGroupPartitioner"
    }  # keyed shuffle via key groups
    assert sorted(e.channel_index for e in fanout) == [0, 1, 2]


async def test_per_operator_summary_is_one_row_per_subtask() -> None:
    # RunSummary.per_operator ships one OperatorSummary per subtask in one process (node is constant):
    # one row per subtask of a parallel operator, summing to the conserved totals.
    serial = (await run_local_chain(_src(), [KeyedCount("word")], clock=TestClock())).telemetry
    par = await _report(3)
    op0_rows = [row for row in par.summary.per_operator if row.operator_id == "op0"]
    assert len(op0_rows) == 3  # one summary row per subtask
    serial_op0 = next(row for row in serial.summary.per_operator if row.operator_id == "op0")
    assert sum(row.rows_out_total for row in op0_rows) == serial_op0.rows_out_total


async def test_p1_parallel_equals_linear_structural_digest() -> None:
    # parallelism-1 green-path invariance: the parallel runner with everything at parallelism 1
    # reproduces the linear run's structural identity exactly (same operator rows, same
    # single-channel edges).
    linear = (await run_local_chain(_src(), [KeyedCount("word")], clock=TestClock())).telemetry
    par1 = await _report(1)
    assert par1.structural_digest() == linear.structural_digest()
