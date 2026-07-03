"""The single-process boundary runner: compile a graph, run it, build one report.

``run_plan`` is the single-process counterpart to the Stage 2 coordinator: it compiles a
:class:`~nautilus.api.LogicalGraph` to a :class:`~nautilus.compile.plan.PhysicalPlan`, runs it through
:func:`~nautilus.runtime.execute.execute` over an in-process
:class:`~nautilus.runtime.connector.InProcessConnector`, then — as the telemetry boundary — translates
the plan's neutral structure into a :class:`~nautilus.telemetry.report.Topology` and aggregates the
worker's snapshots into a :class:`~nautilus.telemetry.report.RunReport`.

This module, not the executor, imports the report layer: the executor returns raw snapshots and stays
report-free, and the same split (workers emit snapshots; the boundary builds the report) is what the
multi-worker coordinator will reuse. ``run_compiled`` is split out so a caller that already holds a
plan — for example a cloudpickle round-trip test, or a worker handed a plan it never compiled — runs it
without re-compiling.
"""

from __future__ import annotations

from time import perf_counter_ns
from typing import Any

from nautilus.api import LogicalGraph
from nautilus.compile import PhysicalPlan, compile_graph
from nautilus.core.time import Clock, SystemClock
from nautilus.driver.meta import make_run_meta
from nautilus.driver.result import RunResult
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.runtime.connector import Deployment, InProcessConnector
from nautilus.runtime.execute import execute
from nautilus.telemetry import RecorderRegistry, TelemetryConfig
from nautilus.telemetry.report import Edge, NullSink, OperatorNode, Sink, Topology, build_report


def plan_to_topology(plan: PhysicalPlan, capacity: int) -> Topology:
    """Translate a plan's neutral structure into a report :class:`Topology`: one
    :class:`OperatorNode` per operator carrying its instance count, and one :class:`Edge` per physical
    channel (the downstream fan-out) tagged with the spec's partitioner name. This is the telemetry
    boundary's job — the executor and the plan never name a report type."""
    width = {op.operator_id: op.parallelism for op in plan.operators}
    nodes = tuple(
        OperatorNode(op.operator_id, op.op_class, op.kind, num_subtasks=op.parallelism)
        for op in plan.operators
    )
    edges = tuple(
        Edge(e.src_operator_id, e.dst_operator_id, d, e.spec.partitioner_name, capacity)
        for e in plan.edges
        for d in range(width[e.dst_operator_id])
    )
    return Topology(nodes, edges)


async def run_compiled(
    plan: PhysicalPlan,
    *,
    capacity: int = DEFAULT_CAPACITY,
    clock: Clock | None = None,
    telemetry: TelemetryConfig | None = None,
    sink: Sink | None = None,
    registry: RecorderRegistry | None = None,
) -> RunResult:
    """Run an already-compiled plan single-process and return its batches plus a telemetry report."""
    clk = clock or SystemClock()
    config = telemetry or TelemetryConfig(clock=clk)
    out_sink = sink or NullSink()
    registry = registry or RecorderRegistry()
    topology = plan_to_topology(plan, capacity)

    connector = InProcessConnector(capacity)
    deployment = Deployment.single_worker()

    started_at = clk.now_micros()
    wall0 = perf_counter_ns()
    result = await execute(
        plan, connector, deployment, capacity=capacity, clock=clk, config=config, registry=registry
    )
    wall_micros = (perf_counter_ns() - wall0) // 1000
    ended_at = clk.now_micros()

    meta = make_run_meta(
        run_id=config.run_id or f"run-{started_at}",
        started_at=started_at,
        ended_at=ended_at,
        wall_micros=wall_micros,
        clk=clk,
        topology=topology,
        config=config,
        capacity=capacity,
    )
    report = build_report(result.snapshots, meta=meta, topology=topology)
    out_sink.emit_report(report)
    return RunResult(result.sink_batches, report)


async def run_plan(
    graph: LogicalGraph, *, key_groups: int | None = None, **kwargs: Any
) -> RunResult:
    """Compile ``graph`` and run it single-process. ``key_groups`` sets the number of key
    groups the keyed shuffles route through (``None`` defaults each to its parallelism). Otherwise
    accepts the same keyword arguments as :func:`run_compiled` (``capacity``, ``clock``, ``telemetry``,
    ``sink``, ``registry``)."""
    return await run_compiled(compile_graph(graph, key_groups=key_groups), **kwargs)
