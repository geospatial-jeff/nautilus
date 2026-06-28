"""The per-worker executor: run one worker's slice of a :class:`~nautilus.compile.plan.PhysicalPlan`.

``execute`` consumes a plan and, for every instance this worker hosts, builds the actor — wiring its
:class:`~nautilus.runtime.mailbox.Mailbox` from the inbound channels and its
:class:`~nautilus.runtime.actor.Output` from the outbound channels, all obtained from the injected
:class:`~nautilus.runtime.connector.Connector` — instantiates a fresh
:class:`~nautilus.runtime.partition.Partitioner` from each edge's spec, runs the actor loops in one
TaskGroup, and returns raw recorder snapshots plus the sink's collected batches.

It is transport-agnostic and report-free by construction. It never names a socket or a channel
implementation — the Connector resolves those — so the same plan slice runs unchanged in one process or
across workers. And it never builds a :class:`~nautilus.telemetry.report.RunReport`: it returns
:class:`~nautilus.telemetry.model.InstanceSnapshot`\\ s, and the boundary
(:mod:`nautilus.runtime.run`, or the Stage 2 coordinator) aggregates them. An import-linter contract
forbids this module from importing the report layer, so report assembly can never reach the data path.

It owns its registry and builds two recorders per one-input instance — the actor's built-in recorder
and a separate ``ctx.metrics`` recorder for operator-author metrics — so the single-writer-per-recorder
invariant holds, exactly as the legacy runners do.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from time import perf_counter_ns
from typing import cast

import pyarrow as pa

from nautilus.compile.plan import (
    ForwardSpec,
    KeyGroupSpec,
    PartitionerSpec,
    PhysicalEdge,
    PhysicalOperator,
    PhysicalPlan,
    RoundRobinSpec,
)
from nautilus.core.operator import OneInputOperator, OperatorContext, SourceOperator
from nautilus.core.records import EOS, Batch
from nautilus.core.time import Clock, SystemClock
from nautilus.runtime.actor import Output, run_source, run_transform
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.runtime.connector import ChannelId, Connector, Deployment
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.partition import Forward, KeyGroupPartitioner, Partitioner, RoundRobin
from nautilus.telemetry import (
    Owner,
    Recorder,
    RecorderRegistry,
    TelemetryConfig,
    Tier,
    make_recorder,
)
from nautilus.telemetry.model import InstanceSnapshot


@dataclass(frozen=True, slots=True)
class ExecuteResult:
    """What one worker returns: every registered recorder's snapshot, and the batches its sink
    collected (empty on a worker that does not host the sink)."""

    snapshots: list[InstanceSnapshot]
    sink_batches: list[pa.RecordBatch]


def partitioner_from_spec(spec: PartitionerSpec) -> Partitioner:
    """Instantiate a fresh runtime partitioner from a plan spec. Fresh per call so each
    :class:`~nautilus.runtime.actor.Output` owns its own :class:`RoundRobin` cursor — the rotation
    state the plan deliberately never carries."""
    if isinstance(spec, ForwardSpec):
        return Forward()
    if isinstance(spec, KeyGroupSpec):
        return KeyGroupPartitioner(spec.key_columns, spec.group_table)
    if isinstance(spec, RoundRobinSpec):
        return RoundRobin()
    raise ValueError(f"no runtime partitioner for spec {spec!r}")


async def _collect_sink(mailbox: Mailbox, out: list[pa.RecordBatch], recorder: Recorder) -> None:
    """The collecting sink: drain every input to EOS, appending data batches. Watermarks / idle /
    active are ignored — the sink has no downstream."""
    rows_in = recorder.counter("operator.rows_in", operator_id="sink", subtask_index=0)
    batches_in = recorder.counter("operator.batches_in", operator_id="sink", subtask_index=0)
    input_wait = recorder.counter("edge.input_wait_micros", operator_id="sink")
    try:
        while not mailbox.exhausted:
            t0 = perf_counter_ns()
            i, frame = await mailbox.get()
            input_wait.add((perf_counter_ns() - t0) // 1000)
            if isinstance(frame, Batch):
                out.append(frame.data)
                batches_in.add(1)
                rows_in.add(frame.num_rows)
            elif isinstance(frame, EOS):
                recorder.incr("eos.received", 1, operator_id="sink", input_index=i)
                mailbox.close_input(i)
    finally:
        mailbox.close()  # cancel any recv still armed if the sink unwound mid-fan-in (fail-fast)
    decoded = (
        mailbox.decode_micros()
    )  # inbound Arrow IPC decode for a sink behind a cross-worker edge
    if decoded:
        recorder.incr("transport.decode_micros", decoded, operator_id="sink")


async def execute(
    plan: PhysicalPlan,
    connector: Connector,
    deployment: Deployment,
    *,
    capacity: int = DEFAULT_CAPACITY,
    clock: Clock | None = None,
    config: TelemetryConfig | None = None,
    registry: RecorderRegistry | None = None,
) -> ExecuteResult:
    """Run the instances this worker hosts to completion and return their snapshots + sink batches."""
    clk = clock or SystemClock()
    cfg = config or TelemetryConfig(clock=clk)
    registry = registry or RecorderRegistry()

    width = {op.operator_id: op.parallelism for op in plan.operators}
    out_edge: dict[str, PhysicalEdge] = {e.src_operator_id: e for e in plan.edges}
    in_edge: dict[str, PhysicalEdge] = {e.dst_operator_id: e for e in plan.edges}

    def rec(
        operator_id: str, op_class: str, kind: str, subtask: int, owner: Owner = Owner.ENGINE
    ) -> Recorder:
        return registry.register(
            make_recorder(
                operator_id=operator_id,
                op_class=op_class,
                kind=kind,
                subtask_index=subtask,
                node=deployment.node,
                config=cfg,
                owner=owner,
            )
        )

    async def build_output(operator_id: str, subtask: int, recorder: Recorder) -> Output:
        edge = out_edge[operator_id]
        fanout = width[edge.dst_operator_id]
        channels = [
            await connector.outbound(ChannelId(operator_id, subtask, edge.dst_operator_id, d))
            for d in range(fanout)
        ]
        return Output(
            channels,
            partitioner_from_spec(edge.spec),
            recorder=recorder,
            edge_src=operator_id,
            edge_dst=edge.dst_operator_id,
            capacity=capacity,
        )

    async def build_mailbox(operator_id: str, subtask: int) -> Mailbox:
        edge = in_edge[operator_id]
        fanin = width[edge.src_operator_id]
        channels = [
            await connector.inbound(ChannelId(edge.src_operator_id, u, operator_id, subtask))
            for u in range(fanin)
        ]
        return Mailbox(channels)

    # The outermost try begins before wiring so connector.close() runs even if Phase A/B raises (a
    # dial/accept failure on a cross-worker edge): a partial mesh is still abortively torn down.
    sink_batches: list[pa.RecordBatch] = []
    try:
        # --- Phase A: instantiate operators and wire every OUTBOUND edge -------------------------
        # Dialing an outbound edge completes once the peer's listener is bound; it never blocks on the
        # peer's accept. Doing every outbound before any inbound therefore makes the cross-worker
        # connect deadlock-free even on a bidirectional mesh — a worker that accepted before it dialed
        # could wait on a peer that is itself waiting to be dialed.
        hosted: list[tuple[PhysicalOperator, int]] = []
        instances: dict[tuple[str, int], object] = {}
        outputs: dict[tuple[str, int], Output] = {}
        op_recorders: dict[tuple[str, int], Recorder] = {}
        metrics_recorders: dict[tuple[str, int], Recorder] = {}
        for op in plan.operators:
            for subtask in range(op.parallelism):
                if not deployment.hosts(op.operator_id, subtask):
                    continue
                key = (op.operator_id, subtask)
                hosted.append((op, subtask))
                op_recorders[key] = rec(op.operator_id, op.op_class, op.kind, subtask)
                if op.kind == "sink":
                    continue  # the sink has no outbound edge; its mailbox is wired in phase B
                instances[key] = _instantiate(op)
                outputs[key] = await build_output(op.operator_id, subtask, op_recorders[key])
                if op.kind == "one_input":
                    # A SEPARATE recorder for operator-author custom metrics (ctx.metrics), preserving
                    # the single-writer invariant; both carry the same (operator_id, subtask) and merge
                    # at build. owner=AUTHOR so it can only write author-owned metrics, never engine keys.
                    metrics_recorders[key] = rec(
                        op.operator_id, op.op_class, op.kind, subtask, owner=Owner.AUTHOR
                    )

        # --- Phase B: wire every INBOUND mailbox and assemble the actor coroutines ---------------
        # Each inbound accept resolves as its producer dials in its own phase A. A mailbox is built with
        # its FULL local+remote input set before its actor starts, so WatermarkTracker(n) and the
        # all-inputs-EOS termination check see every input.
        coros = []
        for op, subtask in hosted:
            key = (op.operator_id, subtask)
            if op.kind == "source":
                ctx = OperatorContext(
                    op.operator_id, subtask_index=subtask, num_subtasks=op.parallelism, clock=clk
                )
                coros.append(
                    run_source(
                        cast(SourceOperator, instances[key]),
                        ctx,
                        [outputs[key]],
                        recorder=op_recorders[key],
                    )
                )
            elif op.kind == "one_input":
                mailbox = await build_mailbox(op.operator_id, subtask)
                ctx = OperatorContext(
                    op.operator_id,
                    subtask_index=subtask,
                    num_subtasks=op.parallelism,
                    clock=clk,
                    metrics=metrics_recorders[key],
                )
                coros.append(
                    run_transform(
                        cast(OneInputOperator, instances[key]),
                        ctx,
                        mailbox,
                        [outputs[key]],
                        recorder=op_recorders[key],
                    )
                )
            elif op.kind == "sink":
                mailbox = await build_mailbox(op.operator_id, subtask)
                coros.append(_collect_sink(mailbox, sink_batches, op_recorders[key]))
            else:
                raise ValueError(f"unknown operator kind {op.kind!r} for {op.operator_id!r}")

        # The hardware sampler runs OUTSIDE the data TaskGroup (as in the legacy runners) so it can
        # neither delay completion nor cancel the data tasks if a psutil call raises. Each worker
        # samples itself, attributing its readings to its own node.
        sampler = None
        sampler_task: asyncio.Task[None] | None = None
        if cfg.tier > Tier.OFF:
            from nautilus.telemetry.system import SystemSampler, make_system_recorder

            proc_rec = registry.register(make_system_recorder(cfg, node=deployment.node))
            # This worker's placement fact: how many operator instances it hosts. Recorded on the
            # per-process recorder (independent of system sampling) so it lands in this node's row.
            proc_rec.set_gauge("placement.instances_per_worker", len(hosted), node=deployment.node)
            if cfg.sample_system:
                sampler = SystemSampler(
                    proc_rec, node=deployment.node, interval_micros=cfg.sample_interval_micros
                )
                sampler_task = asyncio.create_task(sampler.run())

        try:
            async with asyncio.TaskGroup() as tg:
                for coro in coros:
                    tg.create_task(coro)
        finally:
            if sampler_task is not None and sampler is not None:
                sampler_task.cancel()  # cancel FIRST so no pending tick fires after the final reading
                with contextlib.suppress(asyncio.CancelledError):
                    await sampler_task
                sampler.sample_once(sample_cpu=False)  # one final, fully-guarded reading
        # Clean stop: graceful symmetric teardown — drain outbound edges and close inbound edges
        # concurrently (the Connector does the one gather). Symmetric so every worker emits its FIN at
        # once: sequential finish-then-close circular-waits on a bidirectional mesh and eats the full
        # drain timeout. On a TaskGroup failure this is skipped — the exception propagates straight to
        # the abortive close() in finally, so a peer's recv() raises promptly instead of draining.
        await connector.finish()
    finally:
        await connector.close()  # always: abortive close (idempotent) — never leaves a channel open

    return ExecuteResult(registry.snapshot_all(), sink_batches)


def _instantiate(op: PhysicalOperator) -> object:
    """Build a fresh operator from its factory. Only the synthesized sink has no factory, and the
    executor never calls this for the sink — so a ``None`` here is a plan/executor bug, not user error.
    """
    if op.factory is None:
        raise ValueError(f"operator {op.operator_id!r} has no factory to build")
    return op.factory()
