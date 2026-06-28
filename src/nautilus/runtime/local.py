"""A single-process, in-memory chain runner and telemetry boundary.

Wires ``source -> transform0 -> ... -> transformN -> sink`` with bounded :class:`InProcChannel` s on
one event loop. As the telemetry boundary it owns the :class:`RecorderRegistry`: it builds one recorder
per actor, runs them, aggregates their snapshots into a :class:`RunReport`, and returns a
:class:`RunResult` (batches + telemetry). It may import the report layer; the per-record actors and
channels may not. Stage 2 replaces this with the multi-process deployer; the operator and channel
interfaces stay identical.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import platform
from time import perf_counter_ns
from typing import Any

import pyarrow as pa

import nautilus
from nautilus.core.operator import OneInputOperator, OperatorContext, SourceOperator
from nautilus.core.records import EOS, Batch, Frame
from nautilus.core.time import Clock, SystemClock
from nautilus.runtime.actor import Output, run_source, run_transform
from nautilus.runtime.channel import DEFAULT_CAPACITY, Channel, InProcChannel
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.partition import Forward
from nautilus.runtime.result import RunResult
from nautilus.telemetry import (
    Owner,
    Recorder,
    RecorderRegistry,
    TelemetryConfig,
    Tier,
    make_recorder,
)
from nautilus.telemetry.report import (
    Edge,
    NullSink,
    OperatorNode,
    RunMeta,
    Sink,
    Topology,
    build_report,
)


async def _collect(channel: Channel, out: list[pa.RecordBatch], recorder: Recorder) -> None:
    rows_in = recorder.counter("operator.rows_in", operator_id="sink", subtask_index=0)
    batches_in = recorder.counter("operator.batches_in", operator_id="sink", subtask_index=0)
    input_wait = recorder.counter("edge.input_wait_micros", operator_id="sink")
    while True:
        t0 = perf_counter_ns()
        frame: Frame = await channel.recv()
        input_wait.add((perf_counter_ns() - t0) // 1000)
        if isinstance(frame, Batch):
            out.append(frame.data)
            batches_in.add(1)
            rows_in.add(frame.num_rows)
        elif isinstance(frame, EOS):
            recorder.incr("eos.received", 1, operator_id="sink", input_index=0)
            return


def build_topology(
    source: SourceOperator, transforms: list[OneInputOperator], capacity: int
) -> Topology:
    """Build the static source→ops→sink topology. Public so the live server builds the same topology
    the run does."""
    nodes = [OperatorNode("source", type(source).__name__, "source")]
    nodes += [
        OperatorNode(f"op{k}", type(op).__name__, "one_input") for k, op in enumerate(transforms)
    ]
    nodes.append(OperatorNode("sink", "CollectSink", "sink"))

    edges: list[Edge] = []
    prev = "source"
    for k in range(len(transforms)):
        edges.append(Edge(prev, f"op{k}", 0, "Forward", capacity))
        prev = f"op{k}"
    edges.append(Edge(prev, "sink", 0, "Forward", capacity))
    return Topology(tuple(nodes), tuple(edges))


def config_digest(topology: Topology, config: TelemetryConfig, capacity: int) -> str:
    canonical = {
        "capacity": capacity,
        "tier": int(config.tier),
        "nodes": [[n.operator_id, n.op_class, n.kind] for n in topology.nodes],
        "edges": [[e.src_operator_id, e.dst_operator_id, e.channel_index] for e in topology.edges],
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def make_run_meta(
    *,
    run_id: str,
    started_at: int,
    ended_at: int,
    wall_micros: int,
    clk: Clock,
    topology: Topology,
    config: TelemetryConfig,
    capacity: int,
) -> RunMeta:
    """Assemble a :class:`RunMeta`. Public so the live server can build a point-in-time meta mid-run."""
    return RunMeta(
        run_id=run_id,
        started_at_micros=started_at,
        ended_at_micros=ended_at,
        wall_micros=wall_micros,
        clock_kind=type(clk).__name__,
        nautilus_version=nautilus.__version__,
        python_version=platform.python_version(),
        config_digest=config_digest(topology, config, capacity),
        capacity=capacity,
    )


async def run_local_chain(
    source: SourceOperator,
    transforms: list[OneInputOperator],
    *,
    capacity: int = DEFAULT_CAPACITY,
    clock: Clock | None = None,
    telemetry: TelemetryConfig | None = None,
    sink: Sink | None = None,
    registry: RecorderRegistry | None = None,
) -> RunResult:
    """Run a linear pipeline to completion; return the final-stage batches plus a telemetry report.

    Pass an external ``registry`` to read snapshots live while the run is in flight (the live server
    does this); the default creates its own.
    """
    clk = clock or SystemClock()
    config = telemetry or TelemetryConfig(clock=clk)
    out_sink = sink or NullSink()
    registry = registry or RecorderRegistry()
    topology = build_topology(source, transforms, capacity)

    n = len(transforms)
    stage_in = [InProcChannel(capacity) for _ in range(n)]
    final = InProcChannel(capacity)

    def rec(operator_id: str, op_class: str, kind: str, owner: Owner = Owner.ENGINE) -> Recorder:
        return registry.register(
            make_recorder(
                operator_id=operator_id, op_class=op_class, kind=kind, config=config, owner=owner
            )
        )

    first_dst = "op0" if n else "sink"
    src_rec = rec("source", type(source).__name__, "source")
    src_outputs = [
        Output(
            [stage_in[0] if n else final],
            Forward(),
            recorder=src_rec,
            edge_src="source",
            edge_dst=first_dst,
            capacity=capacity,
        )
    ]

    coros = [
        run_source(source, OperatorContext("source", clock=clk), src_outputs, recorder=src_rec)
    ]
    for k, op in enumerate(transforms):
        op_id = f"op{k}"
        dst = f"op{k + 1}" if k + 1 < n else "sink"
        target = stage_in[k + 1] if k + 1 < n else final
        op_rec = rec(op_id, type(op).__name__, "one_input")
        # A SEPARATE recorder for operator-author custom metrics (ctx.metrics), so the actor and the
        # operator never write the same recorder. Both carry the same operator_id and merge at build.
        # owner=AUTHOR so operator code can only write author-owned metrics, never engine keys.
        metrics_rec = rec(op_id, type(op).__name__, "one_input", owner=Owner.AUTHOR)
        outputs = [
            Output(
                [target],
                Forward(),
                recorder=op_rec,
                edge_src=op_id,
                edge_dst=dst,
                capacity=capacity,
            )
        ]
        coros.append(
            run_transform(
                op,
                OperatorContext(op_id, clock=clk, metrics=metrics_rec),
                Mailbox([stage_in[k]]),
                outputs,
                recorder=op_rec,
            )
        )

    sink_rec = rec("sink", "CollectSink", "sink")
    results: list[pa.RecordBatch] = []
    coros.append(_collect(final, results, sink_rec))

    # The hardware sampler runs OUTSIDE the data TaskGroup so it can neither delay completion nor, if
    # a psutil call were to raise, cancel the data tasks. It is the sole writer of the process recorder.
    sampler = None
    sampler_task: asyncio.Task[None] | None = None
    if config.tier > Tier.OFF and config.sample_system:
        from nautilus.telemetry.system import SystemSampler, make_system_recorder

        proc_rec = registry.register(make_system_recorder(config, node="local"))
        sampler = SystemSampler(
            proc_rec, node="local", interval_micros=config.sample_interval_micros
        )
        sampler_task = asyncio.create_task(sampler.run())

    started_at = clk.now_micros()
    wall0 = perf_counter_ns()
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
    report = build_report(registry.snapshot_all(), meta=meta, topology=topology)
    out_sink.emit_report(report)
    return RunResult(results, report)


def run(source: SourceOperator, transforms: list[OneInputOperator], **kwargs: Any) -> RunResult:
    """Synchronous one-liner: run a bounded pipeline to completion and return its :class:`RunResult`.

    A convenience for top-level / boundary use (scripts, a REPL) that wraps :func:`run_local_chain` in
    :func:`asyncio.run`. It cannot be called from inside a running event loop — ``await
    run_local_chain(...)`` directly there (as the CLI and live server do). Accepts the same keyword
    arguments as :func:`run_local_chain` (``capacity``, ``clock``, ``telemetry``, ``sink``)."""
    return asyncio.run(run_local_chain(source, transforms, **kwargs))
