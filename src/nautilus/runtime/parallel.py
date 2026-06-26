"""A parallel-topology runner: run each operator as N instances joined by a P×Q channel mesh.

This is the Stage 1.5 counterpart to :func:`~nautilus.runtime.local.run_local_chain`. It runs a
``source → stage0 → … → stageN → sink`` chain where each stage may have a parallelism > 1, wiring a
grid of channels between adjacent layers: every upstream instance routes its batches through a
partitioner to the owning downstream instance (a :class:`~nautilus.runtime.partition.HashPartitioner`
keyed shuffle, or a :class:`~nautilus.runtime.partition.RoundRobin` rebalance), while a
:class:`~nautilus.runtime.mailbox.Mailbox` fans the upstream instances back in per downstream instance.
Control frames (watermark, idle/active, EOS) are broadcast to *every* downstream instance, so an
instance that receives no data rows still advances event time, runs its watermark callbacks, and
forwards EOS to terminate.

The same graph runs unchanged over in-process channels or socket pairs: the channels come from a
:class:`ChannelFactory` (the in-process default here; a socket-pair factory lives in
:mod:`nautilus.transport.mesh`), so ``runtime`` never imports ``transport``.

It is the telemetry boundary (like ``run_local_chain``): it owns the registry, builds one recorder per
instance, and aggregates them into a :class:`RunReport`. ``Stage`` and ``ChannelFactory`` are ephemeral
runtime wiring that the Stage 2 compiler will subsume.
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter_ns

import pyarrow as pa

from nautilus.core.operator import OneInputOperator, OperatorContext, SourceOperator
from nautilus.core.records import EOS, Batch
from nautilus.core.time import Clock, SystemClock
from nautilus.runtime.actor import Output, run_source, run_transform
from nautilus.runtime.channel import DEFAULT_CAPACITY, Channel, InProcChannel
from nautilus.runtime.local import make_run_meta  # reused verbatim (it calls config_digest)
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.partition import Forward, HashPartitioner, Partitioner, RoundRobin
from nautilus.runtime.result import RunResult
from nautilus.telemetry import Recorder, RecorderRegistry, TelemetryConfig, Tier, make_recorder
from nautilus.telemetry.report import (
    Edge,
    NullSink,
    OperatorNode,
    Sink,
    Topology,
    build_report,
)


@dataclass(frozen=True)
class Stage:
    """One logical operator in a parallel chain.

    ``factory`` builds a *fresh* operator (and, through its ``OperatorContext``, a fresh state backend)
    for each of the ``parallelism`` instances; it must be a side-effect-free constructor (resource
    acquisition belongs in ``open()``), because it is also called once more to read the operator class
    name for the topology. ``key_columns`` is the source of truth for the edge *feeding this stage*:
    set, it selects a :class:`HashPartitioner` keyed shuffle on those columns; unset with
    ``parallelism > 1``, a :class:`RoundRobin` rebalance; ``parallelism == 1`` is a :class:`Forward`.
    """

    factory: Callable[[], OneInputOperator]
    parallelism: int = 1
    key_columns: Sequence[str] | None = None


class ChannelFactory(ABC):
    """Builds the channel pairs the mesh is wired from, so the identical graph runs over in-process
    queues or socket pairs. ``pair`` returns ``(send_end, recv_end)``; ``close_all`` tears every
    created channel down once, after the run's :class:`~asyncio.TaskGroup` has joined."""

    @abstractmethod
    async def pair(self, capacity: int) -> tuple[Channel, Channel]: ...

    @abstractmethod
    async def close_all(self) -> None: ...


class InProcFactory(ChannelFactory):
    """The in-process default: one :class:`InProcChannel` whose send and recv ends are the same object.
    ``close_all`` is a genuine no-op — an ``InProcChannel`` has no ``close`` to call."""

    async def pair(self, capacity: int) -> tuple[Channel, Channel]:
        ch = InProcChannel(capacity)
        return ch, ch

    async def close_all(self) -> None:
        return None


def _op_id(layer: int, num_stages: int) -> str:
    """The operator id of a layer: ``0`` is the source, ``num_stages + 1`` is the sink, between them
    layer ``l`` is stage ``l - 1`` (``op{l-1}``)."""
    if layer == 0:
        return "source"
    if layer == num_stages + 1:
        return "sink"
    return f"op{layer - 1}"


def _partitioner_for(parallelism: int, key_columns: Sequence[str] | None) -> Partitioner:
    """Select the partitioner for an edge feeding a stage of the given parallelism. A fresh instance
    per call, so each upstream :class:`Output` owns its own :class:`RoundRobin` cursor."""
    if parallelism < 1:
        raise ValueError(f"parallelism must be >= 1, got {parallelism}")
    if parallelism == 1:
        return Forward()
    if key_columns:
        return HashPartitioner(key_columns)
    return RoundRobin()


def _check_fanout_partitioner(partitioner: Partitioner, parallelism: int, dst_id: str) -> None:
    """Reject a fan-out edge wired with :class:`Forward` at wiring time, rather than letting
    ``Forward.route`` raise on the first emitted batch."""
    if parallelism > 1 and isinstance(partitioner, Forward):
        raise ValueError(
            f"edge into {dst_id} has parallelism {parallelism} but a Forward partitioner; "
            "a fan-out edge needs a HashPartitioner (set key_columns) or RoundRobin"
        )


def build_parallel_topology(
    source: SourceOperator, stages: Sequence[Stage], capacity: int
) -> Topology:
    """Build the parallel topology: one :class:`OperatorNode` per logical operator carrying its
    ``num_subtasks``, and ``Q`` :class:`Edge` rows (distinct ``channel_index``) per connection tagged
    with the routing partitioner's class name. Public so the live server can build the same topology.
    """
    s = len(stages)
    widths = [1] + [stage.parallelism for stage in stages] + [1]

    nodes = [OperatorNode("source", type(source).__name__, "source", num_subtasks=1)]
    nodes += [
        OperatorNode(
            f"op{j}", type(stage.factory()).__name__, "one_input", num_subtasks=stage.parallelism
        )
        for j, stage in enumerate(stages)
    ]
    nodes.append(OperatorNode("sink", "CollectSink", "sink", num_subtasks=1))

    edges: list[Edge] = []
    for c in range(s + 1):
        q = widths[c + 1]
        key_columns = stages[c].key_columns if c < s else None
        part_name = type(_partitioner_for(q, key_columns)).__name__
        src_id, dst_id = _op_id(c, s), _op_id(c + 1, s)
        edges += [Edge(src_id, dst_id, d, part_name, capacity) for d in range(q)]
    return Topology(tuple(nodes), tuple(edges))


async def _collect_parallel(
    mailbox: Mailbox, out: list[pa.RecordBatch], recorder: Recorder
) -> None:
    """Multi-input collect sink: drain every input to EOS (contrast ``_collect``, which returns on the
    first EOS). Watermarks / idle / active are ignored — the sink has no downstream."""
    rows_in = recorder.counter("operator.rows_in", operator_id="sink", subtask_index=0)
    batches_in = recorder.counter("operator.batches_in", operator_id="sink", subtask_index=0)
    input_wait = recorder.counter("edge.input_wait_micros", operator_id="sink")
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


async def run_parallel_chain(
    source: SourceOperator,
    stages: Sequence[Stage],
    *,
    capacity: int = DEFAULT_CAPACITY,
    clock: Clock | None = None,
    telemetry: TelemetryConfig | None = None,
    sink: Sink | None = None,
    registry: RecorderRegistry | None = None,
    factory: ChannelFactory | None = None,
) -> RunResult:
    """Run ``source → stages → sink`` as a parallel mesh to completion; return the collected batches
    plus a telemetry report. An all-``parallelism == 1`` spec degenerates to the linear path."""
    clk = clock or SystemClock()
    config = telemetry or TelemetryConfig(clock=clk)
    out_sink = sink or NullSink()
    registry = registry or RecorderRegistry()
    chan_factory = factory or InProcFactory()
    topology = build_parallel_topology(source, stages, capacity)

    s = len(stages)
    widths = [1] + [stage.parallelism for stage in stages] + [1]

    def key_columns_of(connection: int) -> Sequence[str] | None:
        # Connection c feeds layer c+1; its partitioner is set by that downstream stage's key_columns
        # (None when the downstream is the sink).
        return stages[connection].key_columns if connection < s else None

    # A FRESH partitioner is built per Output below (so each upstream instance owns its own RoundRobin
    # cursor); here we only validate, once per connection, that no fan-out edge got a Forward.
    for c in range(s + 1):
        _check_fanout_partitioner(
            _partitioner_for(widths[c + 1], key_columns_of(c)), widths[c + 1], _op_id(c + 1, s)
        )

    def rec(operator_id: str, op_class: str, kind: str, subtask_index: int) -> Recorder:
        return registry.register(
            make_recorder(
                operator_id=operator_id,
                op_class=op_class,
                kind=kind,
                subtask_index=subtask_index,
                config=config,
            )
        )

    try:
        # The P×Q grid of (send, recv) pairs per connection c; grid[u][d] feeds downstream d from
        # upstream u.
        grids: list[list[list[tuple[Channel, Channel]]]] = []
        for c in range(s + 1):
            p, q = widths[c], widths[c + 1]
            grids.append([[await chan_factory.pair(capacity) for _ in range(q)] for _ in range(p)])

        coros = []

        # Source (layer 0, single instance): routes through connection 0 to layer 1.
        src_rec = rec("source", type(source).__name__, "source", 0)
        src_out = Output(
            [grids[0][0][d][0] for d in range(widths[1])],
            _partitioner_for(widths[1], key_columns_of(0)),
            recorder=src_rec,
            edge_src="source",
            edge_dst=_op_id(1, s),
            capacity=capacity,
        )
        coros.append(
            run_source(
                source,
                OperatorContext("source", subtask_index=0, num_subtasks=1, clock=clk),
                [src_out],
                recorder=src_rec,
            )
        )

        # Each stage (layer l = j + 1): instance i reads connection j, writes connection j + 1.
        for j, stage in enumerate(stages):
            layer = j + 1
            op_id = f"op{j}"
            p_layer = widths[layer]
            for i in range(p_layer):
                op = stage.factory()  # fresh op + state backend per instance
                op_rec = rec(op_id, type(op).__name__, "one_input", i)
                # A SEPARATE recorder for operator-author custom metrics (ctx.metrics), preserving the
                # single-writer invariant; both carry the same (operator_id, subtask_index) and merge.
                metrics_rec = rec(op_id, type(op).__name__, "one_input", i)
                mailbox = Mailbox([grids[j][u][i][1] for u in range(widths[j])])
                outputs = [
                    Output(
                        [grids[layer][i][d][0] for d in range(widths[layer + 1])],
                        _partitioner_for(widths[layer + 1], key_columns_of(layer)),
                        recorder=op_rec,
                        edge_src=op_id,
                        edge_dst=_op_id(layer + 1, s),
                        capacity=capacity,
                    )
                ]
                ctx = OperatorContext(
                    op_id, subtask_index=i, num_subtasks=p_layer, clock=clk, metrics=metrics_rec
                )
                coros.append(run_transform(op, ctx, mailbox, outputs, recorder=op_rec))

        # Sink (last layer, single instance): fans in every instance of the last stage.
        sink_rec = rec("sink", "CollectSink", "sink", 0)
        sink_mailbox = Mailbox([grids[s][u][0][1] for u in range(widths[s])])
        results: list[pa.RecordBatch] = []
        coros.append(_collect_parallel(sink_mailbox, results, sink_rec))

        # The hardware sampler runs OUTSIDE the data TaskGroup (as in run_local_chain) so it can
        # neither delay completion nor cancel the data tasks if a psutil call raises.
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
    finally:
        await chan_factory.close_all()  # after the TaskGroup join: no unread data left to drop

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
