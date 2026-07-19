"""The declarative catalog of nautilus metrics and events.

Every metric and event nautilus emits is declared here exactly once. Instruments are created *by
catalog key*, so an undeclared metric cannot be emitted and the report schema can never drift from the
data. For each number it states what it measures (a fact) and which other metrics relate to it — never
what a value indicates, never a cause, never a remedy. A unit test checks every metric/event name,
meaning, and derivation string against :data:`BANNED_ANALYSIS_WORDS`, so analysis language cannot creep
into the catalog: nautilus records the data; the analysis is done separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class MetricKind(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class Reduction(StrEnum):
    """How a series *should* roll up across instances/subtasks. Advisory metadata exported in the JSON
    catalog and the reference table for report consumers — ``build_report`` does NOT read it: its merge
    is fixed by instrument kind (counters sum, gauges keep last + min/max, histograms add buckets).
    """

    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"


class Stability(StrEnum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"


class Owner(StrEnum):
    """Who is allowed to write a metric. The runtime owns one recorder per actor (``ENGINE``) and a
    separate ``ctx.metrics`` recorder for operator-author metrics (``AUTHOR``); a recorder may only
    write metrics of its own owner. Since the report aggregates engine metrics by name across snapshots,
    this stops an operator from accidentally writing an engine key (e.g. ``operator.rows_out``) via
    ``ctx.metrics`` and inflating the totals."""

    ENGINE = "engine"
    AUTHOR = "author"


class Tier(IntEnum):
    """Verbosity tiers. A metric/event activates once the configured tier reaches its ``min_tier``."""

    OFF = 0
    COUNTERS = 1  # default: counters, gauges, histograms + essential lifecycle/error events
    COUNTERS_PLUS_EVENTS = 2  # + verbose events (eos forwarded)
    FULL = 3  # + byte accounting (the expensive Arrow buffer-size walk)


# Fixed histogram boundaries (upper-inclusive edges). Power-of-two range covers µs..~1s and 1..65k rows.
DURATION_US_BUCKETS: tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    1048576,
)
ROWS_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)
# Channel-fill levels: a typical capacity is small (default 16), so fine low buckets, then headroom.
QUEUE_DEPTH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

#: Words that would turn a fact into a verdict. The catalog must contain none of these (CI-enforced).
BANNED_ANALYSIS_WORDS: frozenset[str] = frozenset(
    {
        "bottleneck",
        "slow",
        "slower",
        "slowest",
        "bug",
        "leak",
        "leaks",
        "oom",
        "hang",
        "hangs",
        "cause",
        "causes",
        "caused",
        "fix",
        "fixes",
        "should",
        "problem",
        "issue",
        "optimize",
        "optimization",
        "regression",
        "culprit",
        "blame",
        "wrong",
        "bad",
        "unhealthy",
        "indicates",
        "implies",
        "suspect",
        "anomaly",
        "degraded",
        "starved",
        "overloaded",
    }
)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    kind: MetricKind
    unit: str
    labels: tuple[str, ...]
    reduction: Reduction
    meaning: str
    relates_to: tuple[str, ...] = ()
    derivation: str | None = None
    since_stage: int = 0
    stability: Stability = Stability.STABLE
    deterministic: bool = False
    min_tier: Tier = Tier.COUNTERS
    boundaries: tuple[int, ...] = ()
    owner: Owner = Owner.ENGINE


@dataclass(frozen=True, slots=True)
class EventSpec:
    name: str
    fields: tuple[str, ...]
    meaning: str
    since_stage: int = 0
    stability: Stability = Stability.STABLE
    min_tier: Tier = Tier.COUNTERS


_OP = ("operator_id", "subtask_index")
_EDGE = ("operator_id", "edge_src", "edge_dst", "channel_index")

METRIC_SPECS: dict[str, MetricSpec] = {
    s.name: s
    for s in [
        # --- operator throughput -------------------------------------------------------------
        MetricSpec(
            "operator.batches_in",
            MetricKind.COUNTER,
            "batches",
            _OP,
            Reduction.SUM,
            "Number of data batches received.",
            deterministic=True,
        ),
        MetricSpec(
            "operator.rows_in",
            MetricKind.COUNTER,
            "rows",
            _OP,
            Reduction.SUM,
            "Sum of num_rows across received batches.",
            relates_to=("operator.rows_out",),
            deterministic=True,
        ),
        MetricSpec(
            "operator.batches_out",
            MetricKind.COUNTER,
            "batches",
            _OP,
            Reduction.SUM,
            "Number of non-empty data batches emitted.",
            deterministic=True,
        ),
        MetricSpec(
            "operator.rows_out",
            MetricKind.COUNTER,
            "rows",
            _OP,
            Reduction.SUM,
            "Sum of num_rows across emitted batches.",
            relates_to=("operator.rows_in",),
            derivation="rows_out / rows_in = selectivity",
            deterministic=True,
        ),
        MetricSpec(
            "operator.bytes_in",
            MetricKind.COUNTER,
            "bytes",
            _OP,
            Reduction.SUM,
            "Approximate Arrow buffer size of received batches (get_total_buffer_size proxy).",
            relates_to=("operator.rows_in",),
            stability=Stability.EXPERIMENTAL,
            min_tier=Tier.FULL,
        ),
        MetricSpec(
            "operator.bytes_out",
            MetricKind.COUNTER,
            "bytes",
            _OP,
            Reduction.SUM,
            "Approximate Arrow buffer size of emitted batches (get_total_buffer_size proxy).",
            relates_to=("operator.rows_out",),
            stability=Stability.EXPERIMENTAL,
            min_tier=Tier.FULL,
        ),
        # --- operator timing -----------------------------------------------------------------
        MetricSpec(
            "operator.process_micros",
            MetricKind.HISTOGRAM,
            "microseconds",
            _OP,  # op_class is already on the snapshot; no need to repeat it per timing sample
            Reduction.SUM,
            "Wall time of one op.process(batch) call, measured with perf_counter_ns.",
            relates_to=("operator.batch_rows",),
            boundaries=DURATION_US_BUCKETS,
        ),
        MetricSpec(
            "operator.on_eos_micros",
            MetricKind.HISTOGRAM,
            "microseconds",
            _OP,
            Reduction.SUM,
            "Wall time of the op.on_eos(out) end-of-stream flush.",
            boundaries=DURATION_US_BUCKETS,
        ),
        MetricSpec(
            "operator.batch_rows",
            MetricKind.HISTOGRAM,
            "rows",
            _OP,
            Reduction.SUM,
            "num_rows of each inbound batch.",
            relates_to=("operator.process_micros",),
            deterministic=True,
            boundaries=ROWS_BUCKETS,
        ),
        MetricSpec(
            "operator.process_calls",
            MetricKind.COUNTER,
            "calls",
            _OP,
            Reduction.SUM,
            "Number of op.process invocations.",
            deterministic=True,
        ),
        MetricSpec(
            "operator.on_eos_calls",
            MetricKind.COUNTER,
            "calls",
            _OP,
            Reduction.SUM,
            "Number of op.on_eos invocations.",
            deterministic=True,
        ),
        # --- edges (producer-owned) ----------------------------------------------------------
        MetricSpec(
            "edge.send_wait_micros",
            MetricKind.COUNTER,
            "microseconds",
            _EDGE,
            Reduction.SUM,
            "Time the sending actor was suspended inside channel.send awaiting capacity.",
            relates_to=("edge.input_wait_micros", "edge.queue_depth", "edge.queue_capacity"),
            derivation="send_wait_micros > 0 = the send awaited",
        ),
        MetricSpec(
            "edge.input_wait_micros",
            MetricKind.COUNTER,
            "microseconds",
            ("operator_id",),
            Reduction.SUM,
            "Time the actor was suspended in mailbox.get awaiting any input.",
            relates_to=("edge.send_wait_micros",),
        ),
        MetricSpec(
            "edge.frames_sent",
            MetricKind.COUNTER,
            "count",
            (*_EDGE, "frame_type"),
            Reduction.SUM,
            "Frames pushed by the producer.",
            relates_to=("operator.rows_out",),
            deterministic=True,
        ),
        MetricSpec(
            "edge.batches_sent",
            MetricKind.COUNTER,
            "batches",
            _EDGE,  # batches are always data frames — no frame_type dimension needed
            Reduction.SUM,
            "Data batches pushed by the producer.",
            deterministic=True,
        ),
        MetricSpec(
            "edge.rows_sent",
            MetricKind.COUNTER,
            "rows",
            _EDGE,
            Reduction.SUM,
            "Rows pushed by the producer.",
            relates_to=("operator.rows_out",),
            deterministic=True,
        ),
        MetricSpec(
            "edge.queue_depth",
            MetricKind.GAUGE,
            "count",
            _EDGE,
            Reduction.MAX,
            "Channel.depth() sampled by the producer after each send (high-water).",
            relates_to=("edge.queue_capacity",),
            derivation="queue_depth / queue_capacity = saturation",
        ),
        MetricSpec(
            "edge.queue_capacity",
            MetricKind.GAUGE,
            "count",
            _EDGE,
            Reduction.LAST,
            "Configured channel capacity.",
            relates_to=("edge.queue_depth",),
            deterministic=True,
        ),
        MetricSpec(
            "edge.queue_depth_hist",
            MetricKind.HISTOGRAM,
            "count",
            _EDGE,
            Reduction.SUM,
            "Distribution of Channel.depth() sampled by the producer after each send. Where "
            "edge.queue_depth gives the high-water level, this gives how often each level occurred — the "
            "share of sends near capacity. In-process channels only (a socket channel reports no depth).",
            relates_to=("edge.queue_depth", "edge.queue_capacity"),
            boundaries=QUEUE_DEPTH_BUCKETS,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "partition.route_micros",
            MetricKind.HISTOGRAM,
            "microseconds",
            ("operator_id", "edge_dst"),
            Reduction.SUM,
            "Wall time of one partitioner.route(batch) call on the sending actor, measured with "
            "perf_counter_ns. Spans key extraction, per-key assignment, and the take into sub-batches; "
            "sits between the operator's process and the downstream send.",
            relates_to=("edge.rows_sent", "edge.send_wait_micros"),
            boundaries=DURATION_US_BUCKETS,
        ),
        # --- end of stream -------------------------------------------------------------------
        MetricSpec(
            "eos.expected",
            MetricKind.GAUGE,
            "count",
            ("operator_id",),
            Reduction.LAST,
            "Number of input channels (mailbox.num_inputs).",
            relates_to=("eos.received",),
            deterministic=True,
        ),
        MetricSpec(
            "eos.received",
            MetricKind.COUNTER,
            "count",
            ("operator_id", "input_index"),
            Reduction.SUM,
            "Number of EOS frames received, written as each one arrives.",
            relates_to=("eos.expected",),
            deterministic=True,
        ),
        # --- state ---------------------------------------------------------------------------
        MetricSpec(
            "state.entries",
            MetricKind.GAUGE,
            "count",
            ("operator_id", "state_name"),
            Reduction.MAX,
            "Count of (key, namespace) entries held in a named state.",
            relates_to=("state.keys",),
            derivation="entries / keys = entries-per-key",
            deterministic=True,
        ),
        MetricSpec(
            "state.keys",
            MetricKind.GAUGE,
            "count",
            ("operator_id", "state_name"),
            Reduction.MAX,
            "Count of distinct keys held in a named state.",
            relates_to=("state.entries",),
            deterministic=True,
        ),
        # --- runtime occupancy ---------------------------------------------------------------
        MetricSpec(
            "runtime.step_micros",
            MetricKind.COUNTER,
            "microseconds",
            _OP,
            Reduction.SUM,
            "Summed wall time the actor spent producing output: a transform's process and on_eos "
            "critical sections, or a source's frame generation (which includes any await a self-pacing "
            "source performs between frames). Accumulated in nanoseconds and reduced to microseconds "
            "once, so a step shorter than a microsecond still counts.",
            relates_to=("runtime.await_count",),
        ),
        MetricSpec(
            "runtime.await_count",
            MetricKind.COUNTER,
            "count",
            _OP,
            Reduction.SUM,
            "Number of awaits the actor performed.",
            relates_to=("runtime.step_micros",),
            deterministic=True,
        ),
        MetricSpec(
            "io.wait_micros",
            MetricKind.COUNTER,
            "microseconds",
            ("operator_id",),
            Reduction.SUM,
            "Wall time a source spent awaiting external I/O, recorded by the source itself via "
            "ctx.io_wait(). A source is the one operator that may await inside its own code, so its "
            "runtime.step_micros counts both its on-CPU frame construction and the awaits it performs "
            "between frames; subtracting this from step_micros leaves the on-CPU time, so a source whose "
            "io.wait_micros is most of its step_micros is I/O-bound, not compute-bound. Zero unless a "
            "source brackets its awaits.",
            relates_to=("runtime.step_micros",),
            owner=Owner.AUTHOR,
        ),
        # --- async I/O stages (an async sink, driven by run_async_sink) ----------------------
        MetricSpec(
            "async.requests",
            MetricKind.COUNTER,
            "count",
            _OP,
            Reduction.SUM,
            "Number of async I/O tasks an async stage completed — one per batch an async sink writes or an "
            "async transform fetches. Recorded by the actor when it reaps the task, not by the awaiting "
            "code.",
            relates_to=("async.request_micros", "async.in_flight"),
            since_stage=6,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "async.request_micros",
            MetricKind.COUNTER,
            "microseconds",
            _OP,
            Reduction.SUM,
            "Summed wall time an async stage's I/O tasks spent awaiting external I/O (each write's or "
            "fetch's perf_counter span). Several tasks run at once, so this sum can exceed the run's wall "
            "time; the gap to wall is the overlap. Distinct from runtime.step_micros, which for an async "
            "stage counts only the actor's own coordination (a transform's integrate/on_eos "
            "self-time), never the awaited I/O.",
            relates_to=("async.requests", "async.in_flight", "runtime.step_micros"),
            since_stage=6,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "async.in_flight",
            MetricKind.GAUGE,
            "count",
            _OP,
            Reduction.MAX,
            "High-water number of async I/O tasks in flight at once on one instance. At most "
            "async.capacity.",
            relates_to=("async.capacity", "async.requests"),
            since_stage=6,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "async.capacity",
            MetricKind.GAUGE,
            "count",
            _OP,
            Reduction.LAST,
            "The configured max_in_flight bound on concurrent async I/O tasks for an async stage — the "
            "ceiling async.in_flight rises to before the actor stops reading (the stage's backpressure).",
            relates_to=("async.in_flight",),
            since_stage=6,
            stability=Stability.EXPERIMENTAL,
            deterministic=True,
        ),
        MetricSpec(
            "async.timeouts",
            MetricKind.COUNTER,
            "count",
            _OP,
            Reduction.SUM,
            "Number of async I/O tasks cancelled for exceeding the stage's per-request timeout_micros. "
            "Zero unless a timeout is configured.",
            relates_to=("async.requests",),
            since_stage=6,
            stability=Stability.EXPERIMENTAL,
        ),
        # --- errors --------------------------------------------------------------------------
        MetricSpec(
            "operator.errors",
            MetricKind.COUNTER,
            "count",
            ("operator_id", "exc_type"),  # op_class is on the snapshot and the operator.error event
            Reduction.SUM,
            "Number of exceptions raised in an operator lifecycle method.",
            deterministic=True,
        ),
        # --- hardware / process resources (sampled periodically by the SystemSampler) --------
        # Process-scoped: CPU/memory are shared across operators in one process, so these are not
        # attributed per-operator. The process is identified by the snapshot's node attribute, so
        # these carry no labels. All non-deterministic and excluded from STRUCTURAL_METRICS.
        MetricSpec(
            "process.cpu_percent",
            MetricKind.GAUGE,
            "percent",
            (),
            Reduction.LAST,
            "psutil.Process.cpu_percent() over the interval since the previous sample, where 100 "
            "equals one fully used CPU core.",
            relates_to=("runtime.loop_lag_micros",),
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "process.rss_bytes",
            MetricKind.GAUGE,
            "bytes",
            (),
            Reduction.LAST,
            "psutil.Process.memory_info().rss: resident set size of this process at the sample.",
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "process.num_fds",
            MetricKind.GAUGE,
            "count",
            (),
            Reduction.LAST,
            "psutil.Process.num_fds(): open file descriptors at the sample (POSIX; omitted elsewhere).",
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "process.num_threads",
            MetricKind.GAUGE,
            "count",
            (),
            Reduction.LAST,
            "psutil.Process.num_threads(): OS threads in this process at the sample.",
            stability=Stability.EXPERIMENTAL,
        ),
        # host-wide metrics: every worker's SystemSampler runs with host=True, so these are live per
        # node. A machine hosting several workers reports the same host reading on each of their nodes
        # (they are not summed) — collapsing co-located workers into one physical-host rollup needs a
        # shared hostname identity and is future work.
        MetricSpec(
            "host.cpu_percent",
            MetricKind.GAUGE,
            "percent",
            (),
            Reduction.LAST,
            "psutil.cpu_percent(): host-wide CPU utilization since the previous sample. Per OS host; "
            "not summed across processes sharing a host.",
            since_stage=1,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "host.mem_percent",
            MetricKind.GAUGE,
            "percent",
            (),
            Reduction.LAST,
            "psutil.virtual_memory().percent: fraction of host physical memory in use at the sample. "
            "Per OS host; not summed across processes sharing a host.",
            since_stage=1,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "host.net_bytes_sent",
            MetricKind.GAUGE,
            "bytes",
            (),
            Reduction.LAST,
            "Bytes sent across all host network interfaces since the previous sample "
            "(psutil.net_io_counters delta). Host-wide, not summed across co-located workers; the "
            "OS-level counterpart to the per-edge transport.bytes_sent, which counts application payload.",
            relates_to=("transport.bytes_sent",),
            since_stage=1,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "host.net_bytes_recv",
            MetricKind.GAUGE,
            "bytes",
            (),
            Reduction.LAST,
            "Bytes received across all host network interfaces since the previous sample "
            "(psutil.net_io_counters delta). Host-wide, not summed across co-located workers.",
            since_stage=1,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "runtime.loop_lag_micros",
            MetricKind.HISTOGRAM,
            "microseconds",
            (),
            Reduction.SUM,
            "Difference between the requested asyncio.sleep interval and the monotonic time that "
            "actually elapsed before the sampler resumed, measured with perf_counter_ns.",
            relates_to=("runtime.step_micros",),
            boundaries=DURATION_US_BUCKETS,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "runtime.gil_percent",
            MetricKind.GAUGE,
            "percent",
            (),
            Reduction.LAST,
            "Fraction of the sampling interval the interpreter's global interpreter lock was held under "
            "contention, from the gilknocker monitor thread (100 means fully contended). Per process; "
            "recorded only at the FULL tier, and omitted when gilknocker is not installed.",
            relates_to=("runtime.loop_lag_micros", "runtime.step_micros"),
            since_stage=1,
            stability=Stability.EXPERIMENTAL,
            min_tier=Tier.FULL,
        ),
        # --- cross-process edges and placement (emitted by the socket channel + executor) --------
        MetricSpec(
            "edge.credit_wait_micros",
            MetricKind.COUNTER,
            "microseconds",
            _EDGE,
            Reduction.SUM,
            "Time the producer awaited flow-control credit on a channel.",
            since_stage=1,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "transport.bytes_sent",
            MetricKind.COUNTER,
            "bytes",
            _EDGE,
            Reduction.SUM,
            "Bytes written to a cross-process channel.",
            since_stage=1,
            stability=Stability.EXPERIMENTAL,
            min_tier=Tier.FULL,
        ),
        MetricSpec(
            "transport.encode_micros",
            MetricKind.COUNTER,
            "microseconds",
            _EDGE,
            Reduction.SUM,
            "Wall time the producer spent serializing frames to the wire (Arrow IPC for a batch, msgpack "
            "for a control frame) on a cross-process edge. A component of edge.send_wait_micros, "
            "separated out so serialization is distinguishable from flow-control and network waiting.",
            relates_to=("transport.bytes_sent", "edge.send_wait_micros", "transport.decode_micros"),
            since_stage=2,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "transport.decode_micros",
            MetricKind.COUNTER,
            "microseconds",
            ("operator_id",),
            Reduction.SUM,
            "Wall time this instance's inbound socket reader spent deserializing frames from the wire "
            "(Arrow IPC for a batch, msgpack for a control frame). Runs in the background read loop, so "
            "it overlaps the actor's own work; recorded once when the instance closes. No cross-process "
            "inbound edge means zero.",
            relates_to=("transport.encode_micros",),
            since_stage=2,
            stability=Stability.EXPERIMENTAL,
        ),
        MetricSpec(
            "placement.instances_per_worker",
            MetricKind.GAUGE,
            "count",
            ("node",),
            Reduction.LAST,
            "Number of operator instances placed on a worker.",
            since_stage=2,
            stability=Stability.EXPERIMENTAL,
        ),
    ]
}

EVENT_SPECS: dict[str, EventSpec] = {
    s.name: s
    for s in [
        EventSpec(
            "operator.lifecycle.open",
            ("operator_id", "op_class", "source_location", "num_inputs"),
            "An instance opened. Carries the source location anchoring it to code.",
        ),
        EventSpec(
            "operator.lifecycle.close",
            ("operator_id", "rows_in", "rows_out", "wall_micros"),
            "An instance closed, with its end-of-life counts.",
        ),
        EventSpec(
            "operator.error",
            (
                "operator_id",
                "op_class",
                "phase",
                "exc_type",
                "message",
                "traceback",
                "frame_kind",
                "input_index",
                "batch_rows",
                "source_location",
            ),
            "An exception was raised in a lifecycle method (recorded, then re-raised unchanged).",
        ),
        EventSpec(
            "eos.forwarded",
            ("operator_id", "wall_micros"),
            "An instance received EOS on all inputs and broadcast EOS downstream.",
            min_tier=Tier.COUNTERS_PLUS_EVENTS,
        ),
    ]
}

#: The metrics that define run identity in structural_digest(): per-instance row and EOS counts. The
#: computation conserves them — the same input yields the same totals however the engine places, routes,
#: or batches the work — so the digest matches across machines and worker counts. Batch counts are
#: excluded on purpose: deterministic within one run but not portable, because a distributed operator (a
#: join over a TCP shuffle) chunks its output by cross-worker arrival timing, so batches_out differs by
#: machine while every row and result is identical. Chunking is framing, not a correctness property.
STRUCTURAL_METRICS: frozenset[str] = frozenset(
    {
        "operator.rows_in",
        "operator.rows_out",
        "eos.received",
    }
)


def metric_spec(name: str) -> MetricSpec:
    try:
        return METRIC_SPECS[name]
    except KeyError:
        raise KeyError(f"undeclared metric {name!r}; add a MetricSpec to METRIC_SPECS") from None


def event_spec(name: str) -> EventSpec:
    try:
        return EVENT_SPECS[name]
    except KeyError:
        raise KeyError(f"undeclared event {name!r}; add an EventSpec to EVENT_SPECS") from None
