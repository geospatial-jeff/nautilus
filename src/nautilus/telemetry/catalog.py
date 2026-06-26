"""The declarative catalog of nautilus metrics and events.

Every metric and event nautilus emits is declared here exactly once. Instruments are created *by
catalog key*, so an undeclared metric cannot be emitted and the report schema can never drift from the
data. For each number it states WHAT it measures (a fact) and which OTHER metrics relate to it — never
what a value indicates, never a cause, never a remedy. A unit test checks every catalog string against
:data:`BANNED_ANALYSIS_WORDS` so analysis cannot regress in: nautilus records the data; the analysis is
done separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class MetricKind(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class Reduction(StrEnum):
    """How a series is reduced when rolled up across instances/subtasks."""

    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"


class Stability(StrEnum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"


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
            ("operator_id", "op_class", "subtask_index"),
            Reduction.SUM,
            "Wall time of one op.process(batch) call, measured with perf_counter_ns.",
            relates_to=("operator.batch_rows",),
            boundaries=DURATION_US_BUCKETS,
        ),
        MetricSpec(
            "operator.on_watermark_micros",
            MetricKind.HISTOGRAM,
            "microseconds",
            _OP,
            Reduction.SUM,
            "Wall time of one op.on_watermark(t) call.",
            relates_to=("window.fires",),
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
            "operator.on_watermark_calls",
            MetricKind.COUNTER,
            "calls",
            _OP,
            Reduction.SUM,
            "Number of op.on_watermark invocations.",
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
        # --- watermarks ----------------------------------------------------------------------
        MetricSpec(
            "watermark.combined_micros",
            MetricKind.GAUGE,
            "event_time_micros",
            _OP,
            Reduction.MIN,
            "Latest WatermarkTracker.combined for this instance.",
            relates_to=("watermark.advances", "watermark.input_idle"),
        ),
        MetricSpec(
            "watermark.advances",
            MetricKind.COUNTER,
            "count",
            ("operator_id",),
            Reduction.SUM,
            "Number of times the combined watermark strictly increased.",
            relates_to=("watermark.combined_micros",),
            deterministic=True,
        ),
        MetricSpec(
            "watermark.final_micros",
            MetricKind.GAUGE,
            "event_time_micros",
            ("operator_id",),
            Reduction.MIN,
            "Combined watermark at close (WATERMARK_MAX for a finished bounded run).",
            deterministic=True,
        ),
        MetricSpec(
            "watermark.input_idle",
            MetricKind.COUNTER,
            "count",
            ("operator_id", "input_index"),
            Reduction.SUM,
            "Number of StatusIdle frames received on an input.",
            relates_to=("watermark.combined_micros",),
        ),
        MetricSpec(
            "watermark.input_active",
            MetricKind.COUNTER,
            "count",
            ("operator_id", "input_index"),
            Reduction.SUM,
            "Number of StatusActive frames received on an input.",
            relates_to=("watermark.combined_micros",),
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
        # --- windows / state (operator-author or runtime) ------------------------------------
        MetricSpec(
            "window.fires",
            MetricKind.COUNTER,
            "count",
            ("operator_id",),
            Reduction.SUM,
            "Number of windows emitted across on_watermark calls.",
            relates_to=("operator.on_watermark_micros",),
            deterministic=True,
        ),
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
            "Summed wall time spent inside synchronous process/on_watermark critical sections.",
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
        # --- errors --------------------------------------------------------------------------
        MetricSpec(
            "operator.errors",
            MetricKind.COUNTER,
            "count",
            ("operator_id", "op_class", "exc_type"),
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
        # host-wide metrics are sampled only when SystemSampler(host=True); no shipped caller enables
        # that yet (host rollups are a multi-node seam), so they are marked since_stage=1 (reserved),
        # not advertised as live stage-0 facts.
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
        # --- reserved for later stages (declared now so the schema stays additive) -----------
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

#: Metrics whose values are provably reproducible and that define run identity in structural_digest().
STRUCTURAL_METRICS: frozenset[str] = frozenset(
    {
        "operator.rows_in",
        "operator.rows_out",
        "operator.batches_in",
        "operator.batches_out",
        "watermark.advances",
        "eos.received",
        "watermark.final_micros",
    }
)


def metric_spec(name: str) -> MetricSpec:
    try:
        return METRIC_SPECS[name]
    except KeyError:
        raise KeyError(f"undeclared metric {name!r}; add a MetricSpec to the CATALOG") from None


def event_spec(name: str) -> EventSpec:
    try:
        return EVENT_SPECS[name]
    except KeyError:
        raise KeyError(f"undeclared event {name!r}; add an EventSpec to the CATALOG") from None
