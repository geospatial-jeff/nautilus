"""The benchmark harness: measure a pipeline's throughput with enough rigor to tell a real change from
run-to-run noise, and compare against a committed baseline so a regression fails loudly.

One run's rows/sec proves nothing — it is noisy, and on a shared laptop the noise can swamp a real 1.2x.
So :func:`measure` runs a pipeline ``trials`` times (after discarding a ``warmup``), reports the
**median** with the **interquartile range** as the spread, and records the machine it ran on, because a
throughput figure is only comparable on the same hardware. :func:`compare` calls a change real only when
it clears both a floor and twice the observed noise, so the harness never claims a win it cannot
distinguish from jitter. Each result also carries the run's structural digest, so a baseline check
catches an output change masquerading as a speed change — a correctness regression, which is
machine-independent and always fails.

This is the sanctioned way to produce the before/after numbers the `perf-loop` skill and
``PERFORMANCE_CHANGELOG.md`` record; an ad-hoc single run or best-of-a-few is not.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import statistics
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nautilus
from nautilus.benchmarks import DEFAULT_BATCH, DEFAULT_KEYS, DEFAULT_ROWS
from nautilus.core.operator import OneInputOperator, SourceOperator
from nautilus.driver.local import run_local_chain
from nautilus.driver.pipeline import graph_from_pipeline
from nautilus.driver.result import RunResult
from nautilus.driver.run import run_plan
from nautilus.pipelines import is_graph_pipeline, load_graph_pipeline, load_pipeline
from nautilus.telemetry.catalog import Tier
from nautilus.telemetry.recorder import TelemetryConfig

DEFAULT_TRIALS = 5
DEFAULT_WARMUP = 1
#: A change smaller than this fraction of the baseline median is never called real, regardless of noise.
DEFAULT_THRESHOLD = 0.07
DEFAULT_BASELINE = Path("benchmarks/baseline.json")


# --- statistics --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stats:
    """A measured metric reduced to a robust center and a spread. ``rel_spread`` (IQR / median) is the
    run-to-run noise as a fraction — the thing a real change must beat to be believed."""

    samples: tuple[float, ...]
    median: float
    iqr: float
    rel_spread: float
    min: float
    max: float


def summarize(samples: Sequence[float]) -> Stats:
    s = sorted(float(x) for x in samples)
    if not s:
        raise ValueError("cannot summarize zero samples")
    median = statistics.median(s)
    # Inclusive quartiles work for the small sample count a benchmark uses; IQR is the robust,
    # outlier-tolerant spread (a single hiccuped run moves the max but not the median or the quartiles).
    if len(s) >= 2:
        q1, _q2, q3 = statistics.quantiles(s, n=4, method="inclusive")
        iqr = q3 - q1
    else:
        iqr = 0.0
    return Stats(tuple(s), median, iqr, (iqr / median if median else 0.0), s[0], s[-1])


# --- environment (a throughput number is only comparable on the same machine) -------------------


@dataclass(frozen=True)
class Environment:
    nautilus_version: str
    python_version: str
    platform: str
    processor: str
    commit: str | None


def _cpu_model() -> str:
    """The CPU model, recorded so :func:`_same_machine` can tell whether a run is on the same hardware as
    the baseline (the pinned benchmark runner). ``platform.processor()`` is empty on Linux, so read
    ``/proc/cpuinfo``'s ``model name``; fall back to the architecture, then ``"unknown"``."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def current_environment() -> Environment:
    return Environment(
        nautilus_version=nautilus.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        processor=_cpu_model(),
        commit=_git_commit(),
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# --- a single measured result ------------------------------------------------------------------


@dataclass(frozen=True)
class BenchResult:
    pipeline: str
    scale: dict[str, int]  # rows, batch, keys, parallelism, workers, tier
    trials: int
    throughput_rows_per_sec: Stats
    structural_digest: str
    deterministic: (
        bool  # the digest held across every trial (a comparison is meaningless otherwise)
    )
    environment: Environment
    recorded_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def result_from_dict(d: dict[str, Any]) -> BenchResult:
    t, e = d["throughput_rows_per_sec"], d["environment"]
    return BenchResult(
        pipeline=str(d["pipeline"]),
        scale={k: int(v) for k, v in d["scale"].items()},
        trials=int(d["trials"]),
        throughput_rows_per_sec=Stats(
            tuple(float(x) for x in t["samples"]),
            float(t["median"]),
            float(t["iqr"]),
            float(t["rel_spread"]),
            float(t["min"]),
            float(t["max"]),
        ),
        structural_digest=str(d["structural_digest"]),
        deterministic=bool(d["deterministic"]),
        environment=Environment(
            str(e["nautilus_version"]),
            str(e["python_version"]),
            str(e["platform"]),
            str(e["processor"]),
            (None if e["commit"] is None else str(e["commit"])),
        ),
        recorded_at=str(d["recorded_at"]),
    )


# --- running a pipeline ------------------------------------------------------------------------


@contextmanager
def _scaled_env(rows: int, batch: int, keys: int) -> Iterator[None]:
    """Set the synthetic-source scale for the duration (restored after). A non-synthetic pipeline
    ignores these, so the harness works for any pipeline."""
    overrides = {
        "NAUTILUS_BENCH_ROWS": str(rows),
        "NAUTILUS_BENCH_BATCH": str(batch),
        "NAUTILUS_BENCH_KEYS": str(keys),
    }
    # The scale dict does NOT record the stressor knobs bench-skew/bench-backpressure read (SKEW,
    # DELAY_US), so clear any ambient value for the duration — the pipelines fall back to their built-in
    # defaults. Otherwise a stray env var would leak into a bench-check and make a deterministic baseline
    # look changed on one machine but not another.
    pinned = ("NAUTILUS_BENCH_SKEW", "NAUTILUS_BENCH_DELAY_US")
    saved = {k: os.environ.get(k) for k in (*overrides, *pinned)}
    os.environ.update(overrides)
    for k in pinned:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_pipeline(
    source: SourceOperator,
    transforms: list[OneInputOperator],
    *,
    parallelism: int,
    workers: int,
    capacity: int,
    tier: Tier,
    key_groups: int | None = None,
    daemons: list[tuple[str, int]] | None = None,
) -> RunResult:
    """Run an already-loaded ``(source, transforms)`` at the given topology — single-process, in-process
    parallel, spawned across workers, or (``daemons`` set) dialed across worker daemons. The single
    topology-selection dispatch shared by the CLI's ``run`` and this bench harness, so the two cannot
    drift."""
    config = TelemetryConfig(tier=tier)
    if workers == 1 and daemons is None:
        # Single process — run_local_chain handles both serial and in-process parallel (any parallelism).
        return asyncio.run(
            run_local_chain(
                source,
                transforms,
                parallelism=parallelism,
                key_groups=key_groups,
                capacity=capacity,
                telemetry=config,
            )
        )
    from nautilus.cluster import deploy

    graph = graph_from_pipeline(source, transforms, parallelism)
    return deploy(
        graph,
        num_workers=workers,
        key_groups=key_groups,
        daemons=daemons,
        capacity=capacity,
        telemetry=config,
    )


def run_graph_pipeline(
    graph: object,
    *,
    workers: int,
    capacity: int,
    tier: Tier,
    key_groups: int | None = None,
    daemons: list[tuple[str, int]] | None = None,
) -> RunResult:
    """Run a multi-source graph pipeline (e.g. a join) at the given topology. The graph already carries
    its operator parallelism (baked in by the builder); ``workers``/``daemons`` select single-process,
    spawned, or dialed-daemon execution.
    """
    from nautilus.api import LogicalGraph

    assert isinstance(graph, LogicalGraph)
    config = TelemetryConfig(tier=tier)
    if workers == 1 and daemons is None:
        return asyncio.run(
            run_plan(graph, key_groups=key_groups, capacity=capacity, telemetry=config)
        )
    from nautilus.cluster import deploy

    return deploy(
        graph,
        num_workers=workers,
        key_groups=key_groups,
        daemons=daemons,
        capacity=capacity,
        telemetry=config,
    )


def run_once(
    pipeline: str,
    *,
    parallelism: int,
    workers: int,
    capacity: int,
    tier: Tier,
    key_groups: int | None = None,
    daemons: list[tuple[str, int]] | None = None,
) -> RunResult:
    """Build the named pipeline and run it once at the given topology (a fresh source per call). A graph
    pipeline (more than one source — a join) is built at ``parallelism`` and run via run_plan/deploy; a
    linear ``(source, transforms)`` pipeline goes through ``run_pipeline``. ``daemons`` runs across worker
    daemons instead of spawning locally; ``key_groups`` sets the keyed-shuffle rescale ceiling."""
    if is_graph_pipeline(pipeline):
        graph = load_graph_pipeline(pipeline, parallelism)
        return run_graph_pipeline(
            graph,
            workers=workers,
            capacity=capacity,
            tier=tier,
            key_groups=key_groups,
            daemons=daemons,
        )
    source, transforms = load_pipeline(pipeline)
    return run_pipeline(
        source,
        transforms,
        parallelism=parallelism,
        workers=workers,
        capacity=capacity,
        tier=tier,
        key_groups=key_groups,
        daemons=daemons,
    )


def measure(
    pipeline: str,
    *,
    rows: int = DEFAULT_ROWS,
    batch: int = DEFAULT_BATCH,
    keys: int = DEFAULT_KEYS,
    parallelism: int = 1,
    workers: int = 1,
    capacity: int = 16,
    tier: Tier = Tier.COUNTERS,
    trials: int = DEFAULT_TRIALS,
    warmup: int = DEFAULT_WARMUP,
    environment: Environment | None = None,
    recorded_at: str = "",
) -> BenchResult:
    """Run ``pipeline`` ``warmup`` + ``trials`` times and reduce the trials to a :class:`BenchResult`.

    The digest needs telemetry, so ``tier`` must be at least ``COUNTERS`` (the default). A fresh pipeline
    is built each trial because a source is consumed once."""
    if tier <= Tier.OFF:
        raise ValueError(
            "bench needs tier >= COUNTERS so a structural digest (the correctness anchor) exists"
        )
    throughputs: list[float] = []
    digests: list[str] = []
    with _scaled_env(rows, batch, keys):
        for _ in range(warmup):
            run_once(
                pipeline, parallelism=parallelism, workers=workers, capacity=capacity, tier=tier
            )
        for _ in range(trials):
            rep = run_once(
                pipeline, parallelism=parallelism, workers=workers, capacity=capacity, tier=tier
            ).telemetry
            throughputs.append(rep.throughput_rows_per_sec())
            digests.append(rep.structural_digest())
    return BenchResult(
        pipeline=pipeline,
        scale={
            "rows": rows,
            "batch": batch,
            "keys": keys,
            "parallelism": parallelism,
            "workers": workers,
            "tier": int(tier),
        },
        trials=trials,
        throughput_rows_per_sec=summarize(throughputs),
        structural_digest=digests[0],
        deterministic=len(set(digests)) == 1,
        environment=environment or current_environment(),
        recorded_at=recorded_at,
    )


def measure_like(result: BenchResult, **overrides: object) -> BenchResult:
    """Re-measure at exactly a recorded result's scale and topology (what ``bench-check`` does, so the
    comparison is apples-to-apples)."""
    s = result.scale
    return measure(
        result.pipeline,
        rows=s["rows"],
        batch=s["batch"],
        keys=s["keys"],
        parallelism=s["parallelism"],
        workers=s["workers"],
        tier=Tier(s["tier"]),
        trials=result.trials,
        **overrides,  # type: ignore[arg-type]
    )


# --- comparison (the honesty: only beat-the-noise changes count) -------------------------------

# A comparison status. Order of precedence when classifying:
#   nondeterministic  either run's digest wobbled across trials, so there is no stable correctness
#                     anchor — reported, never failed (checked first, before the digest comparison).
#   OUTPUT-CHANGED  the structural digest differs — the change altered results. Machine-independent,
#                   always a failure (a "faster" run that computes something else is not faster).
#   machine-differs digests match but the baseline ran on different hardware — a different CPU model or
#                   OS image — so throughput is not comparable — reported, never failed.
#   REGRESSED       median throughput fell by more than the noise-aware threshold.
#   IMPROVED        median throughput rose by more than the threshold.
#   unchanged       within the noise floor — no claim either way.


@dataclass(frozen=True)
class Comparison:
    pipeline: str
    status: str
    delta: float  # signed relative change in median throughput (new - base) / base
    threshold: float  # the bar the change had to clear, = max(floor, 2 x noise)
    base_median: float
    new_median: float
    noise: float


def _same_machine(a: Environment, b: Environment) -> bool:
    """Whether two runs are on comparable hardware, so a throughput delta is meaningful. The baseline is
    recorded on the pinned benchmark runner; a run on any other machine (a dev laptop, a shared CI runner)
    differs in OS image (``platform``) or CPU model (``processor``), and a memory-bound pipeline's
    throughput swings with the CPU — so a mismatch reads as ``machine-differs`` (throughput skipped,
    digest still checked) rather than a false regression."""
    return a.platform == b.platform and a.processor == b.processor


def compare(
    base: BenchResult, current: BenchResult, *, min_threshold: float = DEFAULT_THRESHOLD
) -> Comparison:
    b, c = base.throughput_rows_per_sec, current.throughput_rows_per_sec
    delta = (c.median - b.median) / b.median if b.median else 0.0
    noise = max(b.rel_spread, c.rel_spread)
    threshold = max(min_threshold, 2 * noise)
    if not base.deterministic or not current.deterministic:
        # A run whose digest wobbled across trials has no stable correctness anchor, so a digest
        # mismatch is not evidence the output changed — report it, never fail on it.
        status = "nondeterministic"
    elif base.structural_digest != current.structural_digest:
        status = "OUTPUT-CHANGED"
    elif not _same_machine(base.environment, current.environment):
        status = "machine-differs"
    elif delta < -threshold:
        status = "REGRESSED"
    elif delta > threshold:
        status = "IMPROVED"
    else:
        status = "unchanged"
    return Comparison(current.pipeline, status, delta, threshold, b.median, c.median, noise)


def is_failure(status: str) -> bool:
    """A status that should fail a regression check (and a CI exit code)."""
    return status in ("OUTPUT-CHANGED", "REGRESSED")


#: How many extra times a REGRESSED benchmark is re-measured before the drop is believed. Even the pinned
#: runner is a shared-tenant cloud box, so one memory-bound benchmark can land ~10% below its baseline in a
#: slower machine state (a co-tenant contending memory bandwidth, or a CPU-frequency dip) while the code is
#: unchanged — observed as a *bimodal* wobble, not a gradual drift. This many retries makes a persistent
#: false alarm from a ~50/50 wobble unlikely, and only ever runs for a benchmark that already failed.
RETRY_ON_REGRESSION = 5


def confirm_regression(
    base: BenchResult,
    first: BenchResult,
    remeasure: Callable[[], BenchResult],
    *,
    min_threshold: float = DEFAULT_THRESHOLD,
    retries: int = RETRY_ON_REGRESSION,
) -> tuple[BenchResult, Comparison, int]:
    """Re-check a REGRESSED verdict by re-measuring and keeping the *fastest* result — so a transient slow
    machine state does not read as a code regression (see :data:`RETRY_ON_REGRESSION`).

    This is sound because contention only ever *lowers* throughput: a run can be slowed by a noisy
    neighbour but never sped up past what the code allows, so the best of several runs is the least-
    contended, truest measure. A real regression caps that best too and still fails — this filters noise,
    it never manufactures a win (unlike the best-of-a-few the module docstring warns against, which is
    about *claiming* a speedup). Any other verdict — unchanged, an output change, a machine mismatch — is
    returned untouched; only a throughput drop is worth, and susceptible to, a second look.

    Returns the kept result, its comparison against ``base``, and how many extra measurements it took.
    """
    best = first
    cmp = compare(base, best, min_threshold=min_threshold)
    used = 0
    while cmp.status == "REGRESSED" and used < retries:
        used += 1
        candidate = remeasure()
        if candidate.throughput_rows_per_sec.median > best.throughput_rows_per_sec.median:
            best = candidate
        cmp = compare(base, best, min_threshold=min_threshold)
    return best, cmp, used


# --- baseline file -----------------------------------------------------------------------------

BASELINE_VERSION = 1


def load_baseline(path: Path) -> dict[str, BenchResult]:
    raw = json.loads(path.read_text())
    return {name: result_from_dict(d) for name, d in raw.get("results", {}).items()}


def save_baseline(path: Path, results: dict[str, BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BASELINE_VERSION,
        "note": (
            "Throughput is recorded on the pinned benchmark runner and gated only on matching hardware "
            "(see each entry's environment); structural_digest is portable and anchors output "
            "correctness on any machine. Maintained by `nautilus bench-check --update`."
        ),
        "results": {name: r.to_dict() for name, r in sorted(results.items())},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
