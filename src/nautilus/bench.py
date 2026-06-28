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
``PERFORMANCE_CHANGELOG.md`` record; an ad-hoc single run or best-of-N is not.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import statistics
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nautilus
from nautilus.benchmarks import DEFAULT_BATCH, DEFAULT_KEYS, DEFAULT_ROWS, DEFAULT_WM_EVERY
from nautilus.core.operator import OneInputOperator, SourceOperator
from nautilus.pipelines import load_pipeline
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.parallel import graph_from_pipeline
from nautilus.runtime.result import RunResult
from nautilus.runtime.run import run_plan
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
    # Inclusive quartiles work for the small N a benchmark uses; IQR is the robust, outlier-tolerant
    # spread (a single hiccuped run moves the max but not the median or the quartiles).
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


def current_environment() -> Environment:
    return Environment(
        nautilus_version=nautilus.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor() or "unknown",
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
    scale: dict[str, int]  # rows, batch, keys, wm_every, parallelism, workers, tier
    trials: int
    throughput_rows_per_sec: Stats
    structural_digest: str
    deterministic: bool  # the digest held across every trial (a comparison is meaningless otherwise)
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
            float(t["median"]), float(t["iqr"]), float(t["rel_spread"]), float(t["min"]), float(t["max"]),
        ),
        structural_digest=str(d["structural_digest"]),
        deterministic=bool(d["deterministic"]),
        environment=Environment(
            str(e["nautilus_version"]), str(e["python_version"]), str(e["platform"]),
            str(e["processor"]), (None if e["commit"] is None else str(e["commit"])),
        ),
        recorded_at=str(d["recorded_at"]),
    )


# --- running a pipeline ------------------------------------------------------------------------


@contextmanager
def _scaled_env(rows: int, batch: int, keys: int, wm_every: int) -> Iterator[None]:
    """Set the synthetic-source scale for the duration (restored after). A non-synthetic pipeline
    ignores these, so the harness works for any pipeline."""
    overrides = {
        "NAUTILUS_BENCH_ROWS": str(rows),
        "NAUTILUS_BENCH_BATCH": str(batch),
        "NAUTILUS_BENCH_KEYS": str(keys),
        "NAUTILUS_BENCH_WM_EVERY": str(wm_every),
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
) -> RunResult:
    """Run an already-loaded ``(source, transforms)`` at the given topology — single-process, in-process
    parallel, or deployed across workers. The single topology-selection dispatch shared by the CLI's
    ``run`` and this bench harness, so the two cannot drift."""
    config = TelemetryConfig(tier=tier)
    if workers == 1 and parallelism == 1:
        return asyncio.run(run_local_chain(source, transforms, capacity=capacity, telemetry=config))
    graph = graph_from_pipeline(source, transforms, parallelism)
    if workers == 1:
        return asyncio.run(run_plan(graph, capacity=capacity, telemetry=config))
    from nautilus.cluster import deploy

    return deploy(graph, num_workers=workers, capacity=capacity, telemetry=config)


def run_once(
    pipeline: str, *, parallelism: int, workers: int, capacity: int, tier: Tier
) -> RunResult:
    """Build the named pipeline and run it once at the given topology (a fresh source per call)."""
    source, transforms = load_pipeline(pipeline)
    return run_pipeline(
        source, transforms, parallelism=parallelism, workers=workers, capacity=capacity, tier=tier
    )


def measure(
    pipeline: str,
    *,
    rows: int = DEFAULT_ROWS,
    batch: int = DEFAULT_BATCH,
    keys: int = DEFAULT_KEYS,
    wm_every: int = DEFAULT_WM_EVERY,
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
        raise ValueError("bench needs tier >= COUNTERS so a structural digest (the correctness anchor) exists")
    throughputs: list[float] = []
    digests: list[str] = []
    with _scaled_env(rows, batch, keys, wm_every):
        for _ in range(warmup):
            run_once(pipeline, parallelism=parallelism, workers=workers, capacity=capacity, tier=tier)
        for _ in range(trials):
            rep = run_once(
                pipeline, parallelism=parallelism, workers=workers, capacity=capacity, tier=tier
            ).telemetry
            throughputs.append(rep.throughput_rows_per_sec())
            digests.append(rep.structural_digest())
    return BenchResult(
        pipeline=pipeline,
        scale={
            "rows": rows, "batch": batch, "keys": keys, "wm_every": wm_every,
            "parallelism": parallelism, "workers": workers, "tier": int(tier),
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
        rows=s["rows"], batch=s["batch"], keys=s["keys"], wm_every=s["wm_every"],
        parallelism=s["parallelism"], workers=s["workers"], tier=Tier(s["tier"]),
        trials=result.trials,
        **overrides,  # type: ignore[arg-type]
    )


# --- comparison (the honesty: only beat-the-noise changes count) -------------------------------

# A comparison status. Order of precedence when classifying:
#   nondeterministic  either run's digest wobbled across trials, so there is no stable correctness
#                     anchor — reported, never failed (checked first, before the digest comparison).
#   OUTPUT-CHANGED  the structural digest differs — the change altered results. Machine-independent,
#                   always a failure (a "faster" run that computes something else is not faster).
#   machine-differs digests match but the baseline ran on different hardware, so throughput is not
#                   comparable — reported, never failed.
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


def compare(base: BenchResult, current: BenchResult, *, min_threshold: float = DEFAULT_THRESHOLD) -> Comparison:
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
    elif base.environment.platform != current.environment.platform:
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
            "Throughput is machine-specific (see each entry's environment); re-baseline per machine. "
            "structural_digest is portable and anchors output correctness across machines. "
            "Maintained by `nautilus bench --update` / `nautilus bench-check`."
        ),
        "results": {name: r.to_dict() for name, r in sorted(results.items())},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
