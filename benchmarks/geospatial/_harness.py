"""Shared harness for the nautilus↔xarray-sql geospatial comparison: timing, data loading, checking.

The sibling xarray-sql suite is *expressibility-first* — it proves each array operation, expressed as
SQL, matches an xarray reference. This harness carries that ethos to a third contender: it runs the
same operation three ways (the xarray reference, xarray-sql's DataFusion SQL, and nautilus's Arrow
dataflow), checks all three agree to floating-point tolerance, and times each.

**This measures in-memory COMPUTE only, and is deliberately NOT comparable to the xarray-sql suite's
published cold-read perf table.** The original suite runs one fresh process per repetition so every
engine pays a cold zarr read each time (`run_perf.sh`); it argues a warm in-process loop unfairly lets
the array reference serve later reps from RAM. This harness does exactly that warm loop on purpose:
nautilus has no zarr/gcs reader, so I/O is factored out for *all three* engines to isolate the compute
kernel. Consequences a reader must keep in mind:

* Every case `.load()`s its window once, outside the timed region — so the xarray reference here is the
  RAM-warm best case the original deliberately avoids, and the numbers are not the suite's numbers.
* A real pipeline over this data still needs xarray (or another reader) to ingest Zarr — nautilus
  cannot. The compute figures do not imply nautilus is a drop-in end-to-end replacement.
* Each relational engine pays its own gridded→relational unravel inside the timed region (nautilus
  flattens the grid to Arrow columns; xarray-sql converts chunks to Arrow inside `.sql()`); the array
  reference pays none (native layout = home field).

Memory is the **peak resident-set growth** during the timed step, sampled from ``/proc/self/statm`` by
a background thread (:class:`_PeakRSS`). Unlike ``tracemalloc`` — which sees only the CPython allocator
and is therefore blind to DataFusion's Rust heap and Arrow's C++ buffers — RSS counts native
allocations on every engine, so the three are measured with one instrument.
"""

from __future__ import annotations

import math
import os
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_PAGE = os.sysconf("SC_PAGE_SIZE")


class CaseSkipped(Exception):
    """Raised when a case cannot run here (data offline, credentials missing)."""


def _rss_bytes() -> int:
    """Current resident-set size — total physical memory the process holds, counting Python, numpy,
    Arrow C++ and DataFusion Rust allocations alike (unlike tracemalloc, which sees only CPython)."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * _PAGE


class _PeakRSS:
    """Sample peak RSS over a region from a daemon thread. Sub-millisecond polling catches the peak of
    the sub-second operations here; the reported figure is peak-minus-baseline, i.e. the working set the
    operation added on top of the already-resident inputs — the fair cross-engine memory signal."""

    def __init__(self, interval: float = 0.0005) -> None:
        self._interval = interval

    def __enter__(self) -> _PeakRSS:
        self._base = _rss_bytes()
        self._peak = self._base
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            r = _rss_bytes()
            if r > self._peak:
                self._peak = r
            self._stop.wait(self._interval)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.growth_mb = max(0.0, (self._peak - self._base) / 1e6)


@dataclass
class Timing:
    """One engine's measured step: median/min/max wall seconds and peak resident-set growth (MB)."""

    label: str
    median_s: float
    min_s: float
    max_s: float
    peak_mb: float
    reps: int

    def rows_per_s(self, n_rows: int) -> float:
        return n_rows / self.median_s if self.median_s else float("nan")


def measure(label: str, fn: Callable[[], Any], *, reps: int = 5, warmup: int = 1) -> tuple[Timing, Any]:
    """Time ``fn`` over ``warmup`` discarded then ``reps`` measured passes; return stats and the last
    result. Peak memory is the largest resident-set growth over the measured passes (see
    :class:`_PeakRSS`) — one instrument that counts native allocations on every engine."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    peak_max = 0.0
    for _ in range(reps):
        with _PeakRSS() as rss:
            t0 = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - t0
        times.append(elapsed)
        peak_max = max(peak_max, rss.growth_mb)
    return (
        Timing(label, statistics.median(times), min(times), max(times), peak_max, reps),
        result,
    )


def max_abs_diff(got: dict[Any, float], ref: dict[Any, float]) -> tuple[float, int]:
    """Largest ``|got - ref|`` over shared keys, and the count of keys in one dict but not the other
    (a query that silently drops or invents grid cells must not pass as a match). A NaN difference — one
    side diverged to NaN while the other did not — is forced to ``inf`` so it fails the tolerance gate
    rather than slipping through ``max``'s position-dependent NaN handling."""
    mismatch = len(set(ref).symmetric_difference(got))
    diffs = [abs(got[k] - ref[k]) for k in ref if k in got]
    if any(math.isnan(d) for d in diffs):
        return float("inf"), mismatch
    return max(diffs, default=float("nan")), mismatch
