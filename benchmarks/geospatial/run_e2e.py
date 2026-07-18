#!/usr/bin/env python3
"""End-to-end cold-read benchmark: each engine reads the day's Zarr chunks off GCS and computes, in a
fresh process per measurement so every read is cold.

``run_bench.py`` isolates the compute kernel (data pre-loaded). This one measures the *whole* pipeline —
including the Zarr read — which is what a real user pays and, per the upstream suite, is what dominates
wall-clock. It exists because nautilus can now read Zarr itself (``_ops.ZarrChunkSource``, obstore +
zarr-python async), so the read is finally something all three engines do rather than a step factored
out. A fresh subprocess per rep (via ``e2e_case.py``) defeats every client-side block cache, so nautilus,
xarray, and xarray-sql each pay a cold GCS read on every measurement — the only fair way to compare reads.

Run:  ``.venv/bin/python benchmarks/geospatial/run_e2e.py [03 06]``  (default: both)
Env:  ``GEOBENCH_E2E_REPS`` (default 5), ``GEOBENCH_PREFETCH`` (nautilus read-ahead depth, default 8).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys

REPS = int(os.environ.get("GEOBENCH_E2E_REPS", "5"))
ENGINES = ["xarray", "xarray-sql", "nautilus"]
CASE_TITLES = {
    "01": "NDVI (Sentinel-2 per-pixel)",
    "02": "climatology (GROUP BY lat,lon,hour)",
    "03": "zonal mean (GROUP BY latitude)",
    "04": "anomaly (climatology self-JOIN)",
    "05": "forecast skill (JOIN + RMSE)",
    "06": "zonal vector (range JOIN)",
}
_HERE = os.path.dirname(os.path.abspath(__file__))


def _one(engine: str, case: str) -> tuple[float, str] | None:
    """Run one cold (engine, case) in a fresh process; return (seconds, digest) or None on failure."""
    p = subprocess.run(
        [sys.executable, os.path.join(_HERE, "e2e_case.py"), engine, case],
        capture_output=True, text=True,
    )
    t = d = None
    for line in p.stdout.splitlines():
        if line.startswith("TIME="):
            t = float(line[5:])
        elif line.startswith("DIGEST="):
            d = line[7:]
    if t is None or d is None:
        sys.stderr.write(f"  ! {engine} {case} failed:\n{p.stderr[-600:]}\n")
        return None
    return t, d


def main() -> int:
    cases = [a for a in sys.argv[1:] if a in CASE_TITLES] or list(CASE_TITLES)
    print(f"END-TO-END COLD READ — fresh process per rep, reps={REPS}, prefetch={os.environ.get('GEOBENCH_PREFETCH', '8')}")
    print("(each engine reads the day's ARCO-ERA5 Zarr chunks off GCS + computes; nautilus via its own async source)")
    rows = []
    for case in cases:
        print(f"\n▸ {case} {CASE_TITLES[case]}")
        stats, digests = {}, {}
        for engine in ENGINES:
            samples = [r for _ in range(REPS) if (r := _one(engine, case))]
            if not samples:
                continue
            times = [t for t, _ in samples]
            digests[engine] = {d for _, d in samples}
            stats[engine] = (statistics.median(times), min(times))
            print(f"    {engine:<12} median {statistics.median(times):6.2f}s   min {min(times):6.2f}s")
        all_digests = {d for ds in digests.values() for d in ds}
        ok = len(all_digests) == 1 and all(len(ds) == 1 for ds in digests.values())
        print(f"    correctness: {'✅ all engines agree' if ok else '❌ digests differ: ' + str(digests)}")
        if "nautilus" in stats:
            rows.append((case, stats))

    print("\n" + "-" * 78)
    print("SPEEDUP — nautilus end-to-end vs each rival  (>1 = nautilus faster, cold read included)")
    print(f"    {'case':<34}{'vs xarray-sql':>16}{'vs xarray':>16}")
    for case, stats in rows:
        n = stats["nautilus"][0]
        sql = f"{stats['xarray-sql'][0] / n:.2f}x" if "xarray-sql" in stats else "—"
        xr_ = f"{stats['xarray'][0] / n:.2f}x" if "xarray" in stats else "—"
        print(f"    {case + ' ' + CASE_TITLES[case]:<34}{sql:>16}{xr_:>16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
