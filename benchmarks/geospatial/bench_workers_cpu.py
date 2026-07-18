#!/usr/bin/env python3
"""Does `run(workers=N)` scale a CPU-bound stage? — the complement to the aggregation tests.

`bench_workers.py` showed a keyed aggregation does NOT scale across workers: all rows cross the process
boundary through one serial source, so transport, not compute, is the wall. This isolates the opposite
case — a stateless, *compute-heavy* stage where the per-batch work dwarfs the bytes shipped. A tiny source
emits a fixed set of tiles; a `.map` at parallelism N (placed on N worker processes) runs a heavy per-tile
numpy kernel. Because each worker is its own OS process on its own core (in-process `parallelism>1` shares
one event-loop thread, so a synchronous kernel there gets no CPU parallelism — that is why p4 never helped),
the compute genuinely parallelises here. This is the shape where nautilus's multi-process scale-out pays.

`deploy` uses multiprocessing `spawn` (re-imports this module), so the `__main__` guard is required.

Run:  ``.venv/bin/python benchmarks/geospatial/bench_workers_cpu.py``
Env:  ``GEOBENCH_WORKERS`` (default ``1,2,4,8``), ``GEOBENCH_TILES`` (default 64),
      ``GEOBENCH_CPU_ITERS`` (kernel intensity, default 60), ``GEOBENCH_WREPS`` (default 3).
"""

# ruff: noqa: E402 — thread env vars must be set before numpy/pyarrow import, so imports follow code.
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import statistics
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa

from nautilus import from_batches, source

pa.set_cpu_count(1)

WORKERS = [int(w) for w in os.environ.get("GEOBENCH_WORKERS", "1,2,4,8").split(",")]
N_TILES = int(os.environ.get("GEOBENCH_TILES", "64"))
K = int(os.environ.get("GEOBENCH_CPU_ITERS", "60"))
WREPS = int(os.environ.get("GEOBENCH_WREPS", "3"))
TILE = 512


def heavy(b: pa.RecordBatch) -> pa.RecordBatch:
    r = b.column("red").to_numpy()
    nir = b.column("nir").to_numpy()
    out = (nir - r) / (nir + r)  # NDVI, then a CPU-heavy transcendental loop (numpy, ~256k elems)
    for _ in range(K):
        out = np.tanh(out) + np.sin(out) * np.cos(out)
    return pa.RecordBatch.from_arrays([pa.array(out)], names=["v"])


def main() -> int:
    rng = np.random.default_rng(0)
    tiles = [
        pa.RecordBatch.from_arrays(
            [pa.array(rng.uniform(0.02, 0.3, TILE * TILE).astype("float32")),
             pa.array(rng.uniform(0.2, 0.6, TILE * TILE).astype("float32"))],
            names=["red", "nir"])
        for _ in range(N_TILES)
    ]
    npix = N_TILES * TILE * TILE

    def run_at(w):
        st = source(from_batches(*tiles)).map(heavy, parallelism=w)
        t0 = time.perf_counter()
        res = st.run() if w == 1 else st.run(workers=w)
        total = time.perf_counter() - t0
        rows = sum(b.num_rows for b in res.batches)
        return total, res.telemetry.meta.wall_micros / 1e6, rows

    print(f"CPU-BOUND MAP SCALE-OUT — {N_TILES} tiles × {TILE}² px, kernel iters={K}, workers={WORKERS}, "
          f"median of {WREPS} (host {os.cpu_count()} cores)")
    print(f"\n    {'workers':>7}  {'total run()':>12}  {'pipeline':>12}  {'pipeline speedup':>16}  {'Mpix/s':>8}")
    base = None
    for w in WORKERS:
        totals, pipes, ok = [], [], True
        for _ in range(WREPS):
            to, pi, rows = run_at(w)
            totals.append(to)
            pipes.append(pi)
            ok = ok and rows == npix
        tot, pipe = statistics.median(totals), statistics.median(pipes)
        if base is None:
            base = pipe
        print(f"    {w:>7}  {tot:>10.2f}s  {pipe:>10.2f}s  {base / pipe:>14.2f}x  "
              f"{npix / pipe / 1e6:>7.0f}  {'✅' if ok else '❌'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
