#!/usr/bin/env python3
"""Distributed Zarr read across workers — nautilus's actual I/O-scaling regime.

The single-source worker test (``bench_workers.py``) feeds all data through one serial source, so
``run(workers=N)`` only adds cross-process shuffle cost — the wrong shape for a distributed engine, and it
does not help. This is the right shape. The IR pins a *source* to one instance, so read fan-out is
expressed as a tiny source of **chunk indices** (24 one-row batches) feeding a parallel async reader
(``_ops.ZarrReadChunk`` at parallelism N): ``run(workers=N)`` places the N reader instances on N worker
processes, each opening its own object-store client and fetching a different subset of the day's ARCO-ERA5
chunks off GCS, then a keyed shuffle feeds ``KeyedMean``. The read — which dominates wall-clock — is what
should scale here.

``deploy`` uses multiprocessing ``spawn`` (re-imports this module), so the ``__main__`` guard is required.

Run:  ``.venv/bin/python benchmarks/geospatial/bench_workers_zarr.py``
Env:  ``GEOBENCH_WORKERS`` (default ``1,2,4,8``), ``GEOBENCH_WREPS`` (default 3),
      ``GEOBENCH_INFLIGHT`` (per-reader concurrent fetches, default 8).
"""

# ruff: noqa: E402 — thread env vars must be set before numpy/pyarrow import, so imports follow code.
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import xarray as xr

from nautilus import from_batches, source

pa.set_cpu_count(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops import KeyedMean, ZarrReadChunk

ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WORKERS = [int(w) for w in os.environ.get("GEOBENCH_WORKERS", "1,2,4,8").split(",")]
WREPS = int(os.environ.get("GEOBENCH_WREPS", "3"))
INFLIGHT = int(os.environ.get("GEOBENCH_INFLIGHT", "8"))


def main() -> int:
    ds = xr.open_zarr(ERA5_URL, chunks=None, storage_options={"token": "anon"})
    tindex = ds.indexes["time"]
    day_idx = np.where(
        (tindex >= np.datetime64("2020-06-01")) & (tindex <= np.datetime64("2020-06-01T23"))
    )[0]
    lat = ds["latitude"].values
    nlat, nlon = len(lat), ds.sizes["longitude"]
    lat_idx = np.arange(nlat, dtype=np.int32)
    ref = (ds["2m_temperature"].isel(time=day_idx).mean(["longitude", "time"]) - 273.15).values

    # One tiny batch per chunk index, so the keyless source→reader edge round-robins them across the N
    # reader instances (batches, not rows, are the unit of fan-out).
    idx_batches = [
        pa.RecordBatch.from_arrays([pa.array([int(t)], pa.int64())], names=["chunk_idx"])
        for t in day_idx
    ]

    def cols(_t, data):
        return {"lat_idx": np.broadcast_to(lat_idx[:, None], (nlat, nlon)).reshape(-1),
                "tempk": data.reshape(-1)}

    def build(w):
        return (
            source(from_batches(*idx_batches))
            .apply_async(ZarrReadChunk(ERA5_URL, "2m_temperature", cols, max_in_flight=INFLIGHT), parallelism=w)
            .apply(KeyedMean("lat_idx", "tempk", "m"), key_columns="lat_idx", parallelism=w)
        )

    def run_at(w):
        st = build(w)
        t0 = time.perf_counter()
        res = st.run() if w == 1 else st.run(workers=w)
        total = time.perf_counter() - t0
        by = {int(i): float(m) - 273.15
              for b in res.batches
              for i, m in zip(b.column("lat_idx").to_pylist(), b.column("m").to_pylist(), strict=True)}
        got = np.array([by[i] for i in range(nlat)])
        return total, res.telemetry.meta.wall_micros / 1e6, np.allclose(got, ref, atol=1e-3), len(by)

    print(f"DISTRIBUTED ZARR READ — {len(day_idx)} chunks, parallel async reader, workers={WORKERS}, "
          f"median of {WREPS}, per-reader in-flight={INFLIGHT} (host {os.cpu_count()} cores)")
    print(f"\n    {'workers':>7}  {'total run()':>12}  {'pipeline':>12}  {'pipeline speedup':>16}")
    base = None
    for w in WORKERS:
        totals, pipes, ok = [], [], True
        for _ in range(WREPS):
            to, pi, good, ngroups = run_at(w)
            totals.append(to)
            pipes.append(pi)
            ok = ok and good and ngroups == nlat
        tot, pipe = statistics.median(totals), statistics.median(pipes)
        if base is None:
            base = pipe
        print(f"    {w:>7}  {tot:>10.2f}s  {pipe:>10.2f}s  {base / pipe:>14.2f}x  {'✅' if ok else '❌'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
