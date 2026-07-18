#!/usr/bin/env python3
"""Multi-worker scale-out: does ``run(workers=N)`` recover the keyed-aggregation losses?

The compute-only benchmark runs nautilus in one process (one event loop, one GIL), where its in-process
``parallelism>1`` only adds shuffle overhead — it cannot use a second core. nautilus's *real* parallelism
is ``run(workers=N)``: N spawned processes, the keyed shuffle crossing sockets between them, each worker
on its own core with its own GIL. This tests whether that recovers the high-cardinality keyed cases
(02 climatology's 535 k groups, 03 zonal mean's 25 M rows) where single-process nautilus lost badly.

Two clocks are reported because they answer different questions:
* **total** — wall time of the whole ``run(workers=N)`` call, including spawning N processes and
  cloudpickling the graph to them. This is what a one-shot job pays.
* **pipeline** — the coordinator's telemetry ``wall_micros``: the actual dataflow execution once the
  workers are up. This is what a long-lived (daemon) deployment pays per job, and it is where compute
  parallelism shows.

``deploy`` uses multiprocessing ``spawn`` (re-imports this module in each worker), so the ``__main__``
guard below is mandatory; data is loaded in ``main`` and reaches the source worker by cloudpickle.

Run:  ``.venv/bin/python benchmarks/geospatial/bench_workers.py``
Env:  ``GEOBENCH_WORKERS`` (default ``1,2,4,8``), ``GEOBENCH_WREPS`` (default 3).
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

from nautilus import source
from nautilus.operators import KeyedMean

pa.set_cpu_count(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops import SlicedSource  # noqa: E402

ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WORKERS = [int(w) for w in os.environ.get("GEOBENCH_WORKERS", "1,2,4,8").split(",")]
WREPS = int(os.environ.get("GEOBENCH_WREPS", "3"))
BATCH = 262_144


def _build_zonal_mean():
    ds = xr.open_zarr(ERA5_URL, chunks=None, storage_options={"token": "anon"})
    day = ds["2m_temperature"].sel(time="2020-06-01").load()
    nt, nlat, nlon = day.sizes["time"], day.sizes["latitude"], day.sizes["longitude"]
    vals = day.values
    lat_idx = np.arange(nlat, dtype=np.int32)

    def slice_fn(t):
        return {"lat_idx": np.broadcast_to(lat_idx[:, None], (nlat, nlon)).reshape(-1),
                "tempk": vals[t].reshape(-1)}

    def build(par):
        return source(SlicedSource(nt, slice_fn, BATCH)).apply(
            KeyedMean("lat_idx", "tempk", "m"), key_columns="lat_idx", parallelism=par)

    return build, nt * nlat * nlon, 721, "03 zonal mean (25M rows, 721 groups)"


def _build_climatology():
    ds = xr.open_zarr(ERA5_URL, chunks=None, storage_options={"token": "anon"})
    w = ds["2m_temperature"].sel(
        time=slice("2020-06-01", "2020-06-03T23"),
        latitude=slice(50.0, 25.0), longitude=slice(235.0, 290.0)).load()
    nt, nlat, nlon = w.sizes["time"], w.sizes["latitude"], w.sizes["longitude"]
    vals = w.values
    hours = w["time"].dt.hour.values.astype(np.int64)
    li = np.broadcast_to(np.arange(nlat)[:, None], (nlat, nlon))
    oi = np.broadcast_to(np.arange(nlon)[None, :], (nlat, nlon))
    base_gid = ((li * nlon + oi) * 24).reshape(-1).astype(np.int64)

    def slice_fn(t):
        return {"gid": base_gid + hours[t], "tempk": vals[t].reshape(-1)}

    def build(par):
        return source(SlicedSource(nt, slice_fn, BATCH)).apply(
            KeyedMean("gid", "tempk", "m"), key_columns="gid", parallelism=par)

    return build, nt * nlat * nlon, nlat * nlon * 24, "02 climatology (1.6M rows, 535k groups)"


def _run(build, workers):
    stream = build(workers)
    t0 = time.perf_counter()
    res = stream.run() if workers == 1 else stream.run(workers=workers)
    total = time.perf_counter() - t0
    pipeline = res.telemetry.meta.wall_micros / 1e6
    keyset = frozenset(
        k for b in res.batches for k in b.column(b.schema.names[0]).to_pylist()
    )
    return total, pipeline, len(res.batches and keyset), keyset


def main() -> int:
    print(f"MULTI-WORKER SCALE-OUT — workers={WORKERS}, median of {WREPS}  (host {os.cpu_count()} cores)")
    for factory in (_build_zonal_mean, _build_climatology):
        build, n_rows, n_groups, title = factory()
        print(f"\n▸ {title}")
        print(f"    {'workers':>7}  {'total run()':>12}  {'pipeline':>12}  {'pipeline speedup':>16}")
        base_pipe, base_keys = None, None
        for w in WORKERS:
            totals, pipes, keys = [], [], None
            for _ in range(WREPS):
                to, pi, nk, keyset = _run(build, w)
                totals.append(to)
                pipes.append(pi)
                keys = keyset
            tot, pipe = statistics.median(totals), statistics.median(pipes)
            if base_pipe is None:
                base_pipe, base_keys = pipe, keys
            ok = "✅" if keys == base_keys and len(keys) == n_groups else "❌"
            print(f"    {w:>7}  {tot:>10.2f}s  {pipe:>10.2f}s  {base_pipe / pipe:>14.2f}x  {ok} ({len(keys)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
