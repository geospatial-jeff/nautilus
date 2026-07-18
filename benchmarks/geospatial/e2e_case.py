#!/usr/bin/env python3
"""One cold end-to-end read+compute for a single (engine, case) — the unit the cold-read driver spawns.

The compute-only benchmark (``run_bench.py``) pre-loads data and times the kernel. This instead times the
*whole* pipeline — open the store, read the day's Zarr chunks off GCS, compute — for one engine, once.
Run in a fresh process per measurement (see ``run_e2e.py``) so every read is cold: no client keeps a warm
block cache across reps, which is the only way the read is measured fairly (the reason the upstream
xarray-sql harness also forks per rep). nautilus reads Zarr through its own async ``ZarrChunkSource``
(obstore backend, prefetching), with no xarray in the read path; xarray and xarray-sql read through
xarray's zarr/gcsfs stack, as they must.

Usage:  e2e_case.py <xarray|xarray-sql|nautilus> <03|06>
Prints: ``TIME=<seconds>`` and ``DIGEST=<sha256 of rounded results>`` (the cross-engine correctness check).
"""

# ruff: noqa: E402 — thread env vars must be set before numpy/pyarrow import, so imports follow code.
from __future__ import annotations

import hashlib
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import xarray as xr
import xarray_sql as xql
from datafusion import SessionConfig

from nautilus import source

pa.set_cpu_count(1)
sys.path.insert(0, os.path.dirname(__file__))
from _ops import KeyedMean, ZarrChunkSource, make_region_tagger  # noqa: E402

ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DAY = os.environ.get("GEOBENCH_ERA5_DAY", "2020-06-01")
PREFETCH = int(os.environ.get("GEOBENCH_PREFETCH", "8"))
_REGIONS = [
    ("Sahara", 18.0, 30.0, 0.0, 30.0),
    ("Amazon", -10.0, 5.0, 290.0, 310.0),
    ("Australia_Outback", -30.0, -20.0, 125.0, 140.0),
    ("Greenland", 65.0, 80.0, 300.0, 340.0),
    ("SE_Asia", 5.0, 20.0, 95.0, 110.0),
]


def _digest(pairs: list[tuple[int, float]]) -> str:
    """Stable hash of (key, value-rounded-to-2dp) pairs — same numbers → same digest across engines."""
    body = ";".join(f"{k}:{round(v, 2)}" for k, v in sorted(pairs))
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _setup():
    """Metadata only (untimed): resolve the day's time indices and the coordinate axes. Every engine then
    reads exactly the day's chunks, so the read volume is identical and the archive-pruning question (a
    separate xarray-sql strength) is out of scope, matching the compute-only benchmark."""
    ds = xr.open_zarr(ERA5_URL, chunks=None, storage_options={"token": "anon"})
    tindex = ds.indexes["time"]
    day_idx = np.where(
        (tindex >= np.datetime64(DAY)) & (tindex <= np.datetime64(f"{DAY}T23"))
    )[0]
    return ds, day_idx, ds["latitude"].values, ds["longitude"].values


def run_xarray(ds, day_idx, lat, lon, case):
    day = ds["2m_temperature"].isel(time=day_idx)
    if case == "03":
        r = day.mean(["longitude", "time"]) - 273.15
        return [(round(float(la), 4), float(v)) for la, v in zip(r.latitude.values, r.values, strict=True)]
    in_region = xr.concat(
        [(day.latitude >= a) & (day.latitude <= b) & (day.longitude >= c) & (day.longitude <= d)
         for _, a, b, c, d in _REGIONS], dim="region_id")
    r = (day.where(in_region).mean(["time", "latitude", "longitude"]) - 273.15).values
    return [(i, float(v)) for i, v in enumerate(r)]


def run_xarray_sql(ds, day_idx, lat, lon, case):
    ctx = xql.XarrayContext(SessionConfig().with_target_partitions(1))
    day = ds[["2m_temperature"]].isel(time=day_idx)
    ctx.from_dataset("era5", day, chunks={"time": 6})
    if case == "03":
        sql = ('SELECT latitude, AVG("2m_temperature") - 273.15 AS c FROM era5 '
               "GROUP BY latitude ORDER BY latitude")
        res = ctx.sql(sql).to_dataset(dims=["latitude"]).c
        return [(round(float(la), 4), float(v)) for la, v in zip(res.latitude.values, res.values, strict=True)]
    b = np.array([r[1:] for r in _REGIONS], dtype="float64")
    regions = xr.Dataset(
        {"lat_min": (["region"], b[:, 0]), "lat_max": (["region"], b[:, 1]),
         "lon_min": (["region"], b[:, 2]), "lon_max": (["region"], b[:, 3])},
        coords={"region": np.arange(len(_REGIONS))})
    ctx.from_dataset("regions", regions, chunks={"region": len(_REGIONS)})
    sql = """SELECT r.region AS region_id, AVG(a."2m_temperature")-273.15 AS c
             FROM era5 a JOIN regions r ON a.latitude BETWEEN r.lat_min AND r.lat_max
             AND a.longitude BETWEEN r.lon_min AND r.lon_max GROUP BY r.region ORDER BY r.region"""
    res = ctx.sql(sql).to_dataset(dims=["region_id"]).c
    return [(int(i), float(v)) for i, v in zip(res.region_id.values, res.values, strict=True)]


def run_nautilus(ds, day_idx, lat, lon, case):
    nlat, nlon = len(lat), len(lon)
    if case == "03":
        lat_idx = np.arange(nlat, dtype=np.int32)

        def cols(pos, data):
            return {"lat_idx": np.broadcast_to(lat_idx[:, None], (nlat, nlon)).reshape(-1),
                    "tempk": data.reshape(-1)}

        src = ZarrChunkSource(ERA5_URL, "2m_temperature", day_idx, cols, prefetch=PREFETCH)
        out = source(src).apply(KeyedMean("lat_idx", "tempk", "m"), key_columns="lat_idx").run().batches
        return [(round(float(lat[i]), 4), float(m) - 273.15)
                for bb in out for i, m in zip(bb.column("lat_idx").to_pylist(),
                                              bb.column("m").to_pylist(), strict=True)]

    def cols(pos, data):
        return {"latitude": np.broadcast_to(lat[:, None], (nlat, nlon)).reshape(-1),
                "longitude": np.broadcast_to(lon[None, :], (nlat, nlon)).reshape(-1),
                "tempk": data.reshape(-1)}

    tag = make_region_tagger(_REGIONS, "tempk")
    src = ZarrChunkSource(ERA5_URL, "2m_temperature", day_idx, cols, prefetch=PREFETCH)
    out = (source(src).map(tag).apply(KeyedMean("region_id", "tempk", "m"), key_columns="region_id")
           .run().batches)
    return [(int(k), float(m) - 273.15)
            for bb in out for k, m in zip(bb.column("region_id").to_pylist(),
                                          bb.column("m").to_pylist(), strict=True)]


_ENGINES = {"xarray": run_xarray, "xarray-sql": run_xarray_sql, "nautilus": run_nautilus}


def main() -> int:
    engine, case = sys.argv[1], sys.argv[2]
    ds, day_idx, lat, lon = _setup()  # untimed metadata
    t0 = time.perf_counter()
    pairs = _ENGINES[engine](ds, day_idx, lat, lon, case)
    dt = time.perf_counter() - t0
    print(f"TIME={dt:.4f}")
    print(f"DIGEST={_digest(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
