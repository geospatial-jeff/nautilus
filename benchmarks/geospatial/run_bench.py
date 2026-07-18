#!/usr/bin/env python3
"""Three-way geospatial benchmark: xarray reference vs xarray-sql SQL vs nautilus dataflow.

Each case takes one spatial operation from the sibling xarray-sql suite (``/tmp/xarray-sql/
benchmarks/geospatial``), reads the *same real data window* into memory once, and runs it three ways —
the plain-xarray array reference, xarray-sql's DataFusion SQL, and the nautilus Arrow dataflow — then
checks all three agree and times each. The xarray reference and SQL reproduce what the xarray-sql
case's own ``measured(...)`` block computes, so nautilus is dropped into an existing comparison.

**Scope — read `_harness.py` first.** This isolates in-memory COMPUTE: the read is factored out for all
three engines (each `.load()`s its window once, outside the timed region) to compare *kernels*, so these
numbers are NOT the xarray-sql suite's cold-read perf table. For the whole pipeline — where nautilus
reads the Zarr itself via `_ops.ZarrChunkSource` and the read (which dominates) overlaps compute — see
`run_e2e.py`. All engines are pinned single-threaded (nautilus p1, DataFusion 1 partition, non-BLAS
numpy) for a like-for-like row; nautilus p4 is an in-process scale-out probe, GIL-bound, not a parity
comparison (nautilus's real scale-out is `run(workers=N)` across processes).

Cases (spatial first, per the request):

* **01 NDVI** — per-pixel ``(nir-red)/(nir+red)``: array ``apply_ufunc`` = SQL column arithmetic =
  nautilus ``.map``. Data: a real Sentinel-2 L2A scene (EOPF Zarr), or a synthetic scene offline.
* **03 zonal mean** — ``AVG(2m_temperature) GROUP BY latitude``: an array reduction = SQL GROUP BY =
  nautilus ``KeyedMean``. Data: one day of ARCO-ERA5.
* **06 zonal vector** — ``AVG … JOIN regions ON lat/lon BETWEEN``: raster×vector range join = SQL
  range JOIN = nautilus broadcast region-tag + ``KeyedMean``. Data: one day of ARCO-ERA5 + 5 boxes.

Run:  ``.venv/bin/python benchmarks/geospatial/run_bench.py [01 03 06 ...]``  (default: all)
Env:  ``GEOBENCH_REPS`` (default 7), ``GEOBENCH_ERA5_DAY`` (default 2020-06-01),
      ``GEOBENCH_PAR`` (nautilus parallelisms, default ``1,4``), ``GEOBENCH_BATCH`` (rows/batch,
      default 262144), ``GEOBENCH_CSV`` (write results table).
"""

# ruff: noqa: E402 — thread env vars must be set before numpy/pyarrow import, so imports follow code.
from __future__ import annotations

import os

# Pin every math backend to one thread BEFORE numpy/pyarrow import them, so the comparison is
# single-thread-vs-single-thread (DataFusion here is tokio current-thread; the numpy reductions are
# non-BLAS anyway). nautilus p4 is the only multi-instance row, labeled as a scale-out probe.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
import xarray_sql as xql
from datafusion import SessionConfig

from nautilus import source
from nautilus.benchmarks import GEO_REGIONS, make_region_tagger
from nautilus.operators import KeyedMean

pa.set_cpu_count(1)  # keep nautilus's own pyarrow compute single-threaded for parity

sys.path.insert(0, os.path.dirname(__file__))
from _harness import CaseSkipped, Timing, max_abs_diff, measure  # noqa: E402
from _ops import SlicedSource  # noqa: E402

REPS = int(os.environ.get("GEOBENCH_REPS", "7"))
ERA5_DAY = os.environ.get("GEOBENCH_ERA5_DAY", "2020-06-01")
PARALLELISMS = [int(p) for p in os.environ.get("GEOBENCH_PAR", "1,4").split(",")]


def _parse_daemons(value):
    """``host:port,...`` → ``[(host, port), ...]`` or ``None`` — set GEOBENCH_DAEMONS to run nautilus
    across dialed worker daemons instead of one process (the keyed op parallelism should match the count)."""
    if not value:
        return None
    return [(hp.rsplit(":", 1)[0], int(hp.rsplit(":", 1)[1])) for hp in value.split(",")]


_DAEMONS = _parse_daemons(os.environ.get("GEOBENCH_DAEMONS"))


def _exec(stream, par):
    """Run a built stream. With GEOBENCH_DAEMONS set, parallelism>1 runs across the dialed daemon roster
    (the distributed row); parallelism 1 stays single-process (the like-for-like baseline)."""
    return stream.run(daemons=_DAEMONS) if (_DAEMONS and par > 1) else stream.run()
BATCH = int(os.environ.get("GEOBENCH_BATCH", "262144"))
ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_WB2_GRID = "64x32_equiangular_conservative"
_WB2 = {
    "era5": f"gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-{_WB2_GRID}.zarr",
    "pangu": f"gs://weatherbench2/datasets/pangu/2018-2022_0012_{_WB2_GRID}.zarr",
    "graphcast": (
        "gs://weatherbench2/datasets/graphcast/2020/"
        f"date_range_2019-11-16_2021-02-01_12_hours-{_WB2_GRID}.zarr"
    ),
}
_WB2_INIT = slice("2020-01-01", "2020-01-10")

_STAC = "https://stac.core.eopf.eodc.eu"
_S2_BBOX = [7.2, 44.5, 7.4, 44.7]
_S2_DATETIME = "2025-04-25/2025-05-05"
_S2_Y0, _S2_X0, _S2_N = 4_000, 6_000, 1_024

# The five region boxes, shared with the library's bench-geo-zonal-vector pipeline (no divergence).
_REGIONS = GEO_REGIONS
# Temporal cases 02/04 use a bounded CONUS-ish window (3 days, hourly) — small enough to hold the whole
# self-join in memory, large enough to have a real diurnal cycle (24 hours × 3 samples each).
_CONUS = {
    "time": slice("2020-06-01", "2020-06-03T23"),
    "latitude": slice(50.0, 25.0),  # ERA5 latitude descends
    "longitude": slice(235.0, 290.0),
}


def xarray_sql_ctx() -> xql.XarrayContext:
    """A single-partition (single-threaded) DataFusion context, so xarray-sql is measured on one core
    like the other engines. Confirmed to change these memory-bound queries within noise vs the 32-core
    default, so this is fair, not a handicap."""
    return xql.XarrayContext(SessionConfig().with_target_partitions(1))


# --- data loading (once, outside every timed region) -------------------------------------------


def load_era5_day() -> xr.DataArray:
    """ARCO-ERA5 2m_temperature for ``GEOBENCH_ERA5_DAYS`` consecutive days (default 1; each day is
    24×721×1440 ≈ 100 MB), read into memory — the scale knob for cases 03/06. Falls back to a synthetic
    field with the same shape and a realistic pole-to-equator gradient if GCS is unreachable."""
    days = int(os.environ.get("GEOBENCH_ERA5_DAYS", "1"))
    start = np.datetime64(ERA5_DAY)
    end = start + np.timedelta64(24 * days - 1, "h")
    try:
        import gcsfs  # noqa: F401

        ds = xr.open_zarr(ERA5_URL, chunks=None, storage_options={"token": "anon"})
        day = ds["2m_temperature"].sel(time=slice(str(start), str(end))).load()
        print(f"  data: ARCO-ERA5 2m_temperature {days} day(s) from {ERA5_DAY}  {dict(day.sizes)} (real)")
        return day
    except Exception as exc:  # noqa: BLE001
        print(f"  data: ARCO-ERA5 unavailable ({exc}); using synthetic {days}-day field")
        rng = np.random.default_rng(0)
        nt = 24 * days
        lat = np.linspace(90, -90, 721)
        lon = np.linspace(0, 360, 1440, endpoint=False)
        base = 273.15 + 30 * np.cos(np.deg2rad(lat))[None, :, None]
        temp = (base + rng.normal(0, 3, (nt, 721, 1440))).astype("float32")
        return xr.DataArray(
            temp,
            dims=["time", "latitude", "longitude"],
            coords={"time": np.arange(nt), "latitude": lat, "longitude": lon},
            name="2m_temperature",
        )


def load_era5_window() -> xr.DataArray:
    """A 3-day hourly ARCO-ERA5 2m_temperature window over a CONUS-ish box (72×101×221), in memory —
    the input for the diurnal climatology (02) and anomaly (04). Synthetic fallback with a diurnal signal
    if GCS is unreachable."""
    try:
        import gcsfs  # noqa: F401

        ds = xr.open_zarr(ERA5_URL, chunks=None, storage_options={"token": "anon"})
        w = ds["2m_temperature"].sel(**_CONUS).load()
        print(f"  data: ARCO-ERA5 2m_temperature CONUS window {dict(w.sizes)} (real)")
        return w
    except Exception as exc:  # noqa: BLE001
        print(f"  data: ARCO-ERA5 unavailable ({exc}); using synthetic window")
        rng = np.random.default_rng(2)
        time = np.arange("2020-06-01", "2020-06-04", dtype="datetime64[h]")
        lat = np.arange(50.0, 24.9, -0.25)
        lon = np.arange(235.0, 290.1, 0.25)
        hour = time.astype("datetime64[h]").astype(int) % 24
        diurnal = 5 * np.cos((hour - 14) / 24 * 2 * np.pi)[:, None, None]
        base = 288.0 + 0.2 * (40 - lat)[None, :, None]
        temp = (base + diurnal + rng.normal(0, 1, (len(time), len(lat), len(lon)))).astype("float32")
        return xr.DataArray(
            temp, dims=["time", "latitude", "longitude"],
            coords={"time": time, "latitude": lat, "longitude": lon}, name="2m_temperature",
        )


def load_wb2() -> tuple[xr.DataArray, xr.DataArray]:
    """WeatherBench2 Pangu + GraphCast forecasts (stacked on a ``model`` dim) and ERA5 truth, at the coarse
    64×32 grid, read into memory. Raises :class:`CaseSkipped` if the bucket is unreachable — no synthetic
    fallback (the valid-time alignment needs real forecast/truth structure)."""
    try:

        def op(u):
            return xr.open_zarr(u, chunks=None, storage_options={"token": "anon"}, decode_timedelta=True)

        era5 = op(_WB2["era5"])
        pangu = op(_WB2["pangu"])[["2m_temperature"]].sel(time=_WB2_INIT)
        graphcast = op(_WB2["graphcast"])[["2m_temperature"]].sel(time=_WB2_INIT)
        forecasts = xr.concat([pangu, graphcast], dim="model").assign_coords(
            model=["pangu", "graphcast"],
            latitude=era5.latitude.values,
            longitude=era5.longitude.values,
        )
        valid_max = pangu.time.values.max() + pangu.prediction_timedelta.values.max()
        truth = era5[["2m_temperature"]].sel(time=slice(_WB2_INIT.start, pd.Timestamp(valid_max))).load()
        f = forecasts["2m_temperature"].load()
        print(f"  data: WeatherBench2 {dict(f.sizes)} forecasts + {truth.sizes['time']} truth steps (real)")
        return f, truth["2m_temperature"]
    except CaseSkipped:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CaseSkipped(f"WeatherBench2 unavailable ({exc})") from exc


def load_s2_scene() -> xr.Dataset:
    """A real Sentinel-2 L2A red/NIR window (EOPF Zarr), read into memory; synthetic 1024×1024 scene
    on failure so the compute comparison still runs offline."""
    try:
        from pystac_client import Client

        catalog = Client.open(_STAC)
        search = catalog.search(
            collections=["sentinel-2-l2a"], bbox=_S2_BBOX, datetime=_S2_DATETIME, max_items=1
        )
        item = next(search.items())
        tree = xr.open_datatree(item.assets["product"].href, engine="zarr", chunks={})
        r10m = tree["measurements/reflectance/r10m"].to_dataset()
        scene = (
            r10m[["b04", "b08"]]
            .rename(b04="red", b08="nir")
            .isel(y=slice(_S2_Y0, _S2_Y0 + _S2_N), x=slice(_S2_X0, _S2_X0 + _S2_N))
            .load()
        )
        print(f"  data: Sentinel-2 L2A {item.id}  {dict(scene.sizes)} (real)")
        return scene
    except Exception as exc:  # noqa: BLE001
        print(f"  data: Sentinel-2 unavailable ({exc}); using synthetic scene")
        rng = np.random.default_rng(1)
        red = rng.uniform(0.02, 0.3, (_S2_N, _S2_N)).astype("float32")
        nir = rng.uniform(0.2, 0.6, (_S2_N, _S2_N)).astype("float32")
        return xr.Dataset(
            {"red": (["y", "x"], red), "nir": (["y", "x"], nir)},
            coords={"y": np.arange(_S2_N), "x": np.arange(_S2_N)},
        )


def _run_nautilus(make_source, build_stream, par: int) -> list[pa.RecordBatch]:
    """Timed body: build a lazy source (unravels grid→Arrow per slice, so the pivot a row engine pays is
    counted but not pre-materialized) and run the stream at parallelism ``par``."""
    return _exec(build_stream(make_source(), par), par).batches


# --- cases -------------------------------------------------------------------------------------


def case_ndvi() -> dict:
    scene = load_s2_scene()
    red_da, nir_da = scene.red, scene.nir
    ny, nx = red_da.sizes["y"], red_da.sizes["x"]
    n = ny * nx
    red, nir = red_da.values, nir_da.values  # float32, native (y, x) layout
    y_grid, x_grid = red_da["y"].values, red_da["x"].values
    print(f"  NDVI over {n:,} pixels")

    # xarray reference — verbatim from xarray-sql case 01's measured step.
    def xr_ref():
        return ((nir_da - red_da) / (nir_da + red_da)).compute()

    t_xr, ref = measure("xarray", xr_ref, reps=REPS)
    ref_flat = ref.values.reshape(-1)

    # xarray-sql — case 01 SQL WITHOUT the ORDER BY (the correctness check realigns by coordinate label,
    # exactly as the original's assert_grid_close does — so the sort would be dead weight neither the
    # array reference nor nautilus pays).
    ctx = xarray_sql_ctx()
    ctx.from_dataset("scene", scene, chunks={"y": 256, "x": 256})
    sql = "SELECT x, y, (nir - red) / (nir + red) AS ndvi FROM scene"

    def xsql():
        return ctx.sql(sql).to_dataset(dims=["y", "x"]).ndvi

    t_sql, sql_res = measure("xarray-sql", xsql, reps=REPS)

    # nautilus — lazy source over y-blocks; map NDVI per batch, carrying x/y (schema parity with the SQL).
    block_y = max(1, BATCH // nx)
    n_slices = (ny + block_y - 1) // block_y

    def slice_fn(i):
        y0, y1 = i * block_y, min(ny, (i + 1) * block_y)
        h = y1 - y0
        return {
            "y": np.broadcast_to(y_grid[y0:y1, None], (h, nx)).reshape(-1),
            "x": np.broadcast_to(x_grid[None, :], (h, nx)).reshape(-1),
            "red": red[y0:y1].reshape(-1),
            "nir": nir[y0:y1].reshape(-1),
        }

    def make_source():
        return SlicedSource(n_slices, slice_fn, BATCH)

    def build(src, par):
        def ndvi(b: pa.RecordBatch) -> pa.RecordBatch:
            r = b.column("red").to_numpy()
            nr = b.column("nir").to_numpy()
            return pa.RecordBatch.from_arrays(
                [b.column("x"), b.column("y"), pa.array((nr - r) / (nr + r))], names=["x", "y", "ndvi"]
            )

        return source(src).map(ndvi, parallelism=par)

    naut_timings, naut_flat = {}, None
    for par in PARALLELISMS:
        t, out = measure(f"nautilus p{par}", lambda p=par: _run_nautilus(make_source, build, p), reps=REPS)
        naut_timings[par] = t
        if par == 1:  # p1 preserves native grid order, so it aligns with ref positionally
            naut_flat = np.concatenate([b.column("ndvi").to_numpy() for b in out])

    sql_al = sql_res.reindex_like(ref).transpose(*ref.dims)
    d_sql = float(np.nanmax(np.abs(sql_al.values - ref.values)))
    d_naut = float(np.nanmax(np.abs(naut_flat - ref_flat)))
    ok = np.allclose(sql_al.values, ref.values, rtol=1e-6, atol=1e-6, equal_nan=True) and np.allclose(
        naut_flat, ref_flat, rtol=1e-6, atol=1e-6, equal_nan=True
    )
    return _result("01 NDVI (per-pixel arithmetic)", n, t_xr, t_sql, naut_timings, d_sql, d_naut, ok)


def _era5_zonal_mean(day: xr.DataArray) -> dict:
    nt, nlat, nlon = day.sizes["time"], day.sizes["latitude"], day.sizes["longitude"]
    n = nt * nlat * nlon
    vals = day.values  # float32 Kelvin, (time, lat, lon)
    lat = day["latitude"].values
    print(f"  zonal mean: {n:,} cells → {nlat} latitude bands")

    def xr_ref():
        return day.mean(["longitude", "time"]) - 273.15

    t_xr, ref = measure("xarray", xr_ref, reps=REPS)
    ref_by = {round(float(v_lat), 6): float(v) for v_lat, v in zip(ref.latitude.values, ref.values, strict=True)}

    ds = day.to_dataset()
    ctx = xarray_sql_ctx()
    ctx.from_dataset("era5", ds, chunks={"time": 6})
    sql = (
        'SELECT latitude, AVG("2m_temperature") - 273.15 AS air_mean_c '
        "FROM era5 GROUP BY latitude ORDER BY latitude DESC"
    )

    def xsql():
        return ctx.sql(sql).to_dataset(dims=["latitude"]).air_mean_c

    t_sql, sql_res = measure("xarray-sql", xsql, reps=REPS)
    sql_by = {round(float(v_lat), 6): float(v) for v_lat, v in zip(sql_res.latitude.values, sql_res.values, strict=True)}

    # Key on the integer latitude index (nautilus's keyed shuffle routes only str/int/bool/bytes/null,
    # not float); aggregate raw Kelvin (a zero-copy float32 view of the grid) and subtract 273.15 from the
    # 721 output means — matching SQL/xarray, which also convert units after aggregating, not per row.
    lat_idx = np.arange(nlat, dtype=np.int32)  # 721 small indices; int32 halves the key column's footprint

    def slice_fn(t):
        return {
            "lat_idx": np.broadcast_to(lat_idx[:, None], (nlat, nlon)).reshape(-1),
            "tempk": vals[t].reshape(-1),
        }

    def make_source():
        return SlicedSource(nt, slice_fn, BATCH)

    def build(src, par):
        return source(src).apply(
            KeyedMean("lat_idx", "tempk", "mean_k"), key_columns="lat_idx", parallelism=par
        )

    naut_timings, naut_by = {}, None
    for par in PARALLELISMS:
        t, out = measure(f"nautilus p{par}", lambda p=par: _run_nautilus(make_source, build, p), reps=REPS)
        naut_timings[par] = t
        if par == 1:
            naut_by = {
                round(float(lat[i]), 6): float(m) - 273.15
                for b in out
                for i, m in zip(b.column("lat_idx").to_pylist(), b.column("mean_k").to_pylist(), strict=True)
            }

    d_sql, miss_sql = max_abs_diff(sql_by, ref_by)
    d_naut, miss_naut = max_abs_diff(naut_by, ref_by)
    ok = miss_sql == 0 and miss_naut == 0 and d_sql < 1e-3 and d_naut < 1e-3
    return _result("03 zonal mean (GROUP BY latitude)", n, t_xr, t_sql, naut_timings, d_sql, d_naut, ok)


def _regions_dataset() -> xr.Dataset:
    b = np.array([r[1:] for r in _REGIONS], dtype="float64")
    return xr.Dataset(
        {
            "lat_min": (["region"], b[:, 0]),
            "lat_max": (["region"], b[:, 1]),
            "lon_min": (["region"], b[:, 2]),
            "lon_max": (["region"], b[:, 3]),
        },
        coords={"region": np.arange(len(_REGIONS))},
    )


def _era5_zonal_vector(day: xr.DataArray) -> dict:
    nt, nlat, nlon = day.sizes["time"], day.sizes["latitude"], day.sizes["longitude"]
    n = nt * nlat * nlon
    vals = day.values
    lat, lon = day["latitude"].values, day["longitude"].values
    print(f"  zonal vector: {n:,} cells × {len(_REGIONS)} regions (range join)")

    def xr_ref():
        in_region = xr.concat(
            [
                (day.latitude >= a) & (day.latitude <= b) & (day.longitude >= c) & (day.longitude <= d)
                for _, a, b, c, d in _REGIONS
            ],
            dim="region_id",
        )
        return day.where(in_region).mean(["time", "latitude", "longitude"]) - 273.15

    t_xr, ref = measure("xarray", xr_ref, reps=REPS)
    ref_by = {i: float(v) for i, v in enumerate(ref.values)}

    ds = day.to_dataset()
    ctx = xarray_sql_ctx()
    ctx.from_dataset("era5", ds, chunks={"time": 6})
    ctx.from_dataset("regions", _regions_dataset(), chunks={"region": len(_REGIONS)})
    sql = """
        SELECT r.region AS region_id, AVG(a."2m_temperature") - 273.15 AS avg_c, COUNT(*) AS n_obs
        FROM era5 a JOIN regions r
          ON a.latitude BETWEEN r.lat_min AND r.lat_max
         AND a.longitude BETWEEN r.lon_min AND r.lon_max
        GROUP BY r.region ORDER BY r.region
    """

    def xsql():
        return ctx.sql(sql).to_dataset(dims=["region_id"]).avg_c

    t_sql, sql_res = measure("xarray-sql", xsql, reps=REPS)
    sql_by = {int(i): float(v) for i, v in zip(sql_res.region_id.values, sql_res.values, strict=True)}

    tag = make_region_tagger(_REGIONS, "tempk")

    def slice_fn(t):
        return {
            "latitude": np.broadcast_to(lat[:, None], (nlat, nlon)).reshape(-1),
            "longitude": np.broadcast_to(lon[None, :], (nlat, nlon)).reshape(-1),
            "tempk": vals[t].reshape(-1),
        }

    def make_source():
        return SlicedSource(nt, slice_fn, BATCH)

    def build(src, par):
        return (
            source(src)
            .map(tag, parallelism=par)
            .apply(KeyedMean("region_id", "tempk", "mean_k"), key_columns="region_id", parallelism=par)
        )

    naut_timings, naut_by = {}, None
    for par in PARALLELISMS:
        t, out = measure(f"nautilus p{par}", lambda p=par: _run_nautilus(make_source, build, p), reps=REPS)
        naut_timings[par] = t
        if par == 1:
            naut_by = {
                int(k): float(m) - 273.15
                for b in out
                for k, m in zip(b.column("region_id").to_pylist(), b.column("mean_k").to_pylist(), strict=True)
            }

    d_sql, miss_sql = max_abs_diff(sql_by, ref_by)
    d_naut, miss_naut = max_abs_diff(naut_by, ref_by)
    ok = miss_sql == 0 and miss_naut == 0 and d_sql < 1e-2 and d_naut < 1e-2
    return _result(
        "06 zonal vector (raster×vector range JOIN)", n, t_xr, t_sql, naut_timings, d_sql, d_naut, ok
    )


def case_zonal_mean() -> dict:
    return _era5_zonal_mean(load_era5_day())


def case_zonal_vector() -> dict:
    return _era5_zonal_vector(load_era5_day())


def _cell_gids(nlat: int, nlon: int) -> np.ndarray:
    """Per-cell base group id ``(lat_idx*nlon + lon_idx)*24`` — add the timestep's hour (0..23) to get the
    (lat,lon,hour) group id, a single integer key encoding the composite GROUP BY (floats can't be shuffle
    keys). ``reshape(-1)`` of a ``(lat,lon,hour)`` array indexes by exactly this id, so a reference grid
    round-trips to gid order for free."""
    li = np.broadcast_to(np.arange(nlat)[:, None], (nlat, nlon))
    oi = np.broadcast_to(np.arange(nlon)[None, :], (nlat, nlon))
    return ((li * nlon + oi) * 24).reshape(-1).astype(np.int64)


def case_climatology() -> dict:
    w = load_era5_window()
    nt, nlat, nlon = w.sizes["time"], w.sizes["latitude"], w.sizes["longitude"]
    n = nt * nlat * nlon
    lat, lon = w.latitude.values, w.longitude.values
    hours = w["time"].dt.hour.values.astype(np.int64)
    vals = w.values
    base_gid = _cell_gids(nlat, nlon)
    print(f"  climatology: {n:,} cells → {nlat * nlon * 24:,} (lat,lon,hour) groups")

    def xr_ref():
        return w.groupby("time.hour").mean("time") - 273.15

    t_xr, ref = measure("xarray", xr_ref, reps=REPS)
    ref_gid = ref.transpose("latitude", "longitude", "hour").values.reshape(-1)

    ctx = xarray_sql_ctx()
    ctx.from_dataset("era5", w.to_dataset(), chunks={"time": 24})
    sql = (
        "SELECT latitude, longitude, date_part('hour', time) AS hour, "
        'AVG("2m_temperature") - 273.15 AS c FROM era5 '
        "GROUP BY latitude, longitude, date_part('hour', time)"
    )

    def xsql():
        return ctx.sql(sql).to_dataset(dims=["latitude", "longitude", "hour"]).c

    t_sql, sql_res = measure("xarray-sql", xsql, reps=REPS)
    sql_gid = (
        sql_res.reindex(latitude=lat, longitude=lon)
        .sortby("hour")
        .transpose("latitude", "longitude", "hour")
        .values.reshape(-1)
    )

    def slice_fn(t):
        return {"gid": base_gid + hours[t], "tempk": vals[t].reshape(-1)}

    def make_source():
        return SlicedSource(nt, slice_fn, BATCH)

    def build(src, par):
        return source(src).apply(KeyedMean("gid", "tempk", "mean_k"), key_columns="gid", parallelism=par)

    naut_timings, naut_gid = {}, None
    for par in PARALLELISMS:
        t, out = measure(f"nautilus p{par}", lambda p=par: _run_nautilus(make_source, build, p), reps=REPS)
        naut_timings[par] = t
        if par == 1:
            naut_gid = np.full(nlat * nlon * 24, np.nan)
            for b in out:
                naut_gid[b.column("gid").to_numpy()] = b.column("mean_k").to_numpy() - 273.15

    d_sql = float(np.nanmax(np.abs(sql_gid - ref_gid)))
    d_naut = float(np.nanmax(np.abs(naut_gid - ref_gid)))
    ok = np.allclose(sql_gid, ref_gid, atol=1e-2, equal_nan=True) and np.allclose(
        naut_gid, ref_gid, atol=1e-2, equal_nan=True
    )
    return _result("02 climatology (GROUP BY lat,lon,hour)", n, t_xr, t_sql, naut_timings, d_sql, d_naut, ok)


def case_anomaly() -> dict:
    w = load_era5_window()
    nt, nlat, nlon = w.sizes["time"], w.sizes["latitude"], w.sizes["longitude"]
    n = nt * nlat * nlon
    hours = w["time"].dt.hour.values.astype(np.int64)
    vals = w.values
    base_gid = _cell_gids(nlat, nlon)
    cell_idx = np.arange(nlat * nlon, dtype=np.int64)
    print(f"  anomaly: {n:,} cells − diurnal climatology (self-JOIN on lat,lon,hour)")

    def xr_ref():
        g = w.groupby("time.hour")
        return g - g.mean("time")

    t_xr, ref = measure("xarray", xr_ref, reps=REPS)
    ref_arr = ref.transpose("time", "latitude", "longitude").values

    ctx = xarray_sql_ctx()
    ctx.from_dataset("era5", w.to_dataset(), chunks={"time": 24})
    sql = """
        WITH clim AS (
            SELECT latitude, longitude, date_part('hour', time) AS hour,
                   AVG("2m_temperature") AS clim_t
            FROM era5 GROUP BY latitude, longitude, date_part('hour', time))
        SELECT a.time, a.latitude, a.longitude, a."2m_temperature" - c.clim_t AS anomaly
        FROM era5 a JOIN clim c
          ON a.latitude = c.latitude AND a.longitude = c.longitude
         AND date_part('hour', a.time) = c.hour
    """

    def xsql():
        return ctx.sql(sql).to_dataset(dims=["time", "latitude", "longitude"]).anomaly

    t_sql, sql_res = measure("xarray-sql", xsql, reps=REPS)
    sql_arr = sql_res.reindex(
        time=w.time, latitude=w.latitude, longitude=w.longitude
    ).transpose("time", "latitude", "longitude").values

    def raw_slice(t):
        return {
            "t_idx": np.full(nlat * nlon, t, np.int64),
            "cell": cell_idx,
            "gid": base_gid + hours[t],
            "tempk": vals[t].reshape(-1),
        }

    def anomaly(b: pa.RecordBatch) -> pa.RecordBatch:
        a = b.column("tempk").to_numpy() - b.column("clim_k").to_numpy()
        return pa.RecordBatch.from_arrays(
            [b.column("t_idx"), b.column("cell"), pa.array(a)], names=["t_idx", "cell", "anom"]
        )

    def naut_run(par):
        # The anomaly IS a self-join: aggregate the raw obs to a per-(lat,lon,hour) climatology, then join
        # it back to the raw obs on that group id and subtract. Two sources over the same window (the DSL
        # reads each built-in source independently), joined on the encoded gid — nautilus's HashJoin.
        clim = source(SlicedSource(nt, raw_slice, BATCH)).apply(
            KeyedMean("gid", "tempk", "clim_k"), key_columns="gid", parallelism=par
        )
        joined = source(SlicedSource(nt, raw_slice, BATCH)).join(clim, on="gid", parallelism=par)
        return _exec(joined.map(anomaly, parallelism=par), par).batches

    naut_timings, naut_arr = {}, None
    for par in PARALLELISMS:
        t, out = measure(f"nautilus p{par}", lambda p=par: naut_run(p), reps=REPS)
        naut_timings[par] = t
        if par == 1:
            naut_arr = np.full((nt, nlat * nlon), np.nan)
            for b in out:
                naut_arr[b.column("t_idx").to_numpy(), b.column("cell").to_numpy()] = b.column("anom").to_numpy()
            naut_arr = naut_arr.reshape(nt, nlat, nlon)

    d_sql = float(np.nanmax(np.abs(sql_arr - ref_arr)))
    d_naut = float(np.nanmax(np.abs(naut_arr - ref_arr)))
    ok = np.allclose(sql_arr, ref_arr, atol=1e-2, equal_nan=True) and np.allclose(
        naut_arr, ref_arr, atol=1e-2, equal_nan=True
    )
    return _result("04 anomaly (climatology self-JOIN)", n, t_xr, t_sql, naut_timings, d_sql, d_naut, ok)


def case_forecast_skill() -> dict:
    f, e = load_wb2()  # forecasts (model,time,lead,lat,lon); truth (time,lat,lon)
    leads, inits = f.prediction_timedelta.values, f.time.values
    nlead, nlat, nlon = len(leads), f.sizes["latitude"], f.sizes["longitude"]
    ncell = nlat * nlon
    n = f.size
    fv, ev = f.values, e.values
    tt = {np.datetime64(t): i for i, t in enumerate(e.time.values)}  # valid_time → truth index
    print(f"  forecast skill: {n:,} forecast rows JOIN truth → RMSE by (model, lead)")

    def xr_ref():
        per_lead = []
        for lead in leads:
            e_at = e.sel(time=inits + lead)
            diff = f.sel(prediction_timedelta=lead) - e_at.values
            per_lead.append(np.sqrt((diff**2).mean(["time", "latitude", "longitude"])))
        return xr.concat(per_lead, dim="lead").transpose("model", "lead")

    t_xr, ref = measure("xarray", xr_ref, reps=REPS)
    ref_by = {(m, li): float(ref.values[mi, li])
              for mi, m in enumerate(["pangu", "graphcast"]) for li in range(nlead)}

    ctx = xarray_sql_ctx()
    ctx.from_dataset("forecasts", f.to_dataset(), chunks={"time": 100})
    ctx.from_dataset("era5", e.to_dataset(), chunks={"time": 100})
    sql = """
        SELECT f.model, f.prediction_timedelta AS lead,
               SQRT(AVG(POWER(CAST(f."2m_temperature" AS DOUBLE) - e."2m_temperature", 2))) AS rmse
        FROM forecasts f JOIN era5 e
          ON e.time = f.time + f.prediction_timedelta
         AND e.latitude = f.latitude AND e.longitude = f.longitude
        GROUP BY f.model, f.prediction_timedelta ORDER BY f.model, lead
    """

    def xsql():
        return ctx.sql(sql).to_dataset(dims=["model", "lead"]).rmse

    t_sql, sql_res = measure("xarray-sql", xsql, reps=REPS)
    lead_to_i = {lead: i for i, lead in enumerate(leads)}
    sql_by = {(str(m), lead_to_i[ld]): float(v)
              for mi, m in enumerate(sql_res.model.values)
              for ld, v in zip(sql_res.lead.values, sql_res.values[mi], strict=True)}

    def truth_batches():
        return [pa.RecordBatch.from_arrays(
            [pa.array(ti * ncell + np.arange(ncell, dtype=np.int64)), pa.array(ev[ti].reshape(-1))],
            names=["gid", "temp_e"]) for ti in range(ev.shape[0])]

    def fc_batches():
        out = []
        for mi in range(2):
            for ti in range(len(inits)):
                for li in range(nlead):
                    vt = np.datetime64(inits[ti] + leads[li])
                    if vt not in tt:
                        continue
                    out.append(pa.RecordBatch.from_arrays(
                        [pa.array(np.full(ncell, mi * nlead + li, np.int64)),
                         pa.array(tt[vt] * ncell + np.arange(ncell, dtype=np.int64)),
                         pa.array(fv[mi, ti, li].reshape(-1))],
                        names=["mlkey", "gid", "temp_f"]))
        return out

    def sq_err(b: pa.RecordBatch) -> pa.RecordBatch:
        d = b.column("temp_f").to_numpy() - b.column("temp_e").to_numpy()
        return pa.RecordBatch.from_arrays([b.column("mlkey"), pa.array(d * d)], names=["mlkey", "se"])

    def naut_run(par):
        # Forecast skill IS a JOIN + aggregate: align forecast to truth at valid_time (encoded as the join
        # key gid), square the difference, average per (model, lead), root. nautilus HashJoin + KeyedMean.
        truth_stream = source(truth_batches())
        joined = source(fc_batches()).join(truth_stream, on="gid", parallelism=par)
        return (_exec(joined.map(sq_err, parallelism=par)
                .apply(KeyedMean("mlkey", "se", "mse"), key_columns="mlkey", parallelism=par), par).batches)

    naut_timings, naut_by = {}, None
    for par in PARALLELISMS:
        t, out = measure(f"nautilus p{par}", lambda p=par: naut_run(p), reps=REPS)
        naut_timings[par] = t
        if par == 1:
            naut_by = {}
            for b in out:
                for k, mse in zip(b.column("mlkey").to_pylist(), b.column("mse").to_pylist(), strict=True):
                    mi, li = divmod(int(k), nlead)
                    naut_by[(["pangu", "graphcast"][mi], li)] = float(mse) ** 0.5

    d_sql, miss_sql = max_abs_diff(sql_by, ref_by)
    d_naut, miss_naut = max_abs_diff(naut_by, ref_by)
    ok = miss_sql == 0 and miss_naut == 0 and d_sql < 1e-3 and d_naut < 1e-3
    return _result("05 forecast skill (JOIN + RMSE by model,lead)", n, t_xr, t_sql, naut_timings, d_sql, d_naut, ok)


def _result(case, n, t_xr, t_sql, naut, d_sql, d_naut, ok) -> dict:
    return {
        "case": case, "n_rows": n, "xarray": t_xr, "xarray-sql": t_sql,
        "nautilus": naut, "diff_sql": d_sql, "diff_naut": d_naut, "ok": ok,
    }


CASES = {
    "01": case_ndvi,
    "02": case_climatology,
    "03": case_zonal_mean,
    "04": case_anomaly,
    "05": case_forecast_skill,
    "06": case_zonal_vector,
}


def _fmt(t: Timing, n: int) -> str:
    return f"{t.median_s:8.3f}s (min {t.min_s:6.3f}) {t.rows_per_s(n) / 1e6:7.1f}M rows/s  {t.peak_mb:6.0f}MB"


def report(results: list[dict]) -> None:
    print("\n" + "=" * 104)
    print("GEOSPATIAL BENCHMARK — xarray reference vs xarray-sql (DataFusion) vs nautilus  [in-memory compute]")
    print(f"reps={REPS}  batch={BATCH}  parallelisms={PARALLELISMS}  (all engines single-threaded; p4 = scale-out probe)")
    print("=" * 104)
    for r in results:
        n = r["n_rows"]
        check = "✅" if r["ok"] else "❌"
        print(f"\n▸ {r['case']}   ({n:,} input rows)   {check} "
              f"(max|Δ| sql={r['diff_sql']:.2e}, nautilus={r['diff_naut']:.2e})")
        print(f"    {'engine':<16}{'median (min)':>22}{'throughput':>16}{'peakRSS':>10}")
        print(f"    {'xarray ref':<16}{_fmt(r['xarray'], n)}")
        print(f"    {'xarray-sql':<16}{_fmt(r['xarray-sql'], n)}")
        for par, t in r["nautilus"].items():
            tag = "" if par == 1 else "  (scale-out probe, GIL-bound)"
            print(f"    {'nautilus p' + str(par):<16}{_fmt(t, n)}{tag}")

    print("\n" + "-" * 104)
    print("SPEEDUP — nautilus p1 (like-for-like, single-threaded) vs each rival  (>1 = nautilus faster)")
    print(f"    {'case':<46}{'vs xarray-sql':>16}{'vs xarray ref':>16}")
    for r in results:
        t_naut = r["nautilus"][1]
        print(f"    {r['case']:<46}{r['xarray-sql'].median_s / t_naut.median_s:>15.2f}x"
              f"{r['xarray'].median_s / t_naut.median_s:>15.2f}x")

    print("\n" + "-" * 104)
    print("CAVEATS (see _harness.py):")
    for c in [
        "COMPUTE kernel only — read factored out for all engines; NOT the xarray-sql suite's cold-read numbers.",
        "For the whole pipeline (nautilus reads Zarr itself, read overlaps compute), see run_e2e.py.",
        "nautilus p4 is in-process (one event loop, one GIL): it adds shuffle+copy overhead for no compute",
        "  parallelism, so it is ≤ p1 here. Real scale-out is run(workers=N) across processes — out of scope.",
        "Dropped the originals' WHERE/time-pruning (data is one day in RAM) — removes xarray-sql's predicate-pushdown lever.",
        "peakRSS = resident growth during the timed step, sampled from /proc (counts DataFusion Rust + Arrow C++).",
        "On zonal-mean the xarray reference accumulates in float32 — the max|Δ| ~8e-5 is ITS rounding error;",
        "  nautilus and DataFusion accumulate in float64 (agree to ~1e-13). Engines agree only on gap-free data.",
    ]:
        print(f"  • {c}")

    csv_path = os.environ.get("GEOBENCH_CSV")
    if csv_path:
        import csv

        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case", "engine", "n_rows", "median_s", "min_s", "max_s", "peak_mb", "rows_per_s"])
            for r in results:
                n = r["n_rows"]
                for name, t in [("xarray", r["xarray"]), ("xarray-sql", r["xarray-sql"]),
                                *[(f"nautilus_p{p}", tt) for p, tt in r["nautilus"].items()]]:
                    w.writerow([r["case"], name, n, f"{t.median_s:.6f}", f"{t.min_s:.6f}",
                                f"{t.max_s:.6f}", f"{t.peak_mb:.1f}", f"{t.rows_per_s(n):.0f}"])
        print(f"\nwrote {csv_path}")


def main() -> int:
    which = [a for a in sys.argv[1:] if a in CASES] or list(CASES)
    results = []
    for key in which:
        print(f"\n{'─' * 104}\nCASE {key}")
        try:
            results.append(CASES[key]())
        except CaseSkipped as exc:
            print(f"  ⏭ SKIPPED: {exc}")
    if results:
        report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
