#!/usr/bin/env python3
"""One cold end-to-end read+compute for a single (engine, case) — the unit the cold-read driver spawns.

The compute-only benchmark (``run_bench.py``) pre-loads data and times the kernel. This instead times the
*whole* pipeline — open the store, read the case's real Zarr off the cloud, compute — for one engine, once,
in a fresh process spawned by ``run_e2e.py`` so the read is cold. nautilus reads through its own async
``ZarrSliceSource`` / ``Wb2ForecastSource``; xarray and xarray-sql read through xarray's zarr/gcsfs stack,
as they must.

Each engine returns ``(key, value)`` pairs the driver hashes into a digest — the cross-engine correctness
check — so the three must agree bit-for-bit after rounding, not merely run.

Usage:  e2e_case.py <xarray|xarray-sql|nautilus> <01|02|03|04|05|06>
Prints: ``TIME=<seconds>`` and ``DIGEST=<sha256 of rounded results>``.
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
import pandas as pd
import pyarrow as pa
import xarray as xr
import xarray_sql as xql
from datafusion import SessionConfig

from nautilus import source
from nautilus.benchmarks import GEO_REGIONS, make_region_tagger
from nautilus.operators import KeyedMean

pa.set_cpu_count(1)
sys.path.insert(0, os.path.dirname(__file__))
from _ops import Wb2ForecastSource, ZarrSliceSource  # noqa: E402

ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DAY = os.environ.get("GEOBENCH_ERA5_DAY", "2020-06-01")
PREFETCH = int(os.environ.get("GEOBENCH_PREFETCH", "8"))
# The window cases (02/04) read one full global chunk per hour, so their cold read grows with the window;
# default to one day here (24 hours) to keep an e2e rep tractable — bump for a heavier read.
WINDOW_DAYS = int(os.environ.get("GEOBENCH_E2E_WINDOW_DAYS", "1"))
_CONUS_LAT = (25.0, 50.0)
_CONUS_LON = (235.0, 290.0)
_REGIONS = GEO_REGIONS

_S2_STAC = "https://stac.core.eopf.eodc.eu"
_S2_BBOX = [7.2, 44.5, 7.4, 44.7]
_S2_DATETIME = "2025-04-25/2025-05-05"
_S2_Y0, _S2_X0, _S2_N = 4_000, 6_000, 1_024

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


def _digest(pairs: list[tuple[float, float]]) -> str:
    """Stable hash of (key-rounded-to-4dp, value-rounded-to-2dp) pairs — same numbers → same digest across
    engines (the key rounding matters for the float latitude keys of cases 03/06)."""
    body = ";".join(f"{round(k, 4)}:{round(v, 2)}" for k, v in sorted(pairs))
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _xsql_ctx() -> xql.XarrayContext:
    return xql.XarrayContext(SessionConfig().with_target_partitions(1))


def _naut_pairs(batches, key_col, value_col, key_fn=float, value_fn=float):
    return [
        (key_fn(k), value_fn(v))
        for b in batches
        for k, v in zip(b.column(key_col).to_pylist(), b.column(value_col).to_pylist(), strict=True)
    ]


# --- 03 zonal mean / 06 zonal vector: one ERA5 day, full grid --------------------------------------


def _era5_day_setup():
    ds = xr.open_zarr(ERA5_URL, chunks=None, storage_options={"token": "anon"})
    tindex = ds.indexes["time"]
    day_idx = np.where((tindex >= np.datetime64(DAY)) & (tindex <= np.datetime64(f"{DAY}T23")))[0]
    return {"ds": ds, "time_idx": [int(i) for i in day_idx],
            "lat": ds["latitude"].values, "lon": ds["longitude"].values}


def zonal_xarray(ctx):
    day = ctx["ds"]["2m_temperature"].isel(time=ctx["time_idx"]).astype(np.float64)  # float64: see _window_da
    r = day.mean(["longitude", "time"]) - 273.15
    return [(float(la), float(v)) for la, v in zip(r.latitude.values, r.values, strict=True)]


def zonal_xarray_sql(ctx):
    c = _xsql_ctx()
    c.from_dataset("era5", ctx["ds"][["2m_temperature"]].isel(time=ctx["time_idx"]), chunks={"time": 6})
    res = c.sql(
        'SELECT latitude, AVG("2m_temperature") - 273.15 AS c FROM era5 GROUP BY latitude'
    ).to_dataset(dims=["latitude"]).c
    return [(float(la), float(v)) for la, v in zip(res.latitude.values, res.values, strict=True)]


def zonal_nautilus(ctx):
    nlat, nlon, lat = len(ctx["lat"]), len(ctx["lon"]), ctx["lat"]
    lat_idx = np.arange(nlat, dtype=np.int32)

    def cols(sel, arrs):
        return {"lat_idx": np.broadcast_to(lat_idx[:, None], (nlat, nlon)).reshape(-1),
                "tempk": arrs["2m_temperature"].reshape(-1)}

    src = ZarrSliceSource(
        ERA5_URL, ["2m_temperature"],
        [(t, slice(None), slice(None)) for t in ctx["time_idx"]], cols, prefetch=PREFETCH,
    )
    out = source(src).apply(KeyedMean("lat_idx", "tempk", "m"), key_columns="lat_idx").run().batches
    return _naut_pairs(out, "lat_idx", "m", key_fn=lambda i: float(lat[i]), value_fn=lambda m: m - 273.15)


def _regions_dataset():
    b = np.array([r[1:] for r in _REGIONS], dtype="float64")
    return xr.Dataset(
        {"lat_min": (["region"], b[:, 0]), "lat_max": (["region"], b[:, 1]),
         "lon_min": (["region"], b[:, 2]), "lon_max": (["region"], b[:, 3])},
        coords={"region": np.arange(len(_REGIONS))},
    )


def zvec_xarray(ctx):
    day = ctx["ds"]["2m_temperature"].isel(time=ctx["time_idx"]).astype(np.float64)  # float64: see _window_da
    in_region = xr.concat(
        [(day.latitude >= a) & (day.latitude <= b) & (day.longitude >= c) & (day.longitude <= d)
         for _, a, b, c, d in _REGIONS], dim="region_id")
    r = (day.where(in_region).mean(["time", "latitude", "longitude"]) - 273.15).values
    return [(i, float(v)) for i, v in enumerate(r)]


def zvec_xarray_sql(ctx):
    c = _xsql_ctx()
    c.from_dataset("era5", ctx["ds"][["2m_temperature"]].isel(time=ctx["time_idx"]), chunks={"time": 6})
    c.from_dataset("regions", _regions_dataset(), chunks={"region": len(_REGIONS)})
    res = c.sql(
        """SELECT r.region AS region_id, AVG(a."2m_temperature")-273.15 AS c
           FROM era5 a JOIN regions r ON a.latitude BETWEEN r.lat_min AND r.lat_max
           AND a.longitude BETWEEN r.lon_min AND r.lon_max GROUP BY r.region"""
    ).to_dataset(dims=["region_id"]).c
    return [(int(i), float(v)) for i, v in zip(res.region_id.values, res.values, strict=True)]


def zvec_nautilus(ctx):
    nlat, nlon, lat, lon = len(ctx["lat"]), len(ctx["lon"]), ctx["lat"], ctx["lon"]

    def cols(sel, arrs):
        return {"latitude": np.broadcast_to(lat[:, None], (nlat, nlon)).reshape(-1),
                "longitude": np.broadcast_to(lon[None, :], (nlat, nlon)).reshape(-1),
                "tempk": arrs["2m_temperature"].reshape(-1)}

    src = ZarrSliceSource(
        ERA5_URL, ["2m_temperature"],
        [(t, slice(None), slice(None)) for t in ctx["time_idx"]], cols, prefetch=PREFETCH,
    )
    tag = make_region_tagger(_REGIONS, "tempk")
    out = (source(src).map(tag).apply(KeyedMean("region_id", "tempk", "m"), key_columns="region_id")
           .run().batches)
    return _naut_pairs(out, "region_id", "m", key_fn=int, value_fn=lambda m: m - 273.15)


# --- 02 climatology / 04 anomaly: an ERA5 CONUS window over WINDOW_DAYS days ------------------------


def _era5_window_setup():
    ds = xr.open_zarr(ERA5_URL, chunks=None, storage_options={"token": "anon"})
    tindex = ds.indexes["time"]
    start = np.datetime64(f"{DAY}T00")
    end = start + np.timedelta64(24 * WINDOW_DAYS - 1, "h")
    time_idx = [int(i) for i in np.where((tindex >= start) & (tindex <= end))[0]]
    lat, lon = ds["latitude"].values, ds["longitude"].values
    lat_sel = np.where((lat >= _CONUS_LAT[0]) & (lat <= _CONUS_LAT[1]))[0]
    lon_sel = np.where((lon >= _CONUS_LON[0]) & (lon <= _CONUS_LON[1]))[0]
    hours = tindex[time_idx].hour.values.astype(np.int64)
    return {"ds": ds, "time_idx": time_idx, "lat_sel": lat_sel, "lon_sel": lon_sel,
            "hours": {t: int(h) for t, h in zip(time_idx, hours, strict=True)}}


def _window_da(ctx):
    """The in-memory windowed DataArray, read inside the timed region (xarray / xarray-sql read path). Cast
    to float64: nautilus and DataFusion accumulate the mean in float64, so the reference must too or its
    float32 rounding (a documented xarray artifact) would flip 2-dp values and break the exact digest."""
    return (ctx["ds"]["2m_temperature"].isel(time=ctx["time_idx"])
            .isel(latitude=ctx["lat_sel"], longitude=ctx["lon_sel"]).load().astype(np.float64))


def _cell_gid_base(nlat, nlon):
    li = np.broadcast_to(np.arange(nlat)[:, None], (nlat, nlon))
    oi = np.broadcast_to(np.arange(nlon)[None, :], (nlat, nlon))
    return ((li * nlon + oi) * 24).reshape(-1).astype(np.int64)


def clim_xarray(ctx):
    w = _window_da(ctx)
    nlat, nlon = w.sizes["latitude"], w.sizes["longitude"]
    ref = (w.groupby("time.hour").mean("time") - 273.15).transpose("latitude", "longitude", "hour")
    gid = (_cell_gid_base(nlat, nlon)[:, None] + ref.hour.values[None, :]).reshape(-1)
    return list(zip(gid.tolist(), ref.values.reshape(-1).tolist(), strict=True))


def clim_xarray_sql(ctx):
    w = _window_da(ctx)
    nlat, nlon = w.sizes["latitude"], w.sizes["longitude"]
    c = _xsql_ctx()
    c.from_dataset("era5", w.to_dataset(), chunks={"time": 24})
    res = c.sql(
        "SELECT latitude, longitude, date_part('hour', time) AS hour, "
        'AVG("2m_temperature") - 273.15 AS v FROM era5 '
        "GROUP BY latitude, longitude, date_part('hour', time)"
    ).to_dataset(dims=["latitude", "longitude", "hour"]).v
    res = res.reindex(latitude=w.latitude, longitude=w.longitude).sortby("hour").transpose(
        "latitude", "longitude", "hour")
    gid = (_cell_gid_base(nlat, nlon)[:, None] + res.hour.values[None, :]).reshape(-1)
    return list(zip(gid.tolist(), res.values.reshape(-1).tolist(), strict=True))


def clim_nautilus(ctx):
    nlat, nlon = len(ctx["lat_sel"]), len(ctx["lon_sel"])
    base_gid, hours, lat_sel, lon_sel = _cell_gid_base(nlat, nlon), ctx["hours"], ctx["lat_sel"], ctx["lon_sel"]

    def cols(sel, arrs):
        sub = arrs["2m_temperature"][np.ix_(lat_sel, lon_sel)]
        return {"gid": base_gid + hours[sel[0]], "tempk": sub.reshape(-1)}

    src = ZarrSliceSource(
        ERA5_URL, ["2m_temperature"],
        [(t, slice(None), slice(None)) for t in ctx["time_idx"]], cols, prefetch=PREFETCH,
    )
    out = source(src).apply(KeyedMean("gid", "tempk", "m"), key_columns="gid").run().batches
    return _naut_pairs(out, "gid", "m", key_fn=int, value_fn=lambda m: m - 273.15)


def anom_xarray(ctx):
    w = _window_da(ctx)
    g = w.groupby("time.hour")
    ref = (g - g.mean("time")).transpose("time", "latitude", "longitude").values.reshape(len(ctx["time_idx"]), -1)
    ncell = ref.shape[1]
    rowid = (np.arange(ref.shape[0])[:, None] * ncell + np.arange(ncell)[None, :]).reshape(-1)
    return list(zip(rowid.tolist(), ref.reshape(-1).tolist(), strict=True))


def anom_xarray_sql(ctx):
    w = _window_da(ctx)
    ncell = w.sizes["latitude"] * w.sizes["longitude"]
    c = _xsql_ctx()
    c.from_dataset("era5", w.to_dataset(), chunks={"time": 24})
    sql = """
        WITH clim AS (SELECT latitude, longitude, date_part('hour', time) AS hour,
                             AVG("2m_temperature") AS clim_t FROM era5
                      GROUP BY latitude, longitude, date_part('hour', time))
        SELECT a.time, a.latitude, a.longitude, a."2m_temperature" - c.clim_t AS anomaly
        FROM era5 a JOIN clim c ON a.latitude = c.latitude AND a.longitude = c.longitude
         AND date_part('hour', a.time) = c.hour
    """
    res = c.sql(sql).to_dataset(dims=["time", "latitude", "longitude"]).anomaly
    res = res.reindex(time=w.time, latitude=w.latitude, longitude=w.longitude).transpose(
        "time", "latitude", "longitude").values.reshape(len(ctx["time_idx"]), -1)
    rowid = (np.arange(res.shape[0])[:, None] * ncell + np.arange(ncell)[None, :]).reshape(-1)
    return list(zip(rowid.tolist(), res.reshape(-1).tolist(), strict=True))


def _anom_subtract(b: pa.RecordBatch) -> pa.RecordBatch:
    anom = b.column("tempk").to_numpy() - b.column("clim").to_numpy()
    return pa.record_batch({"rowid": b.column("rowid"), "anom": pa.array(anom)})


def anom_nautilus(ctx):
    nlat, nlon = len(ctx["lat_sel"]), len(ctx["lon_sel"])
    ncell = nlat * nlon
    base_gid, hours, lat_sel, lon_sel = _cell_gid_base(nlat, nlon), ctx["hours"], ctx["lat_sel"], ctx["lon_sel"]
    local = {t: i for i, t in enumerate(ctx["time_idx"])}  # store time index → 0-based window position
    cell = np.arange(ncell, dtype=np.int64)
    sels = [(t, slice(None), slice(None)) for t in ctx["time_idx"]]

    def raw_cols(sel, arrs):
        sub = arrs["2m_temperature"][np.ix_(lat_sel, lon_sel)]
        return {"gid": base_gid + hours[sel[0]], "tempk": sub.reshape(-1),
                "rowid": local[sel[0]] * ncell + cell}

    # The anomaly is a self-join: aggregate the window to a per-(lat,lon,hour) climatology, then join it
    # back to the same window on that gid and subtract — two independent cold reads of the window.
    clim = source(
        ZarrSliceSource(ERA5_URL, ["2m_temperature"], sels, raw_cols, prefetch=PREFETCH)
    ).apply(KeyedMean("gid", "tempk", "clim"), key_columns="gid")
    raw = source(ZarrSliceSource(ERA5_URL, ["2m_temperature"], sels, raw_cols, prefetch=PREFETCH))
    out = raw.join(clim, on="gid").map(_anom_subtract).run().batches
    return _naut_pairs(out, "rowid", "anom", key_fn=int)


# --- 01 NDVI: a Sentinel-2 L2A reflectance window --------------------------------------------------


def _s2_setup():
    from pystac_client import Client

    item = next(Client.open(_S2_STAC).search(
        collections=["sentinel-2-l2a"], bbox=_S2_BBOX, datetime=_S2_DATETIME, max_items=1).items())
    href = item.assets["product"].href
    r10m = xr.open_datatree(href, engine="zarr", chunks={})["measurements/reflectance/r10m"].to_dataset()
    # The projected y/x coordinates of the window: the pixel is keyed by its coordinate, not its position,
    # so the three engines agree even though y descends and the SQL grid reconstruction re-sorts it.
    yc = r10m.y.values[_S2_Y0:_S2_Y0 + _S2_N].astype(np.int64)
    xc = r10m.x.values[_S2_X0:_S2_X0 + _S2_N].astype(np.int64)
    # nautilus reads raw DN off the store, so it must apply the CF decode xarray does (reflectance =
    # DN·scale + offset, fill → NaN) or the offset would not cancel in the NDVI ratio.
    enc = r10m["b04"].encoding
    return {"href": href, "yc": yc, "xc": xc,
            "scale": enc["scale_factor"], "offset": enc["add_offset"], "fill": float(enc["_FillValue"])}


def _pixel_keys(yc, xc):
    return (yc[:, None] * 10_000_000 + xc[None, :]).reshape(-1)  # x < 1e7, so (y, x) → a unique int key


def _s2_window(ctx):
    tree = xr.open_datatree(ctx["href"], engine="zarr", chunks={})
    r10m = tree["measurements/reflectance/r10m"].to_dataset()
    return (r10m[["b04", "b08"]].rename(b04="red", b08="nir")
            .isel(y=slice(_S2_Y0, _S2_Y0 + _S2_N), x=slice(_S2_X0, _S2_X0 + _S2_N)).load())


def ndvi_xarray(ctx):
    scene = _s2_window(ctx)
    ndvi = ((scene.nir - scene.red) / (scene.nir + scene.red)).transpose("y", "x").values.reshape(-1)
    return list(zip(_pixel_keys(ctx["yc"], ctx["xc"]).tolist(), ndvi.tolist(), strict=True))


def ndvi_xarray_sql(ctx):
    scene = _s2_window(ctx)
    c = _xsql_ctx()
    c.from_dataset("scene", scene, chunks={"y": 256, "x": 256})
    res = c.sql(
        "SELECT y, x, (nir - red) / (nir + red) AS ndvi FROM scene"
    ).to_dataset(dims=["y", "x"]).ndvi
    keys = (res.y.values.astype(np.int64)[:, None] * 10_000_000 + res.x.values.astype(np.int64)[None, :])
    ndvi = res.transpose("y", "x").values.reshape(-1)
    return list(zip(keys.reshape(-1).tolist(), ndvi.tolist(), strict=True))


def _ndvi_map(b: pa.RecordBatch) -> pa.RecordBatch:
    red, nir = b.column("red").to_numpy(), b.column("nir").to_numpy()
    return pa.record_batch({"key": b.column("key"), "ndvi": pa.array((nir - red) / (nir + red))})


def ndvi_nautilus(ctx):
    sel = (slice(_S2_Y0, _S2_Y0 + _S2_N), slice(_S2_X0, _S2_X0 + _S2_N))
    keys = _pixel_keys(ctx["yc"], ctx["xc"])
    scale, offset, fill = ctx["scale"], ctx["offset"], ctx["fill"]

    def decode(raw):
        out = raw.astype(np.float64) * scale + offset
        out[raw == fill] = np.nan  # match xarray's mask_and_scale: the nodata DN becomes NaN
        return out.reshape(-1)

    def cols(s, arrs):
        return {"key": keys, "red": decode(arrs["b04"]), "nir": decode(arrs["b08"])}

    src = ZarrSliceSource(
        ctx["href"], ["b04", "b08"], [sel], cols,
        group_path="measurements/reflectance/r10m", prefetch=PREFETCH,
    )
    out = source(src).map(_ndvi_map).run().batches
    return _naut_pairs(out, "key", "ndvi", key_fn=int)


# --- 05 forecast skill: WeatherBench2 Pangu + GraphCast vs ERA5 truth ------------------------------


def _wb2_setup():
    def op(u):
        return xr.open_zarr(u, chunks=None, storage_options={"token": "anon"}, decode_timedelta=True)

    era5 = op(_WB2["era5"])
    pangu = op(_WB2["pangu"])[["2m_temperature"]].sel(time=_WB2_INIT)
    inits = pangu.time.values  # same init labels in both models, but different store positions each
    leads = pangu.prediction_timedelta.values
    init_pos = {m: [int(i) for i in np.where(np.isin(op(_WB2[m]).time.values, inits))[0]]
                for m in ("pangu", "graphcast")}
    valid_max = inits.max() + leads.max()
    truth = era5[["2m_temperature"]].sel(time=slice(_WB2_INIT.start, pd.Timestamp(valid_max)))
    truth_times = truth.time.values
    truth_pos = [int(i) for i in np.where(np.isin(era5.time.values, truth_times))[0]]
    return {"era5": era5, "pangu": pangu, "leads": leads, "inits": inits,
            "init_pos": init_pos, "truth_times": truth_times, "truth_pos": truth_pos,
            "nlat": era5.sizes["latitude"], "nlon": era5.sizes["longitude"]}


def _wb2_ref(ctx):
    """xarray reference RMSE by (model, lead) — the xarray path returns it directly; the xarray-sql and
    nautilus paths compute their own, and the digest cross-checks all three."""
    def op(u):
        return xr.open_zarr(u, chunks=None, storage_options={"token": "anon"}, decode_timedelta=True)

    era5 = ctx["era5"]
    f = xr.concat(
        [op(_WB2[m])[["2m_temperature"]].sel(time=_WB2_INIT) for m in ("pangu", "graphcast")], dim="model"
    ).assign_coords(model=["pangu", "graphcast"], latitude=era5.latitude.values,
                    longitude=era5.longitude.values)["2m_temperature"].load()
    e = era5["2m_temperature"].sel(time=slice(_WB2_INIT.start, pd.Timestamp(ctx["inits"].max() + ctx["leads"].max()))).load()
    per_lead = []
    for lead in ctx["leads"]:
        e_at = e.sel(time=ctx["inits"] + lead)
        diff = f.sel(prediction_timedelta=lead) - e_at.values
        per_lead.append(np.sqrt((diff**2).mean(["time", "latitude", "longitude"])))
    r = xr.concat(per_lead, dim="lead").transpose("model", "lead")
    nlead = len(ctx["leads"])
    return [(mi * nlead + li, float(r.values[mi, li])) for mi in range(2) for li in range(nlead)]


def forecast_xarray(ctx):
    return _wb2_ref(ctx)


def forecast_xarray_sql(ctx):
    def op(u):
        return xr.open_zarr(u, chunks=None, storage_options={"token": "anon"}, decode_timedelta=True)

    era5 = ctx["era5"]
    f = xr.concat(
        [op(_WB2[m])[["2m_temperature"]].sel(time=_WB2_INIT) for m in ("pangu", "graphcast")], dim="model"
    ).assign_coords(model=["pangu", "graphcast"], latitude=era5.latitude.values,
                    longitude=era5.longitude.values).load()
    e = era5[["2m_temperature"]].sel(
        time=slice(_WB2_INIT.start, pd.Timestamp(ctx["inits"].max() + ctx["leads"].max()))).load()
    c = _xsql_ctx()
    c.from_dataset("forecasts", f, chunks={"time": 100})
    c.from_dataset("era5", e, chunks={"time": 100})
    res = c.sql("""
        SELECT f.model, f.prediction_timedelta AS lead,
               SQRT(AVG(POWER(CAST(f."2m_temperature" AS DOUBLE) - e."2m_temperature", 2))) AS rmse
        FROM forecasts f JOIN era5 e ON e.time = f.time + f.prediction_timedelta
          AND e.latitude = f.latitude AND e.longitude = f.longitude
        GROUP BY f.model, f.prediction_timedelta ORDER BY f.model, lead
    """).to_dataset(dims=["model", "lead"]).rmse
    nlead = len(ctx["leads"])
    lead_i = {ld: i for i, ld in enumerate(ctx["leads"])}
    return [(["pangu", "graphcast"].index(str(m)) * nlead + lead_i[ld], float(v))
            for mi, m in enumerate(res.model.values)
            for ld, v in zip(res.lead.values, res.values[mi], strict=True)]


def _sq_err(b: pa.RecordBatch) -> pa.RecordBatch:
    d = b.column("temp_f").to_numpy() - b.column("temp_e").to_numpy()
    return pa.record_batch({"mlkey": b.column("mlkey"), "se": pa.array(d * d)})


def forecast_nautilus(ctx):
    nlat, nlon = ctx["nlat"], ctx["nlon"]
    ncell = nlat * nlon
    cell = np.arange(ncell, dtype=np.int64)
    tt = {np.datetime64(t): i for i, t in enumerate(ctx["truth_times"])}  # valid time → truth window index
    inits, leads = ctx["inits"], ctx["leads"]

    def valid_gid(init_local, lead_i, cells):
        truth_i = tt.get(np.datetime64(inits[init_local] + leads[lead_i]))
        return None if truth_i is None else truth_i * ncell + cells

    forecast = source(Wb2ForecastSource(
        [(0, _WB2["pangu"], ctx["init_pos"]["pangu"]),
         (1, _WB2["graphcast"], ctx["init_pos"]["graphcast"])],
        len(leads), cell, valid_gid, prefetch=PREFETCH,
    ))
    truth_local = {p: i for i, p in enumerate(ctx["truth_pos"])}

    def truth_cols(sel, arrs):
        return {"gid": truth_local[sel[0]] * ncell + cell, "temp_e": arrs["2m_temperature"].reshape(-1)}

    truth = source(ZarrSliceSource(
        _WB2["era5"], ["2m_temperature"],
        [(p, slice(None), slice(None)) for p in ctx["truth_pos"]], truth_cols, prefetch=PREFETCH,
    ))
    out = (forecast.join(truth, on="gid").map(_sq_err)
           .apply(KeyedMean("mlkey", "se", "mse"), key_columns="mlkey").run().batches)
    return _naut_pairs(out, "mlkey", "mse", key_fn=int, value_fn=lambda m: float(m) ** 0.5)


SETUPS = {
    "01": _s2_setup, "02": _era5_window_setup, "03": _era5_day_setup,
    "04": _era5_window_setup, "05": _wb2_setup, "06": _era5_day_setup,
}
RUNNERS = {
    ("xarray", "01"): ndvi_xarray, ("xarray-sql", "01"): ndvi_xarray_sql, ("nautilus", "01"): ndvi_nautilus,
    ("xarray", "02"): clim_xarray, ("xarray-sql", "02"): clim_xarray_sql, ("nautilus", "02"): clim_nautilus,
    ("xarray", "03"): zonal_xarray, ("xarray-sql", "03"): zonal_xarray_sql, ("nautilus", "03"): zonal_nautilus,
    ("xarray", "04"): anom_xarray, ("xarray-sql", "04"): anom_xarray_sql, ("nautilus", "04"): anom_nautilus,
    ("xarray", "05"): forecast_xarray, ("xarray-sql", "05"): forecast_xarray_sql, ("nautilus", "05"): forecast_nautilus,
    ("xarray", "06"): zvec_xarray, ("xarray-sql", "06"): zvec_xarray_sql, ("nautilus", "06"): zvec_nautilus,
}


def main() -> int:
    engine, case = sys.argv[1], sys.argv[2]
    ctx = SETUPS[case]()  # untimed metadata (open stores, resolve indices/coords)
    t0 = time.perf_counter()
    pairs = RUNNERS[(engine, case)](ctx)
    dt = time.perf_counter() - t0
    print(f"TIME={dt:.4f}")
    print(f"DIGEST={_digest(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
