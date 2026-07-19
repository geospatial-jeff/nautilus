"""Large-scale synthetic sources for performance work — the workloads the built-in examples don't reach.

The examples in ``pipelines.py`` are tiny (tens of rows) so they read clearly and run instantly; none
fills a channel, grows state, or pushes a histogram past its low buckets. These sources generate
millions of rows with *no* real-time pacing (unlike :class:`~nautilus.demos.DemoStreamSource`, which
sleeps), so a run is bound by the engine, not a timer — which is what a throughput measurement needs.

Generation is **deterministic** — even the randomized-looking knobs (skew, varied values, nulls)
draw from a fixed seed — so a re-run produces byte-identical input. That is what lets the dev loop trust a
structural digest as an unchanged-output check across a code change, and read a throughput delta as a real
effect rather than input noise.

The clean stream (uniform keys, constant values) is good for *isolating* one cost, but it is not a real
workload. :class:`SyntheticKeyedSource`'s knobs add the realism a real stream has — key **skew**,
**varied values**, **null** keys, a wider **payload** — each isolating a stressor the clean stream cannot
reach; the ``bench-skew`` / ``bench-backpressure`` pipelines wire them up. See the class docstring.

Scale is read from the environment so the same registered pipeline serves both a quick check and a real
benchmark without code edits:

* ``NAUTILUS_BENCH_ROWS``  — total data rows (default 1,000,000)
* ``NAUTILUS_BENCH_BATCH`` — rows per batch (default 4096)
* ``NAUTILUS_BENCH_KEYS``  — distinct keys / word vocabulary (default 1000)
* ``NAUTILUS_BENCH_SKEW`` — zipfian key-skew exponent for ``bench-skew`` (default 1.2)
* ``NAUTILUS_BENCH_DELAY_US`` — per-batch stall for ``bench-backpressure`` (default 200)

The geospatial ``bench-geo-*`` pipelines draw from a gridded field (:class:`SyntheticGridSource`) sized by
its own knobs — ``NAUTILUS_GEO_DAYS`` / ``NAUTILUS_GEO_NLAT`` / ``NAUTILUS_GEO_NLON`` / ``NAUTILUS_GEO_BATCH``
(see :func:`geo_bench_params`).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from time import perf_counter_ns

import numpy as np
import pyarrow as pa

from nautilus.core.operator import Collector, OneInputOperator, SourceOperator
from nautilus.core.records import EOS_FRAME, Batch, Frame

#: Fixed seed for the realism knobs (skew, nulls, varied values). Fixed so a re-run reproduces
#: byte-identical input even with randomized-looking data — the structural digest stays a usable gate.
_SEED = 0x5EED


def passthrough(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Identity map — the linear benchmark's transform. A module-level function (not a lambda) so a
    ``--workers`` run can cloudpickle the operator to a spawned worker."""
    return batch


async def async_passthrough(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Identity ``fetch`` for the async-transform benchmark: one event-loop yield (so the overlap/reorder
    machinery actually runs) then the batch unchanged — no real I/O, so what is measured is the async
    loop's per-batch engine overhead, the async analog of :func:`passthrough`. Identity output keeps the
    structural digest stable. Module-level (not a lambda) so a ``--workers`` run can cloudpickle it.
    """
    await asyncio.sleep(0)
    return batch


async def async_io_wait(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Latency ``fetch`` for the I/O-bound async-transform benchmark: sleeps ``NAUTILUS_BENCH_FETCH_US``
    microseconds (default 1000) to stand in for a real external lookup, then returns the batch unchanged.
    Where :func:`async_passthrough` isolates the engine's per-batch overhead, this exercises the thing an
    async transform exists for — *overlapping* awaited I/O — so throughput should approach
    ``max_in_flight`` batches per fetch-latency, far above the serial `1 / latency`.

    ``NAUTILUS_BENCH_SLOW_EVERY`` (default 0 = uniform) makes roughly one batch in every
    ``NAUTILUS_BENCH_SLOW_EVERY`` sleep ``NAUTILUS_BENCH_SLOW_FACTOR``× longer (default 20), keyed off the
    batch's first key so it stays
    deterministic. That skew is the head-of-line blocking that separates ordered from unordered emission:
    under ordered a slow head pins buffer slots that finished tails could reuse, so ``ordered=False``
    (completion order) reads further ahead and runs faster. Identity output keeps the structural digest
    stable — latency, and hence emission order, is not a digest input — so this benchmarks safely either
    way; module-level so a ``--workers`` run can cloudpickle it."""
    base_us = _env_int("NAUTILUS_BENCH_FETCH_US", 1000)
    slow_every = _env_int("NAUTILUS_BENCH_SLOW_EVERY", 0)
    if slow_every and int(batch.column(0)[0].as_py()) % slow_every == 0:
        base_us *= _env_int("NAUTILUS_BENCH_SLOW_FACTOR", 20)
    await asyncio.sleep(base_us / 1_000_000)
    return batch


class SlowMap(OneInputOperator):
    """Busy-spins ``delay_micros`` per batch, then emits it unchanged — a deterministic CPU-bound
    consumer. Behind a fast source it forces backpressure: the bounded channel fills, the producer stalls,
    and ``edge.queue_depth_hist`` / ``edge.send_wait_micros`` (and, across a socket, ``edge.credit_wait_micros``)
    finally populate. Identity output, so the structural digest is unaffected — only timing is."""

    def __init__(self, delay_micros: int) -> None:
        self._delay_ns = delay_micros * 1000

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        deadline = (
            perf_counter_ns() + self._delay_ns
        )  # busy-wait (not sleep): a real CPU-bound stage
        while perf_counter_ns() < deadline:
            pass
        out.emit(batch)


# Default scale, shared by bench_params and the `nautilus bench` harness so both agree on one baseline.
DEFAULT_ROWS = 1_000_000
DEFAULT_BATCH = 4096
DEFAULT_KEYS = 1000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def bench_params() -> dict[str, int]:
    """Resolve the benchmark scale from the environment (with defaults). One place so every builder and
    any reporting agent reads the same knobs."""
    rows = _env_int("NAUTILUS_BENCH_ROWS", DEFAULT_ROWS)
    batch_rows = _env_int("NAUTILUS_BENCH_BATCH", DEFAULT_BATCH)
    return {
        "rows": rows,
        "batch_rows": batch_rows,
        "num_batches": max(1, -(-rows // batch_rows)),  # ceil; at least one batch
        "key_cardinality": _env_int("NAUTILUS_BENCH_KEYS", DEFAULT_KEYS),
    }


# Default grid for the geospatial benchmarks (a time × lat × lon field): three days of hourly data over a
# 72×221 window — ~1.1M cells, and the (lat, lon, hour) climatology has ~380k groups of three, the
# high-cardinality-small-group aggregation the `bench-geo-climatology` pipeline exists to stress.
GEO_DEFAULT_DAYS = 3
GEO_DEFAULT_NLAT = 72
GEO_DEFAULT_NLON = 221
GEO_DEFAULT_BATCH = 65_536


def geo_bench_params() -> dict[str, int]:
    """Grid scale for the geospatial benchmarks, from the environment (with defaults) — the geo analog of
    :func:`bench_params`. ``NAUTILUS_GEO_DAYS`` sets the number of hourly days (so timesteps = 24×days and
    a climatology group has ``days`` members), ``NAUTILUS_GEO_NLAT`` / ``NAUTILUS_GEO_NLON`` the spatial
    window, ``NAUTILUS_GEO_BATCH`` the rows per emitted batch."""
    return {
        "n_days": _env_int("NAUTILUS_GEO_DAYS", GEO_DEFAULT_DAYS),
        "nlat": _env_int("NAUTILUS_GEO_NLAT", GEO_DEFAULT_NLAT),
        "nlon": _env_int("NAUTILUS_GEO_NLON", GEO_DEFAULT_NLON),
        "rows_per_batch": _env_int("NAUTILUS_GEO_BATCH", GEO_DEFAULT_BATCH),
    }


class SyntheticKeyedSource(SourceOperator):
    """A configurable keyed stream. With every realism knob at its default it is the simple, clean
    stream: ``key`` (``row_index % key_cardinality``), ``value`` (constant 1), ``ts`` (the row index),
    then EOS. The knobs make it resemble a real stream, each isolating one stressor the clean stream
    cannot exercise:

    * ``skew`` — > 0 draws keys from a zipfian distribution (exponent ``skew``) instead of uniform, so a
      few keys are hot. This is the classic source of partition imbalance: across a parallel shuffle one
      instance gets most of the rows (visible as per-subtask ``operator.rows_in`` / ``process_micros`` skew).
    * ``value_spread`` — > 0 makes ``value`` vary over ``[1, value_spread]`` instead of a constant, so an
      aggregation sums real spread.
    * ``null_fraction`` — > 0 makes that fraction of keys null (a real stream has missing keys).
    * ``payload_bytes`` — > 0 adds a fixed-width ``payload`` string column, widening the schema and the
      bytes an Arrow-IPC frame carries across a socket.
    * ``extra_value_cols`` — adds that many constant int columns (cheap schema width).

    Randomized knobs draw from a fixed-seed generator, so the stream is still byte-for-byte reproducible
    and the structural digest remains a valid correctness gate."""

    def __init__(
        self,
        *,
        num_batches: int,
        batch_rows: int,
        key_cardinality: int,
        extra_value_cols: int = 0,
        skew: float = 0.0,
        value_spread: int = 0,
        null_fraction: float = 0.0,
        payload_bytes: int = 0,
    ) -> None:
        self._num_batches = num_batches
        self._batch_rows = batch_rows
        self._key_cardinality = key_cardinality
        self._extra_value_cols = extra_value_cols
        self._skew = skew
        self._value_spread = value_spread
        self._null_fraction = null_fraction
        self._payload_bytes = payload_bytes
        if skew > 0:  # zipfian PMF over [0, K): key 0 is hottest, mass ∝ 1 / (rank ** skew)
            weights = np.arange(1, key_cardinality + 1, dtype=np.float64) ** -skew
            self._pmf = weights / weights.sum()

    async def frames(self) -> AsyncIterator[Frame]:
        n = self._batch_rows
        ones = pa.array(np.ones(n, dtype=np.int64))
        payload = pa.array(["x" * self._payload_bytes] * n) if self._payload_bytes else None
        # Always built; the default (all-knobs-off) path never draws from it, so its output is unchanged.
        rng = np.random.default_rng(_SEED)
        next_ts = 0
        for _b in range(self._num_batches):
            base = np.arange(next_ts, next_ts + n, dtype=np.int64)
            if self._skew > 0:
                key_ids = rng.choice(self._key_cardinality, size=n, p=self._pmf)
            else:
                key_ids = base % self._key_cardinality
            mask = rng.random(n) < self._null_fraction if self._null_fraction > 0 else None
            value = (
                pa.array(rng.integers(1, self._value_spread + 1, size=n))
                if self._value_spread > 0
                else ones
            )
            columns: dict[str, pa.Array] = {
                "key": pa.array(key_ids, mask=mask),
                "value": value,
                "ts": pa.array(base),
            }
            if payload is not None:
                columns["payload"] = payload
            for c in range(self._extra_value_cols):
                columns[f"v{c}"] = ones
            yield Batch(pa.record_batch(columns))
            next_ts += n
        yield EOS_FRAME


class SyntheticTextSource(SourceOperator):
    """Yields ``num_batches`` batches of ``rows_per_batch`` lines, each line ``tokens_per_row`` space-
    separated words drawn round-robin from a ``vocabulary``-sized set (``w0``..``w{vocabulary-1}``).
    Feeds the tokenize → keyed-count fan-out: one short input row explodes into many output rows, so the
    output batch-size histogram and a keyed shuffle on ``word`` both run hot. Deterministic and unpaced.
    """

    def __init__(
        self,
        *,
        num_batches: int,
        rows_per_batch: int,
        tokens_per_row: int,
        vocabulary: int,
    ) -> None:
        self._num_batches = num_batches
        self._rows_per_batch = rows_per_batch
        self._tokens_per_row = tokens_per_row
        self._vocabulary = vocabulary

    async def frames(self) -> AsyncIterator[Frame]:
        token = 0
        for _ in range(self._num_batches):
            lines = []
            for _ in range(self._rows_per_batch):
                words = [f"w{(token + j) % self._vocabulary}" for j in range(self._tokens_per_row)]
                token += self._tokens_per_row
                lines.append(" ".join(words))
            yield Batch(pa.record_batch({"line": pa.array(lines, pa.string())}))
        yield EOS_FRAME


class SyntheticJoinStreamSource(SourceOperator):
    """The large *probe* side of the join benchmark: a ``key`` that recurs over ``[0, key_cardinality)``
    across every batch — the way a real streaming join re-touches each key many times, not once — plus a
    constant ``value_col`` payload (``lval`` by default). ``num_batches`` × ``batch_rows`` rows, then EOS.
    Set ``key_cardinality`` to the total row count and it emits ascending *unique* keys — two such sources
    join 1:1, the growing-both-sides shape a stream-stream join has. Deterministic and unpaced."""

    def __init__(
        self, *, num_batches: int, batch_rows: int, key_cardinality: int, value_col: str = "lval"
    ) -> None:
        self._num_batches = num_batches
        self._batch_rows = batch_rows
        self._key_cardinality = key_cardinality
        self._value_col = value_col

    async def frames(self) -> AsyncIterator[Frame]:
        n = self._batch_rows
        val = pa.array(np.ones(n, dtype=np.int64))
        idx = 0
        for _ in range(self._num_batches):
            keys = pa.array(np.arange(idx, idx + n) % self._key_cardinality)
            yield Batch(pa.record_batch({"key": keys, self._value_col: val}))
            idx += n
        yield EOS_FRAME


class SyntheticJoinTableSource(SourceOperator):
    """The small bounded *build* side of the join benchmark: each key ``0..key_cardinality-1`` exactly
    once with an ``rval`` payload, in one batch, then EOS — the dimension table a stream-table join
    enriches against. Every probe row matches exactly one table row, so it is a 1:1 join whose output row
    count equals the stream's: the cost measured is the join's internals, not a cross-product blow-up.
    """

    def __init__(self, *, key_cardinality: int) -> None:
        self._key_cardinality = key_cardinality

    async def frames(self) -> AsyncIterator[Frame]:
        keys = pa.array(np.arange(self._key_cardinality))
        yield Batch(pa.record_batch({"key": keys, "rval": keys}))
        yield EOS_FRAME


# --- geospatial benchmarks ---------------------------------------------------------------------
#
# The geospatial `bench-geo-*` pipelines run the streaming forms of the xarray-sql geospatial suite (a
# per-pixel map, three GROUP BY reductions at different key cardinalities, and two joins) so the aggregation
# and join hot paths are exercised on gridded data, not just the abstract keyed stream. They draw from
# :class:`SyntheticGridSource` instead of real Zarr; the cross-engine comparison against real ARCO-ERA5 /
# Sentinel-2 lives outside the library (benchmarks/geospatial/), reusing the same operators
# (:class:`~nautilus.operators.KeyedMean`, the region tagger, the map functions below) over a real-data
# source.

#: Five disjoint lat/lon boxes (name, lat_min, lat_max, lon_min, lon_max) over the ERA5 grid — the vector
#: side of the `bench-geo-zonal-vector` range join, and the same boxes the real-data comparison uses.
GEO_REGIONS: list[tuple[str, float, float, float, float]] = [
    ("Sahara", 18.0, 30.0, 0.0, 30.0),
    ("Amazon", -10.0, 5.0, 290.0, 310.0),
    ("Australia_Outback", -30.0, -20.0, 125.0, 140.0),
    ("Greenland", 65.0, 80.0, 300.0, 340.0),
    ("SE_Asia", 5.0, 20.0, 95.0, 110.0),
]


class SyntheticGridSource(SourceOperator):
    """A deterministic ERA5-like field — a ``time × latitude × longitude`` grid of ``2m_temperature``,
    unraveled to one row per cell — that the ``bench-geo-*`` pipelines aggregate and join.

    Every row carries ``gid`` (the ``(lat, lon, hour)`` group id ``(lat_idx·nlon+lon_idx)·24 + hour`` — the
    high-cardinality aggregation key, and the equi-join key) and the value under ``value_col``. The rest are
    opt-in, so each pipeline emits exactly what it keys or reduces on (unused columns would only inflate the
    source's per-batch cost and muddy the operator's self-time): ``lat_index=True`` adds ``lat_idx`` (the
    low-cardinality latitude band); ``coords=True`` adds ``latitude`` / ``longitude`` for the range join;
    ``bands=True`` adds ``red`` / ``nir`` reflectance for the NDVI map; ``replicas>1`` emits that many copies
    of the grid tagged with a ``replica`` id (model×lead forecasts over one truth field — the many side of
    the forecast-skill join). Latitude spans 90..−90 and longitude 0..360 so :data:`GEO_REGIONS` select real
    subsets, and ``gid`` alone never collides across two joined instances.

    The value is a pole-to-equator gradient plus fixed-seed noise, reseeded per timestep so emission order
    can't change it — deterministic, like the other synthetic sources. Emits each timestep's grid in
    ``rows_per_batch``-row batches, then EOS.
    """

    def __init__(
        self,
        *,
        n_days: int,
        nlat: int,
        nlon: int,
        rows_per_batch: int,
        value_col: str = "value",
        lat_index: bool = False,
        coords: bool = False,
        bands: bool = False,
        replicas: int = 1,
    ) -> None:
        self._nt = n_days * 24
        self._nlat = nlat
        self._nlon = nlon
        self._rpb = rows_per_batch
        self._value_col = value_col
        self._lat_index = lat_index
        self._coords = coords
        self._bands = bands
        self._replicas = max(1, replicas)

    async def frames(self) -> AsyncIterator[Frame]:
        nlat, nlon, n = self._nlat, self._nlon, self._nlat * self._nlon
        lat = np.linspace(90.0, -90.0, nlat)
        lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
        lat_idx = np.repeat(np.arange(nlat, dtype=np.int32), nlon)  # per-cell latitude band
        cell = lat_idx.astype(np.int64) * nlon + np.tile(np.arange(nlon, dtype=np.int64), nlat)
        gid_base = cell * 24  # + hour → the (lat, lon, hour) group id, unique per (cell, hour)
        latitude = np.repeat(lat, nlon)
        base = 273.15 + 30.0 * np.cos(np.deg2rad(latitude))  # pole-to-equator gradient (K)
        lat_idx_arr = pa.array(lat_idx) if self._lat_index else None
        gid_lat = pa.array(latitude) if self._coords else None
        gid_lon = pa.array(np.tile(lon, nlat)) if self._coords else None
        for t in range(self._nt):
            rng = np.random.default_rng(_SEED + t)  # per-timestep seed → order-independent
            gid = pa.array(gid_base + t % 24)
            value = pa.array(base + rng.normal(0.0, 3.0, n))
            red = pa.array(rng.uniform(0.02, 0.3, n).astype(np.float32)) if self._bands else None
            nir = pa.array(rng.uniform(0.2, 0.6, n).astype(np.float32)) if self._bands else None
            for r in range(self._replicas):
                cols: dict[str, pa.Array] = {"gid": gid, self._value_col: value}
                if lat_idx_arr is not None:
                    cols["lat_idx"] = lat_idx_arr
                if gid_lat is not None and gid_lon is not None:
                    cols["latitude"], cols["longitude"] = gid_lat, gid_lon
                if red is not None and nir is not None:
                    cols["red"], cols["nir"] = red, nir
                if self._replicas > 1:
                    cols["replica"] = pa.array(np.full(n, r, dtype=np.int32))
                rb = pa.record_batch(cols)
                for j in range(0, n, self._rpb):
                    yield Batch(rb.slice(j, self._rpb))
        yield EOS_FRAME


def make_region_tagger(
    regions: list[tuple[str, float, float, float, float]], value_col: str
) -> Callable[[pa.RecordBatch], pa.RecordBatch]:
    """Build the ``map`` for the raster×vector range join: tag each pixel with the region box it falls in.
    The returned function takes a batch with ``latitude`` / ``longitude`` / ``value_col`` and emits
    ``(region_id, value_col)`` for every pixel-in-box pair — a pixel in no box drops out, and the boxes are
    disjoint so a pixel matches at most one. This is the broadcast form of ``JOIN regions ON latitude
    BETWEEN … AND longitude BETWEEN …``: with only a handful of boxes, broadcasting them and testing each
    pixel beats shuffling every pixel into a hash join. Module-level factory over a plain bounds list, so
    the returned closure cloudpickles to a worker."""
    bounds = [(i, *r[1:]) for i, r in enumerate(regions)]

    def tag(batch: pa.RecordBatch) -> pa.RecordBatch:
        lat = batch.column("latitude").to_numpy(zero_copy_only=False)
        lon = batch.column("longitude").to_numpy(zero_copy_only=False)
        val = batch.column(value_col).to_numpy(zero_copy_only=False)
        ids: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for rid, lat_min, lat_max, lon_min, lon_max in bounds:
            m = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
            if m.any():
                ids.append(np.full(int(m.sum()), rid, dtype=np.int32))
                vals.append(val[m])
        if not ids:
            return pa.record_batch(
                {"region_id": pa.array([], pa.int32()), value_col: pa.array([], pa.float64())}
            )
        return pa.record_batch(
            {"region_id": pa.array(np.concatenate(ids)), value_col: pa.array(np.concatenate(vals))}
        )

    return tag


def ndvi_map(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Per-pixel ``(nir − red) / (nir + red)`` — the NDVI benchmark's map. Module-level so a ``--workers``
    run can cloudpickle the operator."""
    red = batch.column("red").to_numpy()
    nir = batch.column("nir").to_numpy()
    return pa.record_batch(
        {"lat_idx": batch.column("lat_idx"), "ndvi": pa.array((nir - red) / (nir + red))}
    )


def anomaly_subtract(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Subtract the per-group climatology from each observation — the map after the anomaly self-join,
    which carries the raw ``value`` and the joined climatology ``clim``. Module-level for cloudpickle.
    """
    anom = batch.column("value").to_numpy() - batch.column("clim").to_numpy()
    return pa.record_batch({"gid": batch.column("gid"), "anom": pa.array(anom)})


def squared_error(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Squared forecast error ``(temp_f − temp_e)²`` per cell — the map after the forecast-skill join,
    fed to a per-``replica`` mean whose root is RMSE. Module-level for cloudpickle."""
    err = batch.column("temp_f").to_numpy() - batch.column("temp_e").to_numpy()
    return pa.record_batch({"replica": batch.column("replica"), "se": pa.array(err * err)})
