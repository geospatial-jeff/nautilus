# Geospatial benchmark — nautilus vs xarray-sql vs plain xarray

The [xarray-sql geospatial suite](https://github.com/xqlsystems/xarray-sql/tree/main/benchmarks/geospatial)
makes a claim: the geospatial operations we reach for an *array* library to do are, underneath,
**relational** — `GROUP BY`, `JOIN`, column arithmetic. It proves this by expressing each operation in
SQL and showing the SQL answer matches a plain-xarray reference.

nautilus is a relational streaming engine, so the same operations express directly in its Arrow dataflow.
This benchmark adds nautilus as a third engine across all six cases: the same real data, the same
operation three ways, checked to agree, and timed.

> **Credit.** The cases, the SQL, and the "geospatial array ops are really relational" thesis are the work
> of [**xarray-sql** by xqlsystems](https://github.com/xqlsystems/xarray-sql) (`benchmarks/geospatial`).
> This directory adapts that suite to drop nautilus in as a third engine.

| xarray-sql case | operation | nautilus form | keyed cardinality |
|---|---|---|--:|
| `01_ndvi` | per-pixel `(nir-red)/(nir+red)` | `.map` (column arithmetic) | — |
| `06_zonal_vector` | `AVG … JOIN regions ON lat/lon BETWEEN` | broadcast region-tag `.map` → `KeyedMean` | 5 |
| `03_zonal_mean` | `AVG(temp) GROUP BY latitude` | `KeyedMean` | 721 |
| `05_forecast_skill` | forecast↔truth `JOIN` on valid_time + RMSE | `HashJoin` → squared-error `.map` → `KeyedMean` | ~162 k join keys |
| `02_climatology` | `AVG(temp) GROUP BY lat, lon, hour` | `KeyedMean` on an encoded (lat,lon,hour) id | 535 k |
| `04_anomaly` | climatology CTE self-`JOIN` back to obs | `KeyedMean` → `HashJoin` on the id → subtract | 535 k |

The nautilus operators here are the library's own — `nautilus.operators.KeyedMean` and the region tagger
in `nautilus.benchmarks`, the same code the `bench-geo-*` CLI benchmarks run — so this comparison never
drifts from what the engine actually ships. Only the *reading* of real data lives in `_ops.py`.

## Two measurements: the compute kernel, and the whole pipeline

The read dominates these workloads (the upstream suite says as much), so the benchmark reports both ends:

1. **Compute-only** (`run_bench.py`) — read the window into memory once, then time only the compute.
   Isolates the engine's kernel. Factoring the read out for *all* engines flatters the array reference
   (its native array layout needs no gridded→relational unravel) and hides that nautilus streams — fair to
   compare kernels, not whole pipelines.
2. **End-to-end cold read** (`run_e2e.py`) — each engine reads the case's real store off the cloud
   *itself* and computes, in a fresh process per rep so every read is cold (the reason the upstream
   `run_perf.sh` also forks per rep). **nautilus reads through its own async source** — `_ops.py`'s
   `ZarrSliceSource` (and `Wb2ForecastSource` for the two-model forecast join), with no xarray in the read
   path. This is the number a user actually pays.

Both modes pin every engine single-threaded (DataFusion to one partition — verified within noise of its
32-core default on these memory-bound queries — numpy non-BLAS, nautilus one actor). nautilus p4 is an
in-process scale-out *probe* (one event loop, one GIL): it only adds shuffle overhead, so it is ≤ p1 here
— real scale-out is `run(workers=N)` across processes (see below). Compute-only memory is resident-set
growth sampled from `/proc` (so it counts DataFusion's Rust heap and Arrow's C++ buffers, which
`tracemalloc` cannot).

## Results

Real data — Sentinel-2 1024×1024 scene, ARCO-ERA5 (one day = 24.9 M cells; a 3-day CONUS window =
1.6 M), WeatherBench2 forecasts — single-threaded, all six cases agree across engines to floating-point
tolerance. Representative run on one machine (±~10% machine-to-machine).

**Compute-only** (data pre-loaded; median of 5; peak = resident-set growth; ordered by keyed cardinality):

| Case | groups | xarray ref | xarray-sql | nautilus p1 | vs sql | vs ref |
|---|--:|--:|--:|--:|--:|--:|
| 01 NDVI (elementwise) | — | 0.006 s / 24 MB | 0.052 s / 20 MB | **0.004 s / 25 MB** | **12.1×** | **1.37×** |
| 06 zonal vector (range JOIN) | 5 | 0.495 s / 1246 MB | 0.178 s / 3 MB | **0.090 s / ~0 MB** | **1.99×** | **5.51×** |
| 03 zonal mean (GROUP BY lat) | 721 | 0.054 s / 149 MB | 0.089 s / 2 MB | 0.124 s / **~0 MB** | 0.72× | 0.44× |
| 02 climatology (GROUP BY lat,lon,hour) | 535 k | 0.006 s / ~0 MB | 0.069 s / 45 MB | **0.031 s / 13 MB** | **2.20×** | 0.21× |
| 05 forecast skill (JOIN + RMSE) | 162 k keys | 0.032 s / ~0 MB | 0.379 s / 2 MB | 0.508 s / 2 MB | 0.75× | 0.06× |
| 04 anomaly (self-JOIN) | 535 k | 0.008 s / ~0 MB | 0.141 s / 43 MB | 0.185 s / 55 MB | 0.76× | 0.04× |

**End-to-end** (each engine reads its store off the cloud itself, cold; median of 3 fresh-process reps;
02/04 over a one-day window to keep the cold read tractable):

| Case | store (read shape) | xarray | xarray-sql | nautilus | vs sql | vs xarray |
|---|---|--:|--:|--:|--:|--:|
| 01 NDVI | Sentinel-2 (one 1024² window) | 10.40 s | 10.91 s | **2.28 s** | **4.78×** | **4.56×** |
| 06 zonal vector | ERA5 (24 global chunks) | 2.98 s | 1.75 s | **1.59 s** | **1.11×** | **1.88×** |
| 03 zonal mean | ERA5 (24 global chunks) | 1.61 s | 1.54 s | 1.55 s | 0.99× | 1.04× |
| 02 climatology | ERA5 (24 global chunks) | 1.55 s | 1.65 s | 1.82 s | 0.90× | 0.85× |
| 04 anomaly | ERA5 (24 chunks, read twice) | 1.51 s | 1.61 s | 3.08 s | 0.52× | 0.49× |
| 05 forecast skill | WeatherBench2 (1600 tiny slices) | 0.84 s | 1.37 s | 3.67 s | 0.37× | 0.23× |


**The result tracks one variable: keyed cardinality (the groups column) — and the relational hot paths
are now vectorized.**

- **No keys, or few → nautilus is fastest.** Elementwise NDVI streams zero-copy Arrow arithmetic, beating
  the array reference and running ~12× faster than a DataFusion query whose fixed per-query cost dwarfs a
  trivial op. The 5-region range join *beats* DataFusion 2× and the array reference 5.5× — the array form
  pays 1.2 GB to rasterize per-region boolean masks, where nautilus tags each pixel in one numpy pass at
  flat memory.
- **High cardinality → now faster, where it used to collapse.** The 535 k-group climatology now runs
  **2.2× faster than DataFusion** (0.031 s vs 0.069 s), and the two high-cardinality joins are within ~1.3×
  (anomaly 0.76×, forecast 0.75×). This is the payoff of vectorizing the keyed hot paths: `KeyedMean` and
  `HashJoin` fold each batch into numpy arrays for non-negative integer keys instead of per-key Python
  state, and a shape-aware scatter-add fold took climatology past parity into a win (see
  `PERFORMANCE_CHANGELOG.md`). Earlier — with the per-key-Python MVP state backend — the same climatology
  ran ~34× slower than DataFusion and the joins ~20×. The array *reference* is still far ahead on these
  because a diurnal climatology is a native `reshape`+`mean` for it — no grouping at all.
- **Constant memory throughout.** nautilus holds ~0–100 MB across every case (it streams batches and keeps
  only per-key state); the array reference spikes to 1.2 GB on the range join and 149 MB on zonal mean. On
  data that does not fit in RAM, that is the difference between running and not.

End-to-end, where the cold read is included, a second axis appears: **the shape of the read, not the
compute.** nautilus's async obstore reader is much faster when the read is one large contiguous window — it
reads the Sentinel-2 scene **4.8× faster than xarray's HTTP/gcsfs stack** (2.3 s vs 10.4 s), so NDVI's whole
pipeline is ~4.6× faster — and stays competitive-to-fastest on the mid-size ERA5-day reads (zonal vector,
zonal mean). It loses where its per-slice model is a poor fit: the forecast case issues 1600 tiny
64×32 reads (two models × 20 inits × 40 leads) whose per-request overhead dwarfs the data, and the anomaly
self-join reads its window twice. Those are read-pattern costs, addressable by coalescing requests — not
the compute-kernel story above.

And nautilus reaches all of this as a general distributed dataflow engine, not a single-node array library
— the same pipeline runs across machines (below), which neither rival here does.

## Scaling out (multiple workers)

`run(workers=N)` spawns N processes, places the graph across them, and lets the keyed shuffle cross
sockets — nautilus's real multi-core path. On this **single node** it does not speed these workloads up,
for two structural reasons:

- **Compute (keyed aggregation).** A nautilus *source* is pinned to one instance (the IR rejects a parallel
  source), so every row enters through one actor and the keyed-shuffle routing runs there serially;
  distributing `KeyedMean` across workers cannot get past that, and now every row also crosses a process
  boundary. Pipeline time (telemetry, excludes process spawn):

  | workers | 03 zonal mean (25M rows) | 02 climatology (535k groups) |
  |--:|--:|--:|
  | 1 | 0.14 s (1.00×) | 0.09 s (1.00×) |
  | 2 | 1.21 s (0.12×) | 1.72 s (0.05×) |
  | 4 | 1.89 s (0.07×) | 2.27 s (0.04×) |
  | 8 | 3.74 s (0.03×) | 3.43 s (0.03×) |

- **I/O (reading Zarr).** Reads fan out the idiomatic way — a chunk-index source feeding a parallel async
  reader (`ZarrReadChunk`) across workers — but one process with async prefetch (`in_flight=8`) already
  **saturates the network**, reading the day's 24 chunks in 1.44 s, so extra workers on the same NIC only
  add cross-process transport:

  | workers | end-to-end read |
  |--:|--:|
  | 1 | 1.44 s (1.00×) |
  | 2 | 1.96 s (0.73×) |
  | 4 | 1.92 s (0.75×) |
  | 8 | 2.06 s (0.70×) |

The payoff of `run(workers=N)` is genuinely **multi-machine** — each node its own cores *and* its own
network link — which one host cannot show. On a single box, nautilus's in-process async concurrency (the
event loop + source prefetch) is the right tool. `bench_workers.py` and `bench_workers_zarr.py` reproduce
both tables.

## Running

```shell
.venv/bin/python benchmarks/geospatial/run_bench.py            # compute-only, all 6 cases
.venv/bin/python benchmarks/geospatial/run_e2e.py              # end-to-end cold read, all 6 cases
.venv/bin/python benchmarks/geospatial/bench_workers.py        # run(workers=N) scale-out, keyed cases
.venv/bin/python benchmarks/geospatial/bench_workers_zarr.py   # distributed Zarr read across workers
GEOBENCH_REPS=5 GEOBENCH_CSV=out.csv .venv/bin/python benchmarks/geospatial/run_bench.py 02
```

Needs the geo extras alongside nautilus (`xarray`, `pandas`, `zarr>=3`, `gcsfs`, `obstore`, `xarray_sql`,
`pystac-client`) and network to GCS, WeatherBench2, and the EOPF STAC service. Compute-only cases 01–04/06
fall back to a synthetic field of the same shape when offline (05 skips cleanly); the end-to-end mode
needs the real stores. `_ops.py` holds only the real-data readers — the lazy in-memory `SlicedSource`, the
async `ZarrSliceSource`, `Wb2ForecastSource`, and the distributed `ZarrReadChunk`; the nautilus operators
come from the library. `_harness.py` holds timing, the peak-RSS sampler, and the compute-only scope
caveats; `e2e_case.py` is one cold read+compute that `run_e2e.py` forks.
