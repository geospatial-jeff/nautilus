# Geospatial benchmark — nautilus vs xarray-sql vs plain xarray

The [xarray-sql geospatial suite](https://github.com/xqlsystems/xarray-sql/tree/main/benchmarks/geospatial)
makes a claim: the geospatial operations we reach for an *array* library to do are, underneath,
**relational** — `GROUP BY`, `JOIN`, column arithmetic. It proves this by expressing each operation in
SQL and showing the SQL answer matches a plain-xarray reference.

nautilus is a relational streaming engine, so the same operations express directly in its Arrow dataflow.
This benchmark drops nautilus in as a **third contender** across all six cases: the same real data, the
same operation three ways, checked to agree, and timed.

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
   `ZarrSliceSource` (and `Wb2ForecastSource` for the two-model forecast join), built on `obstore` +
   zarr-python's async API, prefetching so the next fetch overlaps the current compute — with no xarray
   in the read path. This is the number a user actually pays.

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
| 01 NDVI (elementwise) | — | 0.006 s / 24 MB | 0.064 s / 20 MB | **0.005 s / 25 MB** | **12.3×** | **1.18×** |
| 06 zonal vector (range JOIN) | 5 | 0.490 s / 1246 MB | 0.215 s / ~0 MB | **0.107 s / ~0 MB** | **2.01×** | **4.58×** |
| 03 zonal mean (GROUP BY lat) | 721 | 0.059 s / 149 MB | 0.117 s / 2 MB | 0.132 s / **~0 MB** | 0.88× | 0.45× |
| 02 climatology (GROUP BY lat,lon,hour) | 535 k | 0.007 s / ~0 MB | 0.078 s / 55 MB | 0.084 s / 9 MB | **0.92×** | 0.08× |
| 05 forecast skill (JOIN + RMSE) | 162 k keys | 0.038 s / ~0 MB | 0.435 s / 3 MB | 0.714 s / ~0 MB | 0.61× | 0.05× |
| 04 anomaly (self-JOIN) | 535 k | 0.010 s / ~0 MB | 0.167 s / 63 MB | 0.331 s / 101 MB | 0.50× | 0.03× |

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
  trivial op. The 5-region range join *beats* DataFusion 2× and the array reference 4.6× — the array form
  pays 1.2 GB to rasterize per-region boolean masks, where nautilus tags each pixel in one numpy pass at
  flat memory.
- **High cardinality → now competitive, where it used to collapse.** The 535 k-group climatology lands at
  **parity with DataFusion** (0.92×) and the two high-cardinality joins within ~2× (anomaly 0.50×,
  forecast 0.61×). This is the payoff of vectorizing the keyed hot paths: `KeyedMean` folds each batch
  with `np.bincount` into running per-key sum/count arrays instead of a per-key Python dict, and
  `HashJoin` interns keys through a numpy value→id map. Earlier — with the per-key-Python MVP state
  backend — the same climatology ran ~34× slower than DataFusion and the joins ~20×; see
  `PERFORMANCE_CHANGELOG.md`. The array *reference* is still far ahead on these because a diurnal
  climatology is a native `reshape`+`mean` for it — no grouping at all.
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

Headline: after the keyed-path vectorization, nautilus is fastest on elementwise and low-cardinality
relational ops, at parity-to-within-2× on high-cardinality `GROUP BY`/`JOIN`, and streams all of it at
constant memory; end-to-end its async reader is fastest when the read is large and contiguous. It remains
a general distributed dataflow engine, not a single-node array library.

## Scaling out (multiple workers)

`run(workers=N)` spawns N processes, places the graph across them, and lets the keyed shuffle cross
sockets — nautilus's real multi-core path (in-process `parallelism>1` shares one GIL and only adds
overhead, which is why the p4 column above is ≤ p1). On this **single node** it does not speed these
workloads up, for two structural reasons:

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
