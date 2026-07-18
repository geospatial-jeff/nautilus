# Geospatial benchmark — nautilus vs xarray-sql vs plain xarray

The [xarray-sql geospatial suite](https://github.com/xqlsystems/xarray-sql/tree/main/benchmarks/geospatial)
makes a claim: the geospatial operations we reach for an *array* library to do are, underneath,
**relational** — `GROUP BY`, `JOIN`, column arithmetic. It proves this by expressing each operation in
SQL and showing the SQL answer matches a plain-xarray reference.

nautilus is a relational streaming engine, so the same operations express directly in its Arrow
dataflow. This benchmark drops nautilus in as a **third contender** on the spatial cases: the same real
data, the same operation three ways, checked to agree, and timed.

| xarray-sql case | operation | nautilus form |
|---|---|---|
| `01_ndvi` | per-pixel `(nir-red)/(nir+red)` | `.map` (column arithmetic) |
| `02_climatology` | `AVG(temp) GROUP BY lat, lon, hour` | `KeyedMean` on an encoded (lat,lon,hour) id |
| `03_zonal_mean` | `AVG(temp) GROUP BY latitude` | `KeyedMean` (keyed aggregation) |
| `04_anomaly` | climatology CTE self-`JOIN` back to obs | two sources → `KeyedMean` + `HashJoin` on the id |
| `05_forecast_skill` | forecast↔truth `JOIN` on valid_time + RMSE | `HashJoin` → squared-error `.map` → `KeyedMean` |
| `06_zonal_vector` | `AVG … JOIN regions ON lat/lon BETWEEN` | broadcast region-tag `.map` → `KeyedMean` |

Cases 01/03/06 are *spatial* (Sentinel-2, one day of ERA5); 02/04/05 are *temporal* (a 3-day ERA5 window;
WeatherBench2 ML-forecast scoring).

## Two measurements: the compute kernel, and the whole pipeline

The read dominates these workloads (the upstream suite says as much), so the benchmark reports both ends:

1. **Compute-only** (`run_bench.py`) — read the window into memory once, then time only the compute.
   Isolates the engine's kernel. To keep the read out of the timed region for *all* engines it is factored
   out; but that flattered the array reference and hid that nautilus streams. Fair to compare kernels,
   not whole pipelines.
2. **End-to-end cold read** (`run_e2e.py`) — each engine reads the day's ARCO-ERA5 Zarr chunks off GCS
   *itself* and computes, in a fresh process per rep so every read is cold (the reason the upstream
   `run_perf.sh` also forks per rep). **nautilus reads Zarr through its own async source** — `_ops.py`'s
   `ZarrChunkSource`, built on `obstore` + zarr-python's async API, prefetching chunks so the next fetch
   overlaps the current compute — with no xarray in the read path. This is the number a user actually pays.

Notes that apply to both: all engines run single-threaded (DataFusion pinned to one partition — verified
within noise of its 32-core default on these memory-bound queries — numpy is non-BLAS, nautilus is one
actor); the mirrored SQL runs on the pre-sliced day, so the archive `WHERE`-time pruning (a separate
xarray-sql strength) is out of scope; compute-only memory is resident-set growth sampled from `/proc`
(counts DataFusion's Rust heap and Arrow's C++ buffers, unlike `tracemalloc`).

## Results

Real data (Sentinel-2 1024×1024 scene; ARCO-ERA5 one day = 24.9M cells), single-threaded, all engines
agree to floating-point tolerance. Representative run (±~10% machine-to-machine).

**Compute-only** (data pre-loaded; median of 3–7; peak = resident-set growth; ▸groups = keyed cardinality):

| Case | ▸groups | xarray ref | xarray-sql | nautilus | vs sql | vs ref |
|---|--:|--:|--:|--:|--:|--:|
| 01 NDVI (elementwise) | — | 0.008s / 25 MB | 0.085s / 56 MB | **0.004s / 25 MB** | **11.0×** | **1.5×** |
| 03 zonal mean (GROUP BY lat) | 721 | 0.051s / 104 MB | 0.111s / 2 MB | 0.241s / **~0 MB** | 0.46× | 0.21× |
| 06 zonal vector (range JOIN) | 5 | 0.485s / 1246 MB | 0.222s / ~0 MB | **0.124s / ~0 MB** | **1.79×** | **3.91×** |
| 02 climatology (GROUP BY lat,lon,hour) | 535 k | 0.008s / ~0 MB | 0.085s / 56 MB | 2.88s / 260 MB | 0.03× | 0.00× |
| 04 anomaly (self-JOIN) | 535 k | 0.011s / ~0 MB | 0.170s / 34 MB | 3.53s / 169 MB | 0.05× | 0.00× |
| 05 forecast skill (JOIN + RMSE) | 160 k | 0.038s / ~0 MB | 0.449s / 3 MB | 2.09s / ~0 MB | 0.21× | 0.02× |

**End-to-end** (each engine reads the Zarr off GCS itself; median of 5 cold fresh-process reps):

| Case | xarray | xarray-sql | nautilus | vs sql | vs xarray |
|---|--:|--:|--:|--:|--:|
| 03 zonal mean (GROUP BY) | 1.56s | 1.33s | 1.57s | 0.85× | 1.00× |
| 06 zonal vector (range JOIN) | 2.02s | 1.40s | **1.25s** | **1.12×** | **1.62×** |

**The result tracks one variable: keyed cardinality (the ▸groups column).**

- **Low cardinality → nautilus is competitive or fastest.** Elementwise NDVI (no keys) is fastest — its
  zero-copy Arrow arithmetic beats the array reference and is ~11× faster than a DataFusion query whose
  fixed per-query cost dwarfs a trivial op. The range join (5 regions) and 721-group zonal mean stream
  with flat memory; the 721-group `GROUP BY` is ~2× behind DataFusion's fused-Rust aggregation but the
  range join *beats* it, and both crush the array reference (which pays 1.2 GB to rasterize masks).
- **High cardinality → nautilus loses by 5–400×.** The 535 k-group climatology, the 535 k-key self-join,
  and the 160 k-key forecast join are all dominated by nautilus's *per-key Python overhead* — its MVP
  keyed-state backend allocates a `StateScope` per key per batch through a Python dict, and `HashJoin`
  interns keys in Python — where DataFusion aggregates and joins in fused Rust and numpy groups
  vectorially. This is a real, documented limitation of the current in-memory state backend, and the
  clearest signal in the suite: **nautilus's keyed machinery does not yet scale to high-cardinality
  aggregation and joins.**
- **End-to-end, the read dominates and the gap closes.** With the ~1.3 s cold GCS read included — the
  I/O-bound regime nautilus is built for — its prefetching async source overlaps read with compute
  (sequential 3.4 s → prefetch=8 1.35 s) and lands competitive-to-fastest on the read-bound cases.

Headline: nautilus is competitive-to-fastest on elementwise and low-cardinality relational ops and when
I/O dominates, but its Python-level keyed state and join interning make high-cardinality `GROUP BY`/`JOIN`
its clear weak spot — the place a vectorized-Rust engine like DataFusion is simply in a different class.

## Scaling out (multiple workers)

`run(workers=N)` spawns N processes, places the graph across them, and lets the keyed shuffle cross
sockets — nautilus's real multi-core path (in-process `parallelism>1` shares one GIL and only adds
overhead). On this **single node** it does not speed these workloads up, for two structural reasons:

- **Compute (keyed aggregation).** A nautilus *source* is pinned to one instance (the IR rejects a
  parallel source), so every row enters through one actor and the keyed-shuffle *routing* — itself
  Python-per-key — runs there serially. Distributing `KeyedMean` across workers can't get past that:
  535 k-group climatology stays flat and 25 M-row zonal mean gets far *worse* (every row is serialized
  cross-process). Pipeline time (telemetry, excludes spawn):

  | workers | 03 zonal mean (25M rows) | 02 climatology (535k groups) |
  |--:|--:|--:|
  | 1 | 0.24s (1.00×) | 2.90s (1.00×) |
  | 2 | 1.18s (0.21×) | 3.03s (0.96×) |
  | 4 | 1.94s (0.13×) | 2.81s (1.03×) |
  | 8 | 3.73s (0.07×) | 3.55s (0.82×) |

- **I/O (reading Zarr).** Reads are fanned out the idiomatic way — a chunk-index source feeding a parallel
  async reader (`ZarrReadChunk`) across workers — but that doesn't help either: one process with async
  prefetch (`in_flight=8`) already **saturates the network** at ~1.4 s for the day's 24 chunks, so extra
  worker processes on the same NIC only add cross-process transport and run slower.

  | workers | end-to-end read (in_flight=8) |
  |--:|--:|
  | 1 | 1.38s (1.00×) |
  | 2 | 1.79s (0.77×) |
  | 4 | 2.01s (0.69×) |
  | 8 | 1.97s (0.70×) |

The payoff of `run(workers=N)` is genuinely **multi-machine** — each node its own cores *and* its own
network link — which one host can't show. On a single box, nautilus's in-process async concurrency (the
event loop + source prefetch) is the right tool and workers only add overhead. `bench_workers.py`
(single serial source) and `bench_workers_zarr.py` (distributed async reader) reproduce both tables.

## Running

```shell
.venv/bin/python benchmarks/geospatial/run_bench.py         # compute-only, all 6 cases
.venv/bin/python benchmarks/geospatial/run_e2e.py           # end-to-end cold read (03, 06)
.venv/bin/python benchmarks/geospatial/bench_workers.py     # run(workers=N) scale-out, keyed cases
.venv/bin/python benchmarks/geospatial/bench_workers_zarr.py  # distributed Zarr read across workers
GEOBENCH_REPS=5 GEOBENCH_CSV=out.csv .venv/bin/python benchmarks/geospatial/run_bench.py 03
```

Needs the geo extras alongside nautilus (`xarray`, `pandas`, `zarr>=3`, `gcsfs`, `obstore`, `xarray_sql`,
`pystac-client`) and network access to GCS, WeatherBench2, and the EOPF STAC service; cases 01–04/06 fall
back to a synthetic field of the same shape when offline, and case 05 (WeatherBench2) skips cleanly.
`_ops.py` holds the nautilus pieces — `KeyedMean`, the region tagger, the lazy `SlicedSource`, and the
async `ZarrChunkSource`; `_harness.py` holds timing, the peak-RSS sampler, and the scope caveats;
`e2e_case.py` is one cold read+compute that `run_e2e.py` forks.
