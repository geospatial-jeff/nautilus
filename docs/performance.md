# Performance

Because data moves as Arrow batches, per-batch work amortizes and throughput is high for a Python engine.
The numbers below are median rows/second from `nautilus bench` on one modern desktop CPU (single process
unless noted). Yours will differ — run the benchmarks on your own hardware.

| Workload | Throughput | Benchmark |
|---|---|---|
| Stateless, per batch (map / filter) | ~210M rows/s | `bench-linear` |
| Keyed aggregation (group + count) | ~118M rows/s | `bench-keyed` |
| Streaming equi-join | ~66M rows/s | `bench-join` |
| Per-row flat-map (Python per row) | ~9M rows/s | `bench-fanout` |
| Keyed shuffle across 2 workers, over TCP | ~2M rows/s | `bench-keyed --workers 2` |

The drop from the vectorized paths (tens to hundreds of millions of rows/s) to the per-row flat-map
(~9M) is the cost of running Python once per row — prefer the vectorized combinators where they fit.

??? note "Full results — every benchmark"
    Median of 7 trials at 1M rows (batch 4096, 1000 keys) on the same machine, single process unless
    noted. The slower rows are benchmarks that *deliberately* add a stressor — skewed keys, a slow stage,
    a process boundary — not a slower engine.

    | Benchmark | What it stresses | Median rows/s |
    |---|---|---|
    | `bench-chain` | A chain of stateless per-batch operators | ~261M |
    | `bench-linear` | One stateless operator — the per-batch overhead floor | ~210M |
    | `bench-async` | Async `map` — awaited work with no real I/O | ~125M |
    | `bench-keyed` | Keyed shuffle plus per-key count | ~118M |
    | `bench-async-io` | Async `map` overlapping simulated I/O wait | ~114M |
    | `bench-join` | Streaming equi-join, both sides shuffled | ~66M |
    | `bench-backpressure` | A slow stage saturating a bounded channel | ~33M |
    | `bench-skew` | Skewed hot keys — partition imbalance | ~26M |
    | `bench-fanout` | Per-row Python in a flat-map | ~9M |
    | `bench-keyed --workers 2` | Keyed shuffle across 2 processes over TCP | ~2M |

## Run the benchmarks

```bash
nautilus bench bench-keyed --rows 2000000
```

`bench` runs many trials and reports the median with its spread; the pipelines and what each one stresses
are in the table above. Add `--workers N` to spread across processes, so the shuffle crosses a real
socket. `nautilus bench-check` re-runs a committed baseline and fails on a regression.

The [performance changelog](https://github.com/geospatial-jeff/nautilus/blob/main/PERFORMANCE_CHANGELOG.md)
records each optimization and its measured effect.
