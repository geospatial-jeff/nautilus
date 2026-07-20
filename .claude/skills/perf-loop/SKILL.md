---
name: perf-loop
description: Use when doing performance analysis or optimization of nautilus — finding what is slow, why, and fixing it. Drives the telemetry loop (run a workload at scale, read the facts, change the code, re-run, compare) and keeps results honest with the structural digest.
---

# The nautilus performance loop

nautilus emits facts, never verdicts: a run records self-describing measurements and leaves the analysis
out of the engine, on purpose, so the analysis can live and improve here. This skill *is* that analysis.
The loop is: **run a workload at scale → read the telemetry → change one thing → re-run → compare → log.**

Work one hypothesis at a time. A run produces dozens of numbers; resist fixing four things at once,
because then no delta is attributable. Pick the dominant cost, change it, measure, repeat.

**Every change that makes nautilus faster or more scalable gets an entry in `PERFORMANCE_CHANGELOG.md`
(repo root) — no exceptions.** That file is the committed historical record; step 5 covers exactly what
to write. This applies to any speed/scalability change, whether or not it came from a full loop run.

## 1. Run a workload that actually exercises the engine

The built-in `wordcount` example is tens of rows — every duration sits on the sub-microsecond noise
floor, no channel fills, state never grows. Use the benchmarks (`bench-keyed`, `bench-linear`,
`bench-fanout` in `nautilus.benchmarks`), scaled by environment so the same pipeline serves a quick
check and a real measurement:

```bash
NAUTILUS_BENCH_ROWS=2000000 NAUTILUS_BENCH_BATCH=4096 NAUTILUS_BENCH_KEYS=1000 \
  nautilus run bench-keyed --telemetry counters --save report.json --show markdown
```

Pick the workload to match the question. The clean micro-benchmarks isolate one cost each: `bench-keyed`
(keyed shuffle + per-key state, flushed at end of stream), `bench-linear` (per-batch runtime overhead, no
shuffle or state), `bench-fanout` (per-row Python in a flat-map). The realistic-scenario benchmarks add the
stressors a real stream has, which the clean ones can't reach: `bench-skew` (zipfian hot keys → partition
imbalance, the classic distributed killer — read per-subtask `operator.rows_in`/`process_micros`) and
`bench-backpressure` (a deliberately slow stage → the channel saturates so `edge.queue_depth_hist` /
`edge.send_wait_micros` / `edge.credit_wait_micros` finally populate). The `SyntheticKeyedSource` knobs
(`skew`, `value_spread`, `null_fraction`, `payload_bytes`) compose these if you need a custom
mix. Add `--parallelism <n>` to run that many instances (the keyed shuffle then routes by key); add
`--workers <n>` to spawn that many processes (a true shuffle then crosses a TCP socket —
`benchmarks/baseline.json` carries a
`bench-keyed-dist` entry that does exactly this). Keep total rows fixed when sweeping a knob so throughput
is comparable.

For a baseline cheap enough to iterate on, shrink `NAUTILUS_BENCH_ROWS` — the bottleneck ranking holds at
any scale; only the absolute wall changes.

Two different jobs, two tools: `nautilus run … --save report.json` for *one* run you read to **find** the
bottleneck (step 2), and `nautilus bench <pipeline>` to **measure** throughput rigorously once you have a
change (step 4). The committed baseline lives in `benchmarks/baseline.json`, recorded on the pinned
self-hosted benchmark runner; `nautilus bench-check` re-runs the benchmarks and gates against it, failing
on any regression or output change. On the pinned runner (CI, the `bench-pinned` workflow) throughput is
gated; run it anywhere else and it reports `machine-differs` — the machine-independent digest still gates,
throughput does not, so an off-box run never false-fails. Run it before declaring a change done.

## 2. Read the report

`report.json` is the full surface; `--show markdown` is the digest. The digest carries only raw facts
(every number in it is in the JSON) plus a `derive:` hint line — so compute the ratios yourself, or read
the report in Python and use the query helpers:

```python
from nautilus.runtime.run import run_plan   # or run_local_chain / deploy
rep = result.telemetry
rep.throughput_rows_per_sec()   # headline: total rows_out / wall — the number you compare across runs
rep.by_occupancy()              # (operator_id, busy/wall) per instance, busiest first
rep.by_self_time()              # operators ranked by busy_us (runtime.step_micros)
rep.structural_digest()         # the correctness fingerprint — see step 4
```

Read it through these lenses. Each turns facts into *a place to look*, never a verdict:

- **Occupancy** = `busy_us / wall` per instance (`by_occupancy`). The stage that gates the run but shows
  *low* occupancy is spending wall on something `step_micros` doesn't count: the keyed shuffle
  (`partition.route_micros`), waiting for input (`edge.input_wait_micros`), or cross-process I/O
  (`transport.*`). High occupancy means CPU-bound in its own `process`/`on_eos` — *except a source*,
  whose `step_micros` also counts the awaits its `frames()` performs, so a fully-occupied source may be
  I/O-bound, not compute-bound (next).
- **I/O-bound source** = `io.wait_micros / runtime.step_micros`. A source is the only operator that may
  `await` in its own code, so its `step_micros` counts network/disk wait together with the CPU of building
  batches — it reads as fully busy even when it is blocked on I/O. `io.wait_micros` (the wait a source
  brackets with `ctx.io_wait()`) splits them: `step_micros - io.wait_micros` is the on-CPU time. When wait
  dominates, the lever is concurrency / request coalescing in `frames()`, not faster code. A source with
  *no* `io.wait_micros` is uninstrumented, not necessarily compute-bound — bracket its awaits and re-run
  before concluding.
- **Self-time** (`by_self_time`, `operator.process_micros` / `operator.on_eos_micros` histograms):
  the operator with the most busy time is the CPU bottleneck. Read its `process()` — per-row Python and
  `to_pylist()` materialization are the usual cause; the histogram shows whether it is a few slow batches
  or uniformly slow.
- **Selectivity** = `rows_out / rows_in`: the fan-out (tokenize) or fan-in (aggregation) shape; explains
  where row volume — and therefore downstream cost — is created or collapsed.
- **Saturation** = `edge.queue_depth / edge.queue_capacity`, with `edge.send_wait_micros`: a depth at
  capacity and nonzero send-wait is backpressure — the producer is being slowed by that consumer. Follow
  the chain to the slowest consumer.
- **Skew**: compare `operator.process_micros` and `operator.rows_in` *across subtasks of the same
  operator*. One instance with most of the rows/time is an unbalanced key distribution, not a code cost.
- **State growth**: `state.entries` / `state.keys` (max). This is the only unbounded structure; a high or
  climbing value is both a memory risk and the size of the end-of-stream flush scan.
- **Shuffle cost**: `partition.route_micros` (sum) on the *sending* operator — the keyed shuffle's per-row
  key extraction and hashing, otherwise invisible between process and send.
- **Cross-process** (`--workers` runs, FULL tier): `transport.bytes_sent` (wire volume — wide schemas and
  tiny batches inflate it) and `edge.credit_wait_micros` (producer stalled on flow-control credit).

## 3. Form one hypothesis

State it as a falsifiable sentence tied to a metric and a line of code, e.g. "`Tokenize` self-time is X%
of wall because `process` loops in Python over every row (the `for s in ...to_pylist(): ...split()` in
`Tokenize.process`); vectorizing with Arrow string kernels will cut `operator.process_micros` and raise
throughput." When no lens explains the gating cost, the gap is in the *telemetry*, not the code — in one
of two forms. **Unaccounted:** wall greatly exceeds every accounted metric. **Conflated:** one
load-bearing bucket is most of the wall but mixes two costs you cannot separate — e.g. a source's
`runtime.step_micros`, which counts I/O wait and compute together (this is why `io.wait_micros` exists). A
quick way to expose a conflated bucket: diff against a run with the suspect cost removed (swap the network
source for an in-memory one) — whatever the bucket sheds is what it was hiding. Either way add the
instrument first — a fact, declared in `telemetry/catalog.py`, never a verdict (see the `writing-docs`
standards and the catalog's banned-words lint) — re-run, then continue. The loop improves the measurement
too.

## 4. Change the code, then prove you only changed the speed

Make the one targeted change. Then **guard correctness with the structural digest** — a SHA-256 over only
the facts the computation conserves (topology + row and EOS counts), excluding all timing and batch
counts. With a deterministic source (the benchmarks are), a pure speed change must leave it identical:

```python
before = rep_before.structural_digest()
after  = rep_after.structural_digest()
assert before == after   # same results; you changed only how fast they were produced
```

If the digest changed, the optimization changed the output — revert or fix it before trusting any speed
number.

For the speed number itself, **do not eyeball one run or take the best of a few** — that is how noise
gets logged as a win. Use the benchmark harness, which runs many trials and reports the **median** with
the interquartile range as the spread, stamps the machine, and compares to the committed baseline:

```bash
nautilus bench bench-keyed --rows 2000000          # median ± IQR; auto-compares to the baseline
```

It classifies the change as `REGRESSED` / `IMPROVED` / `unchanged`, where a change counts only if it
clears both a 7% floor and twice the measured noise — so it refuses to call a sub-noise wobble a win. A
digest mismatch shows as `OUTPUT-CHANGED` and always fails. (`measure`/`compare` in `nautilus.bench` give
the same thing in Python.) When the win is real, move the baseline forward with `--update`.

## 5. Log the change, then iterate

A speed or scalability win is not done until it is recorded. Commit in **two steps** so the log can cite
the real hash of the fix:

1. **Commit the performance change** on its own (code + any tests), with a `perf:` subject line.
2. **Then** prepend an entry to `PERFORMANCE_CHANGELOG.md` (repo root, newest first) that references that
   commit's short hash, and **commit the changelog separately**.

Follow the format already in that file; each entry states:

- **Date** and **Commit** — the date, and the short hash of the change commit from step 1 (or the PR).
- **Change** — what changed, which files, and the mechanism (one or two sentences).
- **Impact** — the exact workload and scale (pipeline, rows, batch, parallelism/workers), the **median
  before → after** from `nautilus bench` (not a single run), and the factor. Quote the metric you
  targeted, not just throughput.
- **Correctness** — how you proved the output unchanged: structural digest equal, and for a routing or
  value change, the output multiset equal too.

Only changes the harness classifies as `IMPROVED` go in — a within-noise result is not an entry. Then
re-read the new report — the bottleneck has usually moved to the next stage — and repeat from step 2.
Keep `uv run pytest` green and `nautilus bench-check` clean throughout; a perf change that breaks a test,
regresses another benchmark, or lacks a changelog entry, is not done.

## Tiers and overhead

`--telemetry off` records nothing (the zero-cost baseline); `counters` (default) is the standard set;
`full` adds the byte-accounting walk and `transport.bytes_sent`. To measure telemetry's own cost, run the
same workload at `off` vs `counters` and compare throughput — instrumentation is meant to be cheap, so a
large gap is itself a finding.
