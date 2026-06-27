# Performance change log

A historical record of every change that made nautilus measurably faster or more scalable. Newest
first. The performance change is committed first; this log entry is committed right after, citing that
commit's hash (see the `perf-loop` skill) — so this file is the durable record of *what* we sped up, by
*how much*, and *how we proved* the results were unchanged.

Each entry carries:

- **Commit** — short hash of the change commit (or the PR).
- **Change** — what changed, where (files), and the mechanism.
- **Impact** — the workload and scale it was measured on, the metric before → after, and the factor.
- **Correctness** — how the result was proven identical (a pure speed change must not alter output).

Numbers are from one machine and one run; treat the *factor* as the signal, not the absolute rows/s.

---

## 2026-06-27 — Mailbox single-input fast path

- **Commit:** `fadcb2c`
- **Change:** `Mailbox.get` now short-circuits a single-input stage (every linear pipeline operator) to
  `await self._channels[0].recv()`, skipping the per-`get` `asyncio.ensure_future` Task allocation and
  the `asyncio.wait(FIRST_COMPLETED)` merge that only multi-input fan-in needs.
  `src/nautilus/runtime/mailbox.py`.
- **Impact:** `bench-linear`, 500k rows, batch=64, single process — throughput
  **1.33M → 3.73M rows/s (2.79×)**. The win grows as batch size shrinks (more `get` calls per row).
- **Correctness:** structural digest unchanged.

## 2026-06-27 — Keyed-shuffle bucket cache

- **Commit:** `53e8eb1`
- **Change:** the keyed partitioners (`HashPartitioner`, `KeyGroupPartitioner`) now memoize key → owning
  instance in a per-partitioner cache, so a key is validated and hashed (`msgpack` + `blake2b`) once for
  the life of the partitioner instead of once per row. A high-rate stream of few keys collapses ~1M
  hashes to ~1k. `src/nautilus/runtime/partition.py`.
- **Impact:** `bench-keyed`, 400k rows, parallelism=4, single process — `partition.route_micros`
  **517k → 222k µs (2.3× less)**; throughput **448k → 606k rows/s (1.35×)**. (The residual route cost is
  now the per-row Python loop, not hashing.)
- **Correctness:** routing is byte-identical — same per-instance row counts, so the structural digest
  *and* the full output multiset are unchanged.

## 2026-06-27 — Vectorized `KeyedTumblingSum.process`

- **Commit:** `e5c238b`
- **Change:** replaced the per-row Python loop (`to_pylist()` on three columns, a state `get`/`put` per
  row) with a columnar path: compute each row's window start arithmetically, then Arrow `group_by` to
  partial-sum the batch per `(key, window)` and fold each partial into keyed state once — turning a
  per-row state write into one per distinct `(key, window)`. `src/nautilus/operators.py`.
- **Impact:** `bench-keyed`, 300k rows, single process — `operator.process_micros`
  **1.02M → 221k µs (4.6× less)**; throughput **311k → 968k rows/s (3.11×)**.
- **Correctness:** structural digest and the exact window sums are byte-identical (partial sums fold
  correctly because addition is associative).
