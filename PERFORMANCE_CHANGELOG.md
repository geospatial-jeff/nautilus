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

Throughput is the **median of repeated `nautilus bench` trials** (the harness discards a warmup, reports
the spread, and refuses to call a sub-noise wobble a win); treat the *factor* as the signal, not the
absolute rows/s, and re-baseline per machine. The three 2026-06-27 entries originally carried single-run
(best-of-3) estimates; they were re-measured with the harness, isolating each change against the current
code (before = the change's parent version of its one file), and their numbers below are those medians.

Measured on **macOS-12.3-arm64-arm-64bit · Python 3.12.11 · nautilus 0.0.1**, 7 trials each.

---

## 2026-06-28 — Vectorized keyed shuffle (route via Arrow dictionary-encode)

- **Commit:** `8be9259`
- **Change:** `_route_keyed` (`src/nautilus/runtime/partition.py`) no longer loops over rows in Python
  (build a key tuple, dict-look-up its owning instance, append the row index to a per-instance list). It
  `dictionary_encode`s the key column(s) so the owning instance is computed **once per distinct key**,
  expands that to a per-row `int32` bucket column with `pc.take`, and forms each instance's sub-batch
  with one `pc.filter`. `stable_bucket` and the key-scalar validation are byte-for-byte unchanged — they
  just run once per distinct key instead of once per row — so cross-process routing is identical. A
  multi-column key folds each column's per-row dictionary index into one compact combo id (re-encoded
  after each column so it can never overflow `int64`), reconstructing each distinct key from a
  representative row.
- **Impact (harness; Linux x86_64 · Python 3.12.3):** `bench-keyed` at parallelism 4 (1M rows,
  batch 4096): **1000 keys 284k → 352k rows/s (1.24×)**; **50 keys 649k → 1.50M rows/s (2.32×)** — the
  win grows as key cardinality falls, where the per-row Python loop dominated. Parallelism 8, 1000 keys:
  **257k → 311k rows/s (1.21×)**. (The shuffle only runs at parallelism > 1; a single-owner edge
  short-circuits unchanged.)
- **Correctness:** structural digest identical before and after at every scale measured (`5cf30d1e…`
  for P4/1000 keys, `d2215cab…` for P8/1000, `84f69ee3…` for P4/50). A new byte-identical fuzz oracle
  (`tests/test_partition.py::test_route_matches_per_row_reference_byte_identical_under_fuzz`) pins the
  vectorized rid→instance map — and the within-bucket row order — to the original per-row loop across
  single/multi-column str/int/bool/bytes/null keys, with a dedicated high-cardinality multi-column case
  for the overflow guard. (Tokenize was left per-row: the columnar `utf8_split_whitespace`/`list_flatten`
  form split correctly but corrupted transiently under load — a pyarrow buffer-lifetime issue — and a
  streaming engine cannot ship a nondeterministic tokenizer.)

## 2026-06-27 — Mailbox single-input fast path

- **Commit:** `fadcb2c`
- **Change:** `Mailbox.get` now short-circuits a single-input stage (every linear pipeline operator) to
  `await self._channels[0].recv()`, skipping the per-`get` `asyncio.ensure_future` Task allocation and
  the `asyncio.wait(FIRST_COMPLETED)` merge that only multi-input fan-in needs.
  `src/nautilus/runtime/mailbox.py`.
- **Impact (harness):** `bench-linear`, 500k rows, batch=64, single process — throughput
  **1.44M → 3.54M rows/s (2.45×)**, run-to-run noise ~1%. The win grows as batch size shrinks (more
  `get` calls per row).
- **Correctness:** structural digest identical before and after — re-confirmed by the harness.

## 2026-06-27 — Keyed-shuffle bucket cache

- **Commit:** `53e8eb1`
- **Change:** the keyed partitioners (`HashPartitioner`, `KeyGroupPartitioner`) now memoize key → owning
  instance in a per-partitioner cache, so a key is validated and hashed (`msgpack` + `blake2b`) once for
  the life of the partitioner instead of once per row. A high-rate stream of few keys collapses ~1M
  hashes to ~1k. `src/nautilus/runtime/partition.py`.
- **Impact (harness):** `bench-keyed`, 400k rows, parallelism=4, single process — throughput
  **462k → 651k rows/s (1.41×)**, noise ~1.5%. The mechanism: ~1M per-row `msgpack`+`blake2b` hashes
  collapse to ~1k cached lookups, so the residual route cost is now the per-row Python loop, not hashing.
- **Correctness:** routing is byte-identical — same per-instance row counts, so the structural digest
  (harness-confirmed) *and* the full output multiset are unchanged.

## 2026-06-27 — Vectorized `KeyedTumblingSum.process`

- **Commit:** `e5c238b`
- **Change:** replaced the per-row Python loop (`to_pylist()` on three columns, a state `get`/`put` per
  row) with a columnar path: compute each row's window start arithmetically, then Arrow `group_by` to
  partial-sum the batch per `(key, window)` and fold each partial into keyed state once — turning a
  per-row state write into one per distinct `(key, window)`. `src/nautilus/operators.py`.
- **Impact (harness):** `bench-keyed`, 300k rows, single process — throughput
  **344k → 1.02M rows/s (2.97×)**, noise <1%. The mechanism: the per-row Python loop (and its
  `operator.process_micros`) is replaced by one Arrow `group_by` plus one state write per distinct
  `(key, window)`.
- **Correctness:** structural digest and the exact window sums are byte-identical (partial sums fold
  correctly because addition is associative); the harness re-confirmed the digest.
