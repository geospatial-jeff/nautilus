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
Each entry records the machine it was measured on (the project moved from macOS to a Linux x86_64 box
mid-stream), since a throughput figure is only comparable on the same hardware.

---

## Open performance items (found, not yet done)

Costs measured during the join work (2026-06-29) and left unfixed, with the reason — so the next loop
starts from evidence, not a cold read.

- **Stream-stream join is super-linear (≈O(n²)).** A key-unique 1:1 stream⋈stream at fixed batch 4096
  fell from ~906k rows/s at 100k rows to ~425k at 400k (wall grew 0.34s → 2.83s for 4× the rows). The
  symmetric hash join buffers both sides until EOS and re-probes the *growing* state, so the buffered
  side's grouped index (an `argsort` + `unique` over the whole buffer) rebuilds on every probe — O(n) per
  probe, O(n²) over the run. The stream-table benchmarks (`bench-join`) don't show it: the bounded table is
  indexed once and reused. The real fix is a delta index: a large, rarely-rebuilt main index plus a small
  recently-added delta probed directly and merged amortized. That is feature-sized, not a tweak. A
  constant-factor attempt (store the sort `order` + a zero-copy `Table` instead of reordering every
  buffered column per probe) was tried and reverted: ~12% on stream-stream but a ~5% regression on the
  common stream-table case (a `combine_chunks` per emit) and no change to the asymptote.

- **`HashJoin._encode` per-distinct-key intern.** After the two shipped vectorizations, the residual
  stream-table cost is interning each distinct key to its integer id — a `((type, value),)` tuple build
  plus a dict lookup per distinct key (cProfile: ~30% of the join at 1000 keys). A nested `type → {value:
  id}` map would drop the per-value tuple build; expected ~10–15% on `bench-join`, digest-preserving. Left
  as diminishing returns next to the ~90–125× already shipped.

---

## 2026-07-01 — Unordered async-transform emission (completion order) for stateless maps

- **Commit:** `a943670`
- **Change:** `run_async_transform` (`src/nautilus/runtime/actor.py`) gained a completion-order drain
  (`ordered=False`, stateless-only): it emits any finished fetch in the leading pre-barrier segment instead
  of strictly at the deque head, so a slow fetch no longer pins reorder-buffer slots that finished tails
  could reuse (a watermark/EOS stays a hard barrier). The ordered default is untouched — the two drains
  share an extracted `_emit_data` body. Exposed as `AsyncMapBatch(ordered=)` / `.map_async(ordered=)`;
  `bench-async-io` reads `NAUTILUS_BENCH_ORDERED`, and `async_io_wait` grows an opt-in
  `NAUTILUS_BENCH_SLOW_EVERY`/`_FACTOR` latency skew so a benchmark can create head-of-line blocking.
- **Impact (`nautilus bench bench-async-io`, median of 5 trials; Linux x86_64 · Python 3.12.3):** the win
  *is* head-of-line blocking, so it appears only when the reorder buffer — not raw concurrency — is the
  bottleneck: a small `max_in_flight` with occasional slow fetches. At `max_in_flight=4`, 2 ms base fetch,
  one batch in 40 running 15× slower (400k rows), ordered **1,244,844 → unordered 2,021,356 rows/s
  (1.62×)**. With a wide window that already overlaps everything (`max_in_flight=64`, uniform latency) there
  is nothing to unblock and the two are within noise — so this is an opt-in throughput knob for
  order-insensitive stages, not a free default.
- **Regression:** the ordered path — the default, and the only path a keyed stage may use — is unchanged by
  the `_emit_data` extraction: `bench-async` **39.0M → 41.1M rows/s (+5.4%)**, `bench-async-io` **38.5M →
  38.3M (−0.5%)**, both inside the harness's 7% gate; digest `1bcf9d55d7ca` unchanged.
- **Correctness:** structural digest **byte-identical** ordered vs unordered (`092e4b2fdea9` at the skew
  config) — a stateless map's rows/batches/watermark counts are order-invariant, which is *why* unordered
  is sound and stays out of the digest; a keyed stage is rejected up front. Full async-transform suite green
  (completion-order, marker-barrier, in-flight-peak, digest-equals-ordered, keyed-rejection).

---

## 2026-07-01 — Async-transform reorder loop: O(1)-per-completion wakeups

- **Commit:** `e3d8c91`
- **Change:** `run_async_transform` (`src/nautilus/runtime/actor.py`) woke by rebuilding a set of every
  in-flight fetch and passing it to `asyncio.wait(FIRST_COMPLETED)` each iteration, which re-registers a
  callback on every future in the set per call — so one completion cost O(in-flight). Each fetch now carries
  a single persistent done-callback that sets a shared `asyncio.Event` the loop blocks on, so a completion
  costs O(1) however many fetches overlap.
- **Impact (`nautilus bench bench-async-io`, batch 512, max_in_flight 512, median of 5 trials; Linux
  x86_64 · Python 3.12.3):** the gain tracks how many fetches are actually in flight — the more overlap, the
  more the old per-completion rebuild cost. With an awaited fetch of **100 µs: 35.5M → 38.0M rows/s (+7%)**,
  **2 ms: 32.9M → 37.2M (+13%)**, **4 ms: 29.9M → 34.0M (+13%)**. It is small below that, since a fast fetch
  drains before the next launches so little overlaps: at the default `max_in_flight=8`, `bench-async` moves
  **39.0M → 40.8M (+4.6%)**, within the harness's 7% gate.
- **Correctness:** structural digest **identical** before and after — `1bcf9d55d7ca` (1M rows),
  `64940ffd50c9` (500k) — so emission order is unchanged. `bench-check` green with no sync-path regression:
  the guard's per-access check is gone from synchronous operators, leaving `bench-keyed` within noise
  (−0.8%).

---

## 2026-06-29 — Vectorized HashJoin single-column key encoding

- **Commit:** `9f2dcb3`
- **Change:** After the probe was vectorized (below), `HashJoin._encode` (`src/nautilus/operators.py`)
  became ~92% of the join's time (cProfile): it built a `(type, value)` tuple and did a dict lookup *per
  row* — O(rows) Python. The single-column case (the common one) now mirrors the keyed shuffle:
  `dictionary_encode` finds the distinct values and a per-row index, the value→id intern runs once per
  *distinct* key (factored into `_intern`), and the per-row ids are a single numpy take. The multi-column
  case keeps the per-row fallback.
- **Impact (`nautilus bench bench-join`, median of 5 trials; Linux x86_64 · Python 3.12.3):** at the
  baseline scale (200k rows / batch 4096 / 500 keys, single process) **1,650,587 → 8,700,249 rows/s
  (+423%)**; at 250k / 1000 keys **1.65M → 6.29M rows/s (3.8×)**. Combined with the probe vectorization,
  **~90× over the original** per-key-loop join. The distributed variant (`bench-join-dist`, 4 instances /
  2 workers) moves less — **593k → 681k rows/s** — because the cross-process shuffle, not the join's
  Python, dominates there.
- **Correctness:** structural digest **identical** (`020174c88ba4` at the bench-join baseline config) —
  the key ids are computed differently but the matches, rows, and batching are unchanged, so unlike the
  probe change this one is digest-preserving. Full join suite (`int`≠`bool`, `int32`==`int64`, null keys,
  composite key, parallel co-partition, distributed) green; `bench-check` green.

---

## 2026-06-29 — Vectorized HashJoin probe (drop the per-distinct-key loop)

- **Commit:** `e8e6388`
- **Change:** `HashJoin.process_left` / `process_right` (`src/nautilus/operators.py`) no longer loop over
  the batch's distinct keys in Python (per key: a group-`take`, a probe of the other side's per-key
  buffer, a cross-product `take`, an `emit`, and a `concat` into this side's running buffer). Each side
  now accumulates whole batches in a `_SideBuffer` indexed by an integer key id — one id map shared by
  both inputs, keyed on each scalar's value **and** Python type so it matches the keyed shuffle's
  `msgpack` equality exactly (`int` 1 ≠ `bool` `True`; `int32` 1 = `int64` 1). A batch probes the other
  side in one shot: a vectorized lookup of each row's key-id run (`start`/`count` arrays), then a ragged
  `repeat`/offset expand to build the match index arrays and one `take` per side — no per-key Python, one
  `emit` per call. The buffer is append-only (no per-key `concat`); the other side's grouped index is
  built once and cached until it next grows, so the bounded table in a stream-table join is grouped once
  and reused.
- **Impact (median-of-trials script; Linux x86_64 · Python 3.12.3):** stream-table inner equi-join (a
  large `key`-recurring stream ⋈ a small bounded table, 1:1 match), 250k rows / batch 4096 / 1000 keys,
  single process: **70,260 → 1,650,587 rows/s (23.5×)**, IQR < 0.5% each. The old per-batch
  `operator.process_micros` was ~93 ms and scaled with **distinct keys per batch** (throughput ∝ 1/K:
  422k→110k→57k→15k rows/s at K = 100/500/1000/4000); the new path is flat at ~1.6M across all K, so the
  factor grows with key cardinality (~100× at K = 4000). Measured with a median-of-trials script (warmup
  + 5 trials via the harness's `summarize()`) because at the time `nautilus bench`'s `(source,
  transforms)` pipeline shape couldn't express a two-source join; a first-class `bench-join` /
  `bench-join-dist` harness pipeline and baseline entry now exist (commits `51f709e` / `537256d`), so
  `bench-check` guards the join — including the cross-process co-partitioned shuffle.
- **Correctness:** the output **multiset** is identical, proven old-vs-new on the benchmark input (same
  `rows_out` = 53,248 and same order-independent multiset hash `83ba97cc…` at the 50k probe), and the full
  join suite — cross-product, order-independence, composite key, `int32`==`int64`, `int`≠`bool`, null
  keys, parallel co-partition, distributed — is green. The structural digest **does** change here, but only
  because `operator.batches_out` is a structural metric and the join now emits one batch per `process`
  call instead of one per key (**13,000 → 13** output batches on the 50k probe — a 1000× cut in batch
  fragmentation, a secondary win); no row changed, so for this re-batching change the multiset is the
  correctness anchor, not the digest.

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
- **Impact (harness; Linux x86_64 · Python 3.12.3):** `bench-linear`, 500k rows, batch=64, single process
  — throughput **1.44M → 3.54M rows/s (2.45×)**, run-to-run noise ~1%. The win grows as batch size shrinks
  (more `get` calls per row).
- **Correctness:** structural digest identical before and after — re-confirmed by the harness.

## 2026-06-27 — Keyed-shuffle bucket cache

- **Commit:** `53e8eb1`
- **Change:** the keyed partitioners (`HashPartitioner`, `KeyGroupPartitioner`) now memoize key → owning
  instance in a per-partitioner cache, so a key is validated and hashed (`msgpack` + `blake2b`) once for
  the life of the partitioner instead of once per row. A high-rate stream of few keys collapses ~1M
  hashes to ~1k. `src/nautilus/runtime/partition.py`.
- **Impact (harness; Linux x86_64 · Python 3.12.3):** `bench-keyed`, 400k rows, parallelism=4, single
  process — throughput **462k → 651k rows/s (1.41×)**, noise ~1.5%. The mechanism: ~1M per-row
  `msgpack`+`blake2b` hashes collapse to ~1k cached lookups, so the residual route cost is now the per-row
  Python loop, not hashing.
- **Correctness:** routing is byte-identical — same per-instance row counts, so the structural digest
  (harness-confirmed) *and* the full output multiset are unchanged.

## 2026-06-27 — Vectorized `KeyedTumblingSum.process`

- **Commit:** `e5c238b`
- **Change:** replaced the per-row Python loop (`to_pylist()` on three columns, a state `get`/`put` per
  row) with a columnar path: compute each row's window start arithmetically, then Arrow `group_by` to
  partial-sum the batch per `(key, window)` and fold each partial into keyed state once — turning a
  per-row state write into one per distinct `(key, window)`. `src/nautilus/operators.py`.
- **Impact (harness; Linux x86_64 · Python 3.12.3):** `bench-keyed`, 300k rows, single process —
  throughput **344k → 1.02M rows/s (2.97×)**, noise <1%. The mechanism: the per-row Python loop (and its
  `operator.process_micros`) is replaced by one Arrow `group_by` plus one state write per distinct
  `(key, window)`.
- **Correctness:** structural digest and the exact window sums are byte-identical (partial sums fold
  correctly because addition is associative); the harness re-confirmed the digest.
