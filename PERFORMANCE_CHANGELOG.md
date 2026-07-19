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
  common stream-table case (a `combine_chunks` per emit) and no change to the asymptote. The full delta
  index *was* then built and merged (PR #22) — lazy per-row folding into per-id buckets, a cached
  vectorized path for a stable side, and it did take the synthetic `bench-join-stream` from O(n²) to O(n)
  (5.7× at 1M rows). But it was **reverted**: the per-batch bucket bookkeeping (a Python loop over the
  batch's distinct keys, which the old `argsort`-once path did not have) regressed the *real* join
  workloads that never reach the O(n²) regime — the geospatial join cases `geo-anomaly` (+10–19%) and
  `geo-forecast` (+37%), both moderate-scale / high-cardinality. No real workload joins two large unbounded
  streams, so the win was synthetic and the cost was not. A future attempt must not slow the common,
  high-cardinality, bounded-ish join — measure `geo-anomaly`/`geo-forecast`, not just `bench-join`.

- **No explicit rebalance to opt out of a forward edge.** Equal-width keyless edges now forward `i → i`
  by default (2026-07-03 entry below), which is right when the upstream is evenly loaded. But a keyless
  stage that *creates* skew — a filter that keeps most rows on a few instances — propagates that imbalance
  straight down the forwarded chain, with no way to re-spread it. Spark and Flink keep locality the default
  and expose an explicit `repartition()` / `rebalance()` for exactly this case. A DSL `.rebalance()` that
  forced a `RoundRobinSpec` on the next edge would restore that escape hatch; not built, because no current
  workload needs it — the source's fan-out to every instance already balances the initial spread, and every built-in
  keyless stage is roughly row-preserving.

- **Sentinel-2 source lists STAC items serially.** `Sentinel2ItemSource.frames` awaits each item's STAC
  lookup one at a time, so `io.wait_micros` on the single source is ~0.28s per item (1.76s for 6 items).
  After the decode/reduce fusion that is ~26% of the 6-worker wall (6.82s) — the next bottleneck. Coalescing
  the lookups (a bounded `asyncio.gather` prefetch, or one STAC `/search` POST by id list instead of one GET
  per item)
  would cut it to ~one round-trip; expected to take 6-scene/6-worker toward ~5.4s. Independent of the
  fusion and output-preserving (same rows, same order not required downstream — the reduce is keyed).

---

## 2026-07-18 — HashJoin vectorized integer-key intern: 6× on high-cardinality joins

- **Commit:** `41dc3d9`.
- **Change:** `HashJoin._encode` (`operators.py`) interned each distinct key to its dense integer id
  through a Python per-value loop (`_intern_single`) over `dictionary_encode`'s distinct values — which at
  high cardinality is effectively once per *row* (cProfile of an anomaly self-join: `_intern_single`
  called 2.14M times, 87% of the join). For non-negative integer keys it now interns vectorially: a
  value→id numpy lookup array gathers every row's id and assigns unseen values in one bulk
  `np.unique`+`arange` pass, with no per-key Python. A non-integer batch (e.g. the bool side of an int↔bool
  join) keeps the dict path on a disjoint id space; nulls share the dict null id (null still matches null);
  negative keys fall back on the first batch. Same fix, applied to the join intern, as the 2026-07-18
  `KeyedCount`/`KeyedMean` bincount fold.
- **Impact (`nautilus.bench.measure("bench-join", rows=2M, batch=4096, keys=100000)`, median of 5; Linux
  x86_64):** **7.17M → 43.0M rows/s (6.0×)**. On the geospatial join cases the same fix took the
  **anomaly self-join 0.76s → 0.25s (3×, 0.19× → 0.59× vs xarray-sql)** and **forecast-skill 1.51s → 0.59s
  (2.6×, 0.25× → 0.67×)** — the aggregation half was already vectorized, so this was the remaining bottleneck.
- **Correctness:** structural digest identical before → after (`e0fae4e3…`); full `pytest` green
  (393 passed, incl. the join value/type/null/width-consistency tests); `bench-check` adds no digest
  failure (the lone `bench-join-dist` OUTPUT-CHANGED is pre-existing on clean HEAD).

## 2026-07-18 — KeyedCount integer-key bincount fold: 12–16× on keyed aggregation (parity with DataFusion)

- **Commit:** `b6a6330`.
- **Change:** for non-negative integer keys — the common case (indices, hashed buckets) — `KeyedCount`
  (`operators.py`) now accumulates counts in a numpy array
  indexed by the key value (`np.bincount` per batch, one vectorized add into the running array) instead of
  `pyarrow value_counts` → `to_pylist` → per-key keyed-state fold. That removes *all* per-key Python from
  both the hot path and the end-of-stream flush; other key types keep the keyed-state fold, and null keys
  are counted as their own group either way. The geospatial benchmark's `KeyedMean` carries the same fast
  path (running per-key `sum`/`count`).
- **Impact (`nautilus.bench.measure("bench-keyed", rows=1M, batch=4096)`, median of 7, warmup 1; Linux
  x86_64):** **keys=1000 10.3M → 122M rows/s (11.8×)** and **keys=500000 1.70M → 26.7M (15.8×)**. On the
  geospatial climatology (`GROUP BY lat,lon,hour`, 535k groups) the same fast path took nautilus **1.70s →
  0.067s (25×)** — from 23× *slower* than xarray-sql to **parity (0.067s vs 0.069s)**. Anomaly (agg
  half) 2.32s → 0.76s (3×). Stacks on the 2026-07-17 nested-store fold.
- **Correctness:** structural digest identical before → after at both scales (`cd4180929b4a` keys=1000,
  `3bf27b23…` keys=500000); keyed/count/state `pytest` green; `bench-check` adds no digest failure (the
  lone `bench-join-dist` OUTPUT-CHANGED is pre-existing on clean HEAD).

## 2026-07-17 — Keyed-state nested store: no per-fold StateScope alloc (2.2× on keyed aggregation)

- **Commit:** `79b8125`.
- **Change:** `InMemoryStateBackend` (`state/__init__.py`) kept a flat `dict[StateScope, value]`, so
  `reduce_all` — the hot path every keyed aggregation folds each batch through — built and hashed a
  four-field frozen `StateScope` per `(key, value)` fold. It now nests the store as
  `dict[(operator_id, name, namespace), {key: value}]`, so a fold is one inner-dict update keyed by the
  bare partition key: no `StateScope` built or hashed per fold, and `entries()` (the end-of-stream flush)
  iterates only the matching `(operator, name)` group instead of scanning the whole store. Benefits every
  keyed aggregation — the built-in `KeyedCount`, and the geospatial-benchmark `KeyedMean`.
- **Impact (`nautilus.bench.measure("bench-keyed", rows=1M, batch=4096)`, median of 7 trials, warmup 1;
  Linux x86_64 / WSL2):** the fold cost falls at every cardinality, since the removed `StateScope` is per
  *fold*, not per key — at **keys=1000 4.29M → 9.28M rows/s (2.16×)** and **keys=500,000 837k → 1.49M
  (1.78×)**. Per operator at 500k keys, `KeyedCount`'s `operator.process_micros` 1.17s → 0.56s (2.1×) and
  `operator.on_eos_micros` 0.65s → 0.47s (1.4×). Transfers to the geospatial climatology
  (`GROUP BY lat,lon,hour`, 535k groups): nautilus 2.88s → 1.86s (1.55×).
- **Correctness:** structural digest identical before → after at both scales (`cd4180929b4a` at keys=1000,
  `3bf27b23…` at keys=500,000); full `pytest` green (391 passed); `bench-check` adds no digest failure
  (the lone `bench-join-dist` OUTPUT-CHANGED reproduces on clean HEAD — pre-existing distributed
  nondeterminism vs the off-machine baseline, unrelated to this change).

## 2026-07-04 — HashJoin nested key intern: up to 1.5× on high-cardinality joins

- **Commit:** `d9459f3`.
- **Change:** `HashJoin._encode` (`operators.py`) interned each distinct single-column key by building a
  `((type, value),)` tuple and looking it up in one dict — a tuple allocation per distinct key per batch,
  the residual per-key cost left after the earlier vectorizations. It now interns through a nested
  value-type → value → id map (`_intern_single`), so the common single-key join builds no per-key tuple;
  composite keys keep the tuple form (`_intern_multi`). Both draw ids from one dense counter, so the ids
  stay 0..n-1 for the vectorized probe.
- **Impact (`nautilus bench bench-join --rows 2000000`, median of 5 trials; Linux x86_64, the baseline
  machine):** the win scales with the distinct-key count, since the removed tuple build is per distinct
  key — at **100k keys 2.17M → 2.58M rows/s (1.19×)** and at **1M keys 1.37M → 2.10M (1.53×)**. At the
  low-cardinality baseline (500–1000 keys) it is ~+5% (within noise, so `bench-join` reads unchanged), and
  every existing baseline entry is unchanged. A new `bench-join-wide` entry (100k keys) is the committed
  guard — reverting the change reads REGRESSED −15%.
- **Correctness:** the id is an internal label — equal keys still intern to equal ids and distinct keys to
  distinct ids, and the value+type distinction the keyed shuffle draws is preserved (`int` 1 and `bool`
  `True` stay separate; `int32` 1 and `int64` 1 share an id) — so the join output multiset and the
  structural digest are identical (`d6f80bc9b5` at 200k). The 402-test suite passes.

## 2026-07-04 — Keyed shuffle: single-pass partition, sender cost stops scaling with width

- **Commit:** `c337c7a`.
- **Change:** `_route_keyed` (`runtime/partition.py`) — the sender-side split every keyed shuffle runs —
  filtered the batch once per downstream instance (`pc.equal` + `filter`), rescanning the whole batch
  `num_downstream` times. It now groups the rows by owning instance in a single reorder: numpy
  `flatnonzero` collects each instance's row indices (input order preserved), one `take` lays the batch
  out in instance order, and each instance receives a zero-copy slice. The per-instance rescan is gone,
  so the cost no longer grows with the downstream width.
- **Impact (`nautilus bench bench-keyed --parallelism 16 --rows 2000000`, median of 5 trials; Linux
  x86_64, the baseline machine):** a wide, 16-way keyed shuffle **914k → 988k rows/s, 1.08×**. The gain
  scales with the shuffle width (the removed rescans are `O(width)`): at parallelism 4–8 it is within
  noise, and every existing baseline entry is unchanged (`bench-check`: no regressions). This is a
  scale-out win, not a common-case one — the single-instance source (nautilus fans one source out to
  every instance) is still the ceiling for a parallel keyed pipeline; this only trims the route cost on
  top of it. A new `bench-keyed-wide` baseline entry (parallelism 16) is the committed guard — reverting
  the change reads REGRESSED −7.1%.
- **Correctness:** a pure routing change — each instance keeps exactly its rows, in input order
  (`np.flatnonzero` preserves order), so the per-key co-location the downstream keyed operators rely on is
  unchanged and the structural digest is identical (`9a9dbf867d` at parallelism 4). The 402-test suite
  passes.

## 2026-07-04 — KeyedCount bulk state fold: 1.47× keyed, 1.36× skew

- **Commit:** `4b08176`.
- **Change:** `KeyedCount.process` (`operators.py`) folded a batch by looping over every distinct key,
  calling `reducing_state(KeyContext((v,)), _add).add(count)` — building a `KeyContext` and a
  `ReducingState` handle, and hashing a `StateScope` three times (get + membership + set), per distinct
  key per batch. It now folds the batch's `pc.value_counts` in one call to a new bulk primitive,
  `OperatorContext.reduce_all` → `StateBackend.reduce_all` (the default routes through `get`/`put`;
  `InMemoryStateBackend` overrides it with an inlined get + reducer + set), so no `KeyContext` or handle
  is built per key. On the keyed-shuffle workloads KeyedCount is the gate, and that per-key churn
  dominated it. The primitive is reusable by every keyed aggregation.
- **Impact (`nautilus bench-check`, baseline scale — 200k rows, batch 4096, 500 keys, median of 5 trials;
  Linux x86_64, the baseline machine):** `bench-keyed` **2.11M → 3.10M rows/s, 1.47×**; `bench-skew`
  **2.31M → 3.14M, 1.36×**. Every other baseline entry is unchanged — the fold touches only the keyed
  aggregation. Isolated before/after at the harder 1000-key scale (2M rows) is larger, where more distinct
  keys mean more per-key churn removed: `bench-keyed` 1.11M → 1.69M (1.53×), `bench-skew` 1.80M → 2.53M
  (1.41×), `bench-keyed` at parallelism 4 846k → 1.16M (1.37×). The baseline was re-measured on this
  change so `bench-check` gates the new normal.
- **Correctness:** `reduce_all` matches `ReducingState.add` semantics exactly (a `None` current is a first
  write), so the stored state, `sizes()` (state.entries/keys), and the structural digest are identical
  (`8a736e5dde` at 2M) and the 401-test suite passes. A unit test,
  `test_reduce_all_matches_per_key_fold_and_tracks_sizes`, pins that the bulk fold lands the same state
  and size counts as the per-key loop it replaced.

## 2026-07-04 — Run on Python 3.14 (GIL): faster interpreter, +8–37% across the suite

- **Commit:** `a688a10`.
- **Change:** the project now runs on CPython 3.14 (GIL build), up from 3.12 — `requires-python`, the CI
  and compose workflows, the `Dockerfile`, the README, `uv.lock`, and a new `.python-version`. No engine
  code changed; the speedup is the 3.14 interpreter itself. The bump's incidental code changes are
  behavior-preserving: PEP 695 type parameters for `ValueState`/`ReducingState` (`state/__init__.py`, ruff
  UP046) and an `isinstance` narrowing in `tensors._stack` (3.14 lets mypy type-check numpy, so the
  stub-parse workaround was dropped, which flagged the old `hasattr` guard). black targets py313 so it keeps
  parenthesized `except` tuples.
- **Impact (`nautilus bench-check` at the committed baseline scales, median of 5–9 trials; Linux x86_64,
  the baseline machine; 3.12.3 → 3.14.6):** the keyed shuffle+state paths gain most — `bench-keyed`
  **1.54M → 2.11M rows/s, 1.37×**, `bench-skew` **1.78M → 2.31M, 1.30×** — because that path spends the
  most wall in Python orchestrating the per-key route and state, so a faster interpreter helps it most. The
  rest: async transforms 1.10–1.13×, fanout and the cross-worker chain 1.08×, linear/join/backpressure
  1.03–1.06× (within noise but positive). Nothing regressed. The baseline was re-measured on 3.14 in this
  change, so `bench-check` now gates against the 3.14 normal.
- **Free-threaded 3.14t was evaluated and rejected:** on the same workloads no-GIL *regressed* throughput
  ~11–13% (`bench-keyed` 1.12M → 974k rows/s at 2M rows) with no offsetting gain. nautilus parallelizes
  across processes and runs one cooperative event-loop thread per process, so there is no multi-threaded
  Python bytecode for no-GIL to accelerate — confirmed directly (four instances on one event loop are
  slower than one; two worker processes are faster than one) and by a pure-Python-thread ceiling test
  (no-GIL scales 7.45× at eight threads, which nautilus never exercises). gilknocker also ships no
  free-threaded wheel, so the FULL-tier `runtime.gil_percent` gauge would be absent there anyway.
- **Correctness:** every `bench-check` structural digest is identical across 3.12 / 3.14 / 3.14t (a pure
  speed change alters no output), and the hermetic suite (401 tests) passes on 3.14.

## 2026-07-03 — Forward equal-width keyless edges: data locality instead of always shuffling

- **Commit:** `e41fae6`.
- **Change:** `_spec_for` (`src/nautilus/compile/lower.py`) now selects an edge's partitioner from *both*
  stages' widths, not only the downstream's. A keyless hop between two stages of the same width takes a
  `ForwardSpec` — sender `i` to instance `i` — instead of a `RoundRobinSpec`; keyed edges (the key-group
  shuffle) and the single source's fan-out to every instance (round-robin) are unchanged. `Forward`
  (`src/nautilus/runtime/partition.py`) gained a sender index and routes `i → i` (collapsing to instance 0
  for a single owner); `execute` threads each output's subtask index in. With same-index placement the
  forwarded edge is a free in-process channel, so across workers it moves no bytes and does no Arrow-IPC
  encode/decode. This is the narrow-vs-shuffle split Spark and Flink default to (`DESIGN.md` mechanism 9);
  it resolves the "no co-located forward edge" open item above.
- **Measured with** a new committed benchmark, `bench-chain` (two keyless stages `source → map → map`,
  the shape `bench-linear`'s single stage cannot make; 256-byte payload so the inter-stage edge moves real
  bytes). Its `bench-chain-dist` baseline entry (2 workers, parallelism 4) exercises the forward edge in CI.
- **Impact (`nautilus bench bench-chain --workers 2 --parallelism 4 --rows 200000`, median of 7 trials;
  Linux x86_64 · Python 3.12.3):** round-robin (the change reverted) **659k → 704k rows/s forward, 1.07×**.
  The gain is modest because `--workers` here is loopback TCP, memory-fast, so eliminating the shuffle
  mostly saves the Arrow-IPC encode/decode — a thin slice of a passthrough pipeline; the win scales with
  the *cost* of the removed bytes, and on a real network the shuffled volume is the bottleneck (the NDVI
  fusion entry below removed the same class of shuffle, 2537 MB → 0, for 1.41× on a live-S3 run). What the
  forward edge removes is unambiguous: on a 2-worker A/B, the middle edge's data crossing a socket goes
  from ~half the stream (66 KB on the guard test) to **zero** — only EOS control frames remain.
- **Correctness:** a routing change, so the check is the **output multiset**, not the digest — and here
  the digest is in fact *identical* under both routings (`0bc7cff84d2b`), because a same-width keyless edge
  moves whole batches and each instance ends with the same row count either way. That is exactly why the
  digest cannot guard this and `bench-check` cannot catch a revert; the guard is a transport assertion,
  `test_equal_width_keyless_edge_co_locates_and_crosses_no_socket` (`tests/test_cluster_deploy.py`), which
  deploys across two workers and fails if any data batch crosses the forward edge's sockets (verified: it
  fails, 66 KB > threshold, when the edge is forced back to round-robin). Row conservation and termination
  are pinned in-process by `test_forward_edge_conserves_rows_across_equal_width_keyless_stages`. Every
  other committed baseline digest and throughput is unchanged.

---

## 2026-07-03 — Sentinel-2 NDVI: fuse the reduction into decode so raw pixels never shuffle

- **Commit:** `8c41627`.
- **Change:** the example (`examples/sentinel2_ndvi.py`) split scene decode (`AsyncOpenAndDecode`) from a
  separate keyless `TileNdvi` that computed NDVI. Between two keyless operators the compiler picks a
  `RoundRobinSpec`, so across workers that edge is a network shuffle — raw uint16 pixel tensors crossed the
  wire only to be summed to two numbers per tile on the far side. Fused the two into one `AsyncNdviTiles`
  whose `fetch` reduces each tile-row to its `(ndvi_sum, valid_count)` partials *as it reads it*, freeing
  the row's pixels before the next loads; `integrate` emits only the accumulated partials. Peak memory per
  scene drops from the whole decoded scene (~0.5 GB) to one tile-row (~tens of MB), and only the tiny
  partials reach the keyed `MeanNdviByItem`. NDVI math is unchanged (extracted verbatim to `_ndvi_partials`).
- **Impact (6 and 12 real `sentinel-2-l2a` scenes; Linux x86_64 · Python 3.12.3; median of 3 warm trials —
  a live-S3 workload, so wider variance than the synthetic benches, and not in `bench-check`):** the metric
  is `transport.bytes_sent`, **2537 MB → 0** at 6 scenes / 6 workers. Wall **9.62s → 6.82s (1.41×)** at 6
  scenes / 6 workers; **16.18s → 10.78s (1.50×)** at 12 scenes / 6 workers (that run shuffled 5.1 GB before,
  0.1 MB after). Six-way scaling over the single-worker baseline went from 1.06× to 2.0×. The default
  one-scene single-process run is unchanged (5.37s → 5.82s, within noise).
- **Trade-off (not a regression of the target):** many scenes on *one* worker lose the I/O/CPU overlap that
  two separate operators gave — 6 scenes on a single worker went 10.24s → 13.70s — because a fused
  operator's decode I/O and NDVI CPU no longer run as concurrent stages. This is the config `--workers`
  exists to avoid; the reduction is not in the committed baseline, so `bench-check` does not gate it.
- **Correctness:** topology changed (one fewer operator, and the intermediate edge now carries partials not
  pixels), so the structural digest legitimately differs and is not the anchor here. Instead the **per-item
  mean-NDVI output multiset is byte-identical** before vs after, on both a single worker and six (and
  cross-checked the single-worker before-run against the six-worker after-run).
  `tests/test_examples_sentinel2.py` green.

---

## 2026-07-01 — Unordered async-transform emission (completion order) for stateless maps

- **Commit:** `a943670` (unordered drain), reconciled onto the watermark-free loop in the PR #5 merge.
- **Change:** `run_async_transform` (`src/nautilus/runtime/actor.py`) gained a completion-order drain
  (`ordered=False`, stateless-only): `_drain_unordered` emits any finished fetch — found by scanning the
  reorder buffer in completion order (`_first_ready_index`) — instead of strictly at the deque head, so a
  slow fetch no longer pins buffer slots that finished tails could reuse. With watermarks removed the only
  barrier is terminal EOS, so the scan spans the whole buffer (there is no mid-stream marker). The ordered
  default is untouched — the two drains share an extracted `_emit_data` body. Exposed as
  `AsyncMapBatch(ordered=)` / `.map_async(ordered=)`; `bench-async-io` reads `NAUTILUS_BENCH_ORDERED`, and
  `async_io_wait` grows an opt-in `NAUTILUS_BENCH_SLOW_EVERY`/`_FACTOR` latency skew so a benchmark can
  create head-of-line blocking.
- **Impact (`nautilus bench bench-async-io`, median of 7 trials; Linux x86_64 · Python 3.12.3):** the
  driver of the win is latency *skew* — occasional slow fetches with finished ones queued behind them — not
  the window size. With one batch in 40 running 15× slower (2 ms base fetch, 400k rows), unordered beats
  ordered at every `max_in_flight`: **159k → 503k rows/s (+217%)** at 4, **633k → 1.81M (+186%)** at 16,
  **1.96M → 4.93M (+151%)** at 64. The absolute gap widens with the window — both throughputs scale with it
  — while the relative win stays ~1.5–2.2×. Under *uniform* latency there is nothing to unblock and the two
  are within noise at any window, so this is an opt-in throughput knob for order-insensitive stages under
  skewed I/O latency, not a free default.
- **Regression:** the ordered path — the default, and the only path a keyed stage may use — is unchanged by
  the `_emit_data` extraction: `nautilus bench-check` reports all ten pipelines within noise of the
  baseline (`bench-async` −0.4%, `bench-async-io` −1.5%), every structural digest unchanged.
- **Correctness:** structural digest **byte-identical** ordered vs unordered (`d709cc94b8` at the skew
  config) — a stateless map's rows/batches/EOS counts are order-invariant, which is *why* unordered is
  sound and stays out of the digest; a keyed stage is rejected up front (DSL build-time, with an actor
  backstop for a hand-built IR). Full async-transform suite green (completion-order, in-flight-peak,
  digest-equals-ordered, keyed build/IR rejection).

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
