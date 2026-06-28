# Nautilus — code review (end of Stage 2)

A full-repo review taken before Stage 3, covering the engine core, control plane, transport, telemetry,
compiler, public API, docs, and tests. Dated 2026-06-27, commit `c05451b` on branch `vibes`.

## Verdict

**Overall: 7.5 / 10.** Nautilus is unusually well-engineered for a 0.0.1: the hard parts of a streaming
dataflow engine — watermark combination, EOS termination, credit backpressure, the keyed shuffle, the
two-phase bootstrap, and a zero-cost-when-off telemetry layer — are present, correct in the common case,
and carefully documented. The architecture's load-bearing seams are not just claimed but *mechanically
enforced* (four import-linter contracts hold). What holds the score below 8 is a small number of real
problems: one latent event-time correctness bug, a public API surface that has sprawled, a single-process
engine that exists in two duplicated copies, and a scatter of dead/reserved code. None are structural;
all are fixable before Stage 3 builds on top.

### Ground truth (what actually runs)

| Check | Result |
|---|---|
| `pytest` | **243 passed** |
| `mypy --strict` (58 files) | **clean** |
| `ruff check` | **clean** |
| `import-linter` (4 architecture contracts) | **all 4 KEPT** |
| `black --check` | **fails — 14 files would be reformatted** |

The first four are a strong engineering signal. The last is a real gap: `ruff` is configured to ignore
`E501` *specifically* because "black owns line length" (`pyproject.toml`), yet black is not actually
applied, so 14 source/test files are unformatted. Either run black in CI or drop the deferral.

### How this review was done

Two independent passes, cross-checked against each other: a direct read of ~30 of the most central
source files plus the full doc set and the verification suite above; and a multi-agent sweep (ten
subsystem readers + six cross-cutting reviewers → dedup → adversarial verification of every finding
against the source). 138 raw findings collapsed to 115; 104 were verified against the code, 4 rejected
as false positives, 7 left unverified. The single high-severity bug was reproduced by hand.

## What is genuinely strong

- **The distribution seams.** The `Connector` (one `ChannelId`-keyed interface — `outbound`/`inbound`/
  `finish`/`close`) lets the *same* `execute()` wiring run a plan in-process, across sockets, or mixed
  (`HybridConnector`) with no change to the executor. `finish` vs `close` cleanly separates graceful
  drain from abortive teardown.
- **Plan neutrality, enforced.** `PhysicalPlan` carries operator factories and stateless partitioner
  *specs* (no live `RoundRobin` cursor) and no telemetry topology, so it cloudpickles to a worker that
  never compiled it. The import-linter contracts make the firewalls (`runtime`/`core`/`transport` ⊄
  `cluster`; `compile` imports only `api`; per-record code ⊄ the report layer) impossible to violate by
  accident.
- **Telemetry as facts, not verdicts.** Hot-path instrumentation only records raw numbers; the report is
  assembled at the boundary. Disabled telemetry resolves to shared no-op instruments and costs nothing.
  The `structural_digest` (reproducible facts only — no timing, no run id) is what makes the benchmark
  harness and the two engine paths comparable.
- **The credit transport.** `SocketChannel` spends a credit and writes the frame under one lock with no
  `await` between, so a cancelled `send` cannot lose a credit or split a frame; control frames are
  credit-exempt so a full data window never delays a watermark.
- **The docs.** The per-altitude standard is real and largely met — the "why" lives in `DESIGN.md`, the
  mechanism in docstrings, the line-level reason in inline comments. This is rare.

## Findings that matter

Locations are `file:line`. IDs (`C#`) trace to the verification record.

### Correctness

- **[HIGH] Event-time unit bug — `ColumnTimestampAssigner` does not normalize the timestamp unit**
  (`core/time.py:85-89`, C1). For an Arrow `timestamp` column it does `pc.cast(col, pa.int64())`, whose
  inline comment claims "units are already micros-or-less ints." That is wrong: casting a timestamp to
  int64 returns the raw value in the column's *own* unit with no conversion. Reproduced with the repo's
  pyarrow: a `timestamp('ms')` value of `1000` (1 second) yields `1000` micros instead of `1_000_000`;
  `'s'` and `'ns'` are wrong too. Only `'us'` columns work. Tests pass only because the built-ins feed
  int64-micros columns — but `ColumnTimestampAssigner`'s docstring explicitly advertises "an int64
  micros column, *or an Arrow timestamp*," so this is a latent wrong-result on a documented input.
  Fix: `pc.cast(col, pa.timestamp("us"))` before the int64 cast, or reject non-`us` timestamps loudly.
- **[MED] `graph_from_stages` silently round-robins a keyed operator** (`runtime/parallel.py:73`, C93).
  It takes key columns only from `Stage.key_columns` (default `None`) and ignores the built operator's
  own `key_columns()`, so `Stage(lambda: KeyedCount("word"), 2)` *without* restating `["word"]` compiles
  to a keyless round-robin edge that splits each key across instances — a silently wrong aggregation.
  Its sibling `graph_from_pipeline` reads `op.key_columns()`. Make keying have one source of truth.
- **[MED] Fan-in always picks the lowest-index ready channel** (`runtime/mailbox.py:64`, C10). A
  continuously-ready input 0 can starve a higher-index input, stalling the *combined* watermark (it is
  the min over inputs) and deferring fail-fast on a later input. Per-channel FIFO is preserved by any
  pick, so rotate the scan start (`(last+1) % n`) for fairness.
- **[LOW, worth noting] `recv()` on a terminated `SocketChannel` raises once, then hangs**
  (`transport/socket_channel.py:100-108`), and **`execute()` skips `connector.close()` on a Phase-A/B
  wiring failure** (`runtime/execute.py:249-267`, C41) even though `worker_main`'s teardown comment
  assumes it always ran. Plus small ones: an empty `key_columns` tuple is accepted and downgraded to
  round-robin (`api/graph.py`, C32); `nautilus examples` raises `IndexError` if any builder lacks a
  docstring (`cli.py:211`, C84); the bench harness can't reproduce `bench-skew`/`bench-backpressure`
  because `NAUTILUS_BENCH_SKEW`/`DELAY_US` aren't part of the recorded scale (`bench.py`, C81).

### Architecture & duplication

- **[MED] Two single-process engines kept in lockstep by hand** (`runtime/local.py:117-235` vs
  `runtime/run.py` + `runtime/execute.py`, C18). `run_local_chain` (the original hand-wired mesh) and the
  `compile_graph → execute` path do the same job, duplicating the sink-collect loop, the sampler/
  TaskGroup/report block (verbatim), and the topology builder. The legacy one is still the default: it
  backs the public `run()`, the CLI single-process path, the live dashboard, `bench`, and `testing`.
  `IMPLEMENTATION_PLAN.md:31` claims Stage 2 "subsumed" it, and `local.py:7` says "Stage 2 replaces
  this" (C23) — neither is true. Mitigation: digest-parity tests compare the two engines, so they can't
  silently diverge. Finish the subsumption: make `run_local_chain` a thin shim over `run_plan` via the
  existing `graph_from_ops` bridge, exactly as `run_parallel_chain` already is.
- **[MED→LOW] Dead / reserved code.** Verified unused: `RunResult.schema` and `OperatorContext.config`
  (never populated or read — and the empty-typed-table branch is therefore unreachable, C22); the
  `Broadcast` partitioner (never selected or tested, C24); `ListState`/`MapState` (unreachable, and their
  in-place mutation actually breaks the `StateBackend` get/put contract, C3); `HashPartitioner` (the
  runtime only ever builds `KeyGroupPartitioner` — `HashPartitioner` survives only in tests);
  `graph_from_ops` (tests only, C101); `PhysicalPlan.operator()`, `TimeWindow.max_timestamp()`,
  `Gauge.add`, `framing.split()`, `MetricSpec.reduction` (documented but never read, C55). Either wire
  these into the path they were reserved for, or delete them so the surface reflects what's live.

### End-user API (the headline ask)

The CLI, the telemetry, and `RunResult` are good; the *library* surface is where the friction is.

- **[MED] The curated public surface can't complete its own example** (`__init__.py`, C74). The module
  docstring shows `run(from_batches(...), [Tokenize, KeyedCount])`, but no exported name builds a
  `Batch`/`Watermark`/`EOS_FRAME` — every example reaches into `nautilus.testing` (a module documented as
  *test* helpers) or `nautilus.core.records`. Re-export the frame builders, or let `from_batches` accept a
  raw `pa.RecordBatch`.
- **[MED] `from_batches` is a footgun by name** (`operators.py:44`, C91). It reads like Arrow's
  `Table.from_batches` but takes nautilus `Frame` wrappers; pass a raw `pa.RecordBatch` and it is
  *silently dropped* — `run_transform`'s frame dispatch has no `else`, so an unknown frame vanishes with
  no error. Add a terminal `else: raise` to the dispatch, and validate inputs at the entry point.
- **[MED] The surface has sprawled** (C92): seven run-like entry points (`run`, `run_local_chain`,
  `run_plan`, `run_compiled`, `run_parallel_chain`, `deploy`, `run_ops`), four graph builders, and two
  near-duplicate vertex types (`Stage` vs `LogicalVertex`, where `Stage` is strictly less capable). The
  six examples use three builders and four runners for essentially one job. Scaling from
  `run(source, transforms)` to parallel means changing the input shape *and* the module *and* the
  sync/async convention. Trim: delete `graph_from_ops`, retire `run_parallel_chain`, fold `Stage` into
  `LogicalVertex`.
- **[LOW] Operator authors must drop to the raw backend to enumerate/clear state.** `OperatorContext`
  exposes `value_state`/`reducing_state` but no way to iterate or bulk-clear keys, so both built-in keyed
  operators reach into `ctx.state_backend.entries(...)` + hand-built `StateScope(...)`, re-specifying the
  operator id and state name the context already holds. That's the *expected* pattern, not an escape
  hatch — promote it into the typed-handle API.

### Telemetry rough edges

- **[MED] Distributed runs drop most of `TelemetryConfig`** (`cluster/coordinator.py:78` →
  `worker_main.py:142`, C38): only `tier` and `sample_system` reach workers; `clock`,
  `sample_interval_micros`, `event_log_capacity`, `validate`, `run_id` revert to defaults. Cloudpickle the
  whole config alongside the plan.
- **[MED] The summary can't disambiguate subtasks in a parallel run** (`report/report.py`, C64):
  `OperatorSummary` keeps only `operator_id`, so a P=3 run shows three indistinguishable `op0` rows and
  `operator()` silently returns the first. Add `subtask_index`/`node`.
- **[MED] The markdown "token budget" doesn't bound output** (`report/serialize.py:278-294`, C66): the
  by-send-wait tail is appended after the budget check, so the digest can exceed the budget. Budget the
  tail too.
- **[MED] An operator metric named like an engine metric double-counts** (C56): the actor recorder and
  `ctx.metrics` share the grouping key and the report sums by catalog name, so `ctx.metrics.incr(
  "operator.rows_out", ...)` would inflate totals. Add an owner field to `MetricSpec` and reject
  engine-owned keys on the author recorder.
- **[MED] The live dashboard's "frozen final snapshot" is not frozen** (`telemetry/live.py:186-196`,
  C67): `wall_micros` keeps growing during linger, so the displayed throughput keeps dropping after the
  run ends. Capture the meta once when status flips to "completed."

### Documentation

The standard is high and mostly met, but a few claims contradict the code: README says Arrow is
"zero-copy across processes" while the socket path serializes and copies every batch
(`transport/framing.py:83-87`, C104 — true only in-process); `docs/cli-reference.md` claims to mirror
`cli.py` but omits the shipped `bench`/`bench-check` commands and has wrong exit codes (C96); and
`local.py:7`'s "Stage 2 replaces this" is stale (C23). Most of the 23 verified doc findings are
low-severity funnel/precision nits. (`IMPLEMENTATION_PLAN.md` promises "property/fuzz tests"; the suite
has randomized/oracle tests but no property-based library — a defensible loose use of the word, but worth
aligning.)

## Score by category

| Category | Score | Why |
|---|---:|---|
| Architecture & abstractions | 8 / 10 | Excellent seams, enforced firewalls; docked for the dual engine and a few leaks. |
| Code cleanliness | 7.5 / 10 | Clean, strict-typed, lint-clean; docked for dead/reserved code, black drift, two engines. |
| Correctness & robustness | 7 / 10 | Core semantics solid and tested; one latent event-time bug, a keyed-shuffle footgun, fan-in fairness, teardown edges. |
| API / developer experience | 6 / 10 | Great CLI + telemetry + `RunResult`; overlapping, incompletely-exported library surface; awkward scale-up path. |
| Documentation | 8 / 10 | Genuinely good standard, largely met; a handful of stale/contradictory claims. |
| Testing | 8 / 10 | 243 tests, strict types, digest-parity across engines; gaps in parallel-subtask telemetry, the timestamp path, and dead handles. |
| Scalability | 7 / 10 | Strong backpressure/credit/key-group design; per-row Python in shuffle & `Tokenize`, naive round-robin placement, O(n) bits, single-instance sink, unbounded in-memory state (documented). |

## Suggested priorities before Stage 3

1. **Fix the event-time unit bug (C1)** and add a non-`us` timestamp test. It is the one issue that
   produces silently wrong results on a supported input.
2. **Close the two API footguns** — `graph_from_stages` keyless-shuffle (C93) and the silently-dropped
   raw `RecordBatch` (C91) — both are silent wrong-result / lost-data traps.
3. **Unify the single-process engine (C18)**: shim `run_local_chain` over `run_plan`. This deletes the
   largest duplication and makes the default path exercise the compiler everything else uses.
4. **Tidy the public surface (C74, C92)** before Stage 3's DSL hardens it: re-export frame builders, trim
   the redundant runners/builders/vertex types, and decide `RunResult.schema`/`OperatorContext.config`
   (wire or delete).
5. **Enforce black in CI** (or drop the ruff deferral), so formatting can't drift.

## Resolution (implemented 2026-06-28)

All five priorities above and the great majority of the 104 verified findings were implemented across
seven commits (correctness → dead code → engine unification → API → telemetry/transport → docs →
tests/format). Highlights: the event-time unit bug (C1) is fixed and regression-tested; both silent
traps (C91, C93) fail loudly; `run_local_chain` is now a thin shim over the compiled `run_plan` path, so
there is one execution engine (C18); the public surface builds its own one-liner and `run()` takes
`parallelism=` (C74, C94); dead code is gone (C24, C3, C22, C35, C6, C2, …); telemetry/transport edges
(C56, C64, C66, C67, C38, C13, C60, …) and ~25 doc findings are addressed; the generated
telemetry-reference is regenerated from the catalog.

Post-change ground truth: **pytest 267 passed**, mypy --strict clean, ruff clean, **black --check clean
(105 files)**, import-linter 4/4. All eight examples and the multi-process `deploy` examples run; the
distributed result matches single-process.

Deliberately deferred (low value vs. churn/risk; noted at the call sites):

- **C46 / C52 code** — credit-return coalescing and control-frame coalescing on the socket channel: a
  protocol change on delicate, well-tested flow-control code for a low-magnitude win. The control-frame
  unboundedness tradeoff is now documented instead.
- **C54** — metric label-key validation: would touch every hot-path recorder call; the redundancy it
  targets is cosmetic.
- **C36** — a shared kind StrEnum: the IR firewall (api imports nothing internal) resists a single
  cross-layer enum, and the kind strings are a small, consistent, localized set.
- **C113** — deduping copy-pasted test helpers across ~6 files: test-only churn.
