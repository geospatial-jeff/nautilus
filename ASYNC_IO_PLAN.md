# Nautilus — Async I/O in operators and sinks (working plan)

A design + staged implementation plan for letting **intermediate operators and sinks do efficient
async I/O**, not only the source. This is a working plan; when the work lands it folds into
`DESIGN.md` (a new mechanism 8, plus amendments to mechanisms 3 and 4) and `IMPLEMENTATION_PLAN.md`
(Stage 6). The design below was generated and then adversarially verified against the code; the four
defects that verification found are folded into the loop contract here, not left open.

> **Superseded (watermarks removed).** After this plan landed, event-time watermarks and windowing were
> removed from nautilus entirely. The async loop no longer carries watermark markers or a `WATERMARK_MAX`
> terminal flush, and the flush hook is `on_eos` (not `on_watermark`): the ordered reorder buffer now holds
> only DATA slots and forwards `EOS` after draining them, calling `on_eos` for the terminal flush. The
> watermark/marker language in the design below is the original plan; the current async design is
> `DESIGN.md` mechanism 8 and the `run_async_transform` / `run_async_sink` docstrings.
>
> **Status.** The **async sink** (sink scope of 6.0–6.2) and the **async transform** (6.3 — the
> fetch/integrate split: `AsyncOneInputOperator`, `AsyncMapBatch`, `run_async_transform`'s reorder
> loop, the enforced `OperatorContext` state guard, the DSL `.map_async`/`.apply_async`, the
> `async_one_input` IR kind, the cross-process path) have landed — stateless *and* keyed; `DESIGN.md`
> mechanism 8 records them. **6.4 has landed**: (a) **unordered mode** — a stateless map may emit in
> completion order (`ordered=False`) via `_drain_unordered` in `run_async_transform`, with the DSL/actor
> rejection for keyed stages and the `AsyncMapBatch(ordered=)` knob; with watermarks removed only the
> terminal `EOS` is a barrier, so a finished fetch never waits behind a slow one (the head-of-line win,
> and the unordered-rejected-for-keyed rule below is now live); and (b) the **NDVI example rework** —
> `examples/sentinel2_ndvi.py` is now a `Stream` graph whose `AsyncOpenAndDecode` async transform does the
> COG open + range-read + decode the source used to, with an opt-in `NdviSink` (`--write`); it is
> registered as a graph example and dashboarded via the new `serve_graph`. Stage 6 is complete.

## The problem

Today async/await — efficient I/O-bound code — can be written in exactly one place: a
`SourceOperator.frames()` is an `async def` generator, so a source may `await` between batches. Every
transform is synchronous: `OneInputOperator.process`/`on_watermark` and
`TwoInputOperator.process_left`/`process_right` must not `await`; they emit into a `Collector` and the
actor performs the backpressured sends *between* steps. And there is no user-authorable sink at all —
`compile_graph` synthesizes a `CollectSink` and `runtime.execute._collect_sink` drains batches into an
in-memory list, so the only way to write output to an external store is *after* `run()` returns,
outside the streaming model (unbatched, unbackpressured, not parallel).

This forces all I/O to the edge of the graph, and only the *input* edge. Two common dataflow shapes
have nowhere to live:

1. **A sink that writes results to an external store** — write the NDVI means to S3 or a database,
   backpressured and parallel like any stage, instead of collecting the whole result in memory and
   writing it in the driver.
2. **An intermediate operator that does external I/O** — enrich a record from an async lookup, or read
   and write an external store to track per-key status — with the I/O of many in-flight batches
   *overlapping*, not serialized one batch at a time.

The motivating example already shows the distortion. `examples/sentinel2_ndvi.py` crams COG range-read
and decode into the source, and its own docstring explains why: "an operator's `process` is
synchronous and only sees an Arrow batch, but a range request must `await`". The decode belongs in its
own operator with a declared parallelism; it lives in the source only because the source is the one
place that may `await`.

## The constraint that creates the problem

Operators are synchronous on purpose. `DESIGN.md` mechanism 4: `process`/`on_eos` never `await`;
each per-batch step runs to completion as one critical section, so it cannot interleave with anything
else on the event loop. That is exactly what makes keyed state (`nautilus.state`) **lock-free and
single-writer**: each operator instance is driven by one asyncio task and owns its own state backend,
and `ReducingState.add`'s read→reduce→put is safe without a lock *because* it never spans an `await`.

So the naive fix — "make `process` an `async def`" — is wrong twice over. It gives no I/O overlap (one
batch's I/O still completes before the next batch is processed), and it normalizes awaiting inside the
state-mutating method, so the single-writer guarantee survives only by the accident that concurrency
happens to be 1 and collapses the moment anyone raises it.

## The design: the fetch/integrate split

Split every async stage into an **awaiting half** that is state-free and runs concurrently, and a
**synchronous half** that runs on the actor task and is the only code that touches state or emits. The
engine — not the operator — owns concurrency, ordering, and the watermark/EOS barriers, and the actor
stays the sole emitter, the sole state writer, and the sole telemetry-recorder writer. Concurrency is
confined to value-producing I/O.

For an **intermediate async transform** (`AsyncOneInputOperator`):

- `async def fetch(self, batch) -> object` — the awaiting half. Does the external I/O (range request,
  DB read, HTTP lookup) and *returns* an opaque per-batch result. Runs as a bounded set of concurrent
  `asyncio.Task`s. It is handed only the batch; it must not emit, must not touch nautilus keyed state,
  and must not write telemetry (enforced — see below).
- `def integrate(self, batch, result, ctx, out) -> None` — the synchronous half, run only on the actor
  task, exactly like today's `process`. It may fold `result` into keyed state via `ctx` and emit via
  `out`. It never `await`s, so the keyed-state critical section never spans a yield.

For an **async sink** (`AsyncSink`) there is no synchronous half, because a sink has no downstream:

- `async def write(self, batch) -> None` — the whole job. The external write (S3 PUT, DB insert). Runs
  as bounded concurrent tasks; handed only the batch.

This is strictly more capable than a "stateless-only async context": because `integrate` runs on the
actor task, a *keyed* async operator can keep many `fetch`es in flight (real overlap) while its keyed
state stays single-writer and its integration is serial and in input order (deterministic). Use case 2
— a keyed external-status tracker — is expressible without giving up the lock-free guarantee.

### Why the awaiting half cannot reach state (the corrected barrier)

`fetch`/`write` taking only the batch is necessary but **not sufficient**: they are bound methods, so a
naive `open(self, ctx)` that stashes `self._ctx` would let a concurrently-scheduled `fetch` reach
`ctx.value_state(...)` and read-modify-write keyed state across its own `await` — the exact lost-update
the design must forbid. The barrier is made real two ways, together:

1. **Don't hand the state to the awaiting half.** `integrate` and `on_watermark` receive the
   state-capable `ctx` *as a call argument* (`integrate(batch, result, ctx, out)`,
   `on_watermark(t, ctx, out)`), and the contract is that the operator must not stash it on `self`.
   `open(ctx)` keeps only author-owned I/O resources (a pooled client), not the state handle. This is
   the pit of success.
2. **A runtime guard that turns a violation into a loud failure, not silent corruption.** The actor opens
   the guard only for the synchronous duration of each `open`/`integrate`/`on_watermark`/`close` call and
   closes it around every `fetch`. The guard wraps the *state backend and recorder themselves*, not just
   the `ctx` accessors — so a keyed read/write or metric write from `fetch` raises `StateAccessError`
   whether it arrives through `ctx` or through a handle the operator cached in `integrate` (the handle
   holds the guarded backend). A `fetch` runs only while the actor is awaiting, when the guard is closed,
   so any such touch raises immediately. (A synchronous `integrate` cannot interleave with a `fetch`,
   since it never yields — so the flag is unambiguous.)

The same reasoning forbids `fetch`/`write` from touching `ctx.metrics`/`ctx.io_wait()`: I/O time is
attributed by the engine (below), so the awaiting half has no recorder and the guard covers
`ctx.metrics` too.

## The driver loop

Landed as separate loops `run_async_transform` and `run_async_sink` in `runtime/actor.py` — siblings of
the proven `_run_operator_loop`, not branches inside it, because the in-flight reorder/barrier shape
shares nothing with it. Each reuses the proven pieces verbatim: `WatermarkTracker(n)`, the `_capture`
fail-fast wrapper, `_flush`/`_broadcast`, the `eos.*` bookkeeping, and the `WATERMARK_MAX` terminal
flush. The reorder mechanism itself — an ordered `deque` of DATA and MARKER slots, fetches woken through
one `asyncio.Event` (the loop blocks once per completion, not once per in-flight task), `max_in_flight`
bounding the buffer, per-fetch fail-fast recorded in each fetch's completion callback, and the
every-iteration terminal-drain invariant that keeps `WATERMARK_MAX`/EOS strictly after the last in-flight
batch — is `run_async_transform`'s docstring to specify and own; this plan does not restate it.

**Still planned (6.4): unordered mode.** For a stateless map, emit in *completion* order instead of input
order — drain any done DATA slot in the leading pre-barrier segment, and treat a watermark/EOS marker as a
hard barrier (drain the in-flight segment to zero before firing). Lower latency, but sound only with no
keyed state and no digest-order guarantee, so it stays opt-in and stateless-only; until it lands the loop
rejects `ordered()=False`.

**Per-request timeout.** Each `_timed` task wraps its `fetch`/`write` in `asyncio.wait_for(...,
timeout)` (the timeout is an operator knob, default off). On expiry the task raises, the failure scan
fires, `async.timeouts` is incremented, and the job fails fast. Retry is out of scope for v1
(`DESIGN.md` robustness is at-least-once + fail-fast); a default-value/skip seam is a documented later
option.

## Ordered vs unordered

`ordered()` defaults `True`: integrate and emit strictly by input `seq` (the deque head is the reorder
buffer), so emission, the keyed-state fold order, and the structural-digest counts are deterministic
and reproducible — and a watermark/EOS marker is forwarded only once it reaches the head, after every
earlier batch is emitted, *without* forcing concurrency to 1 (later fetches keep running behind the
marker). `ordered()=False` integrates in completion order for lower latency and treats a marker as a
hard barrier; it is opt-in throughput for order-insensitive stages.

**Unordered is rejected for keyed/stateful stages** (the fix for the determinism defect): `rows_out`
and `batches_out` are structural-digest inputs and are order-dependent when `integrate` emits
conditionally on running keyed state, so an unordered keyed stage would produce a non-reproducible
digest and flake CI. The builders/compiler reject `ordered()=False` when `key_columns()` is non-`None`;
v1 unordered is limited to the stateless `AsyncMapBatch` (one row out per row in, count
order-invariant). The DESIGN/docstring prose states the precise condition, not the blanket "the digest
is order-independent."

## Authoring surface

`core/operator.py` — two new ABCs beside the existing ones:

```python
class AsyncOneInputOperator(ABC):
    def open(self, ctx: OperatorContext) -> None: ...                 # actor task; acquire client/pool (NOT state)
    async def fetch(self, batch) -> object: ...                       # CONCURRENT, state-free: only the batch; may await
    def integrate(self, batch, result, ctx, out: Collector) -> None:  # ACTOR TASK, sync critical section: state + emit
    def on_watermark(self, t, ctx, out: Collector) -> None: ...       # actor task, sync; default no-op
    def key_columns(self) -> tuple[str, ...] | None: ...              # default None
    def max_in_flight(self) -> int: return 8                          # concurrency bound (backpressure)
    def ordered(self) -> bool: return True                            # input-order integrate/emit
    def timeout_micros(self) -> int | None: return None              # per-request deadline; None = off
    def close(self) -> None: ...

class AsyncSink(ABC):
    def open(self, ctx: OperatorContext) -> None: ...
    async def write(self, batch) -> None: ...                         # CONCURRENT, state-free: the external write
    async def on_watermark(self, t) -> None: ...                      # actor task; default no-op; commit-on-event-time seam
    def key_columns(self) -> tuple[str, ...] | None: ...             # keyed dedup/upsert co-partitions
    def max_in_flight(self) -> int: return 8
    def timeout_micros(self) -> int | None: return None
    async def close(self) -> None: ...                               # final flush/commit + client close (the at-least-once point)
```

`operators.py` — `AsyncMapBatch(AsyncOneInputOperator)` wrapping `fn: Callable[[RecordBatch],
Awaitable[RecordBatch]]`: `fetch = await fn(batch)`, `integrate = out.emit(result)`. The stateless
enrich/lookup built-in.

`dsl.py` — `Stream._extend` gains a `kind` parameter (it hard-codes `"one_input"` today):

- `.map_async(fn, *, ordered=True, max_in_flight=8, parallelism=1) -> Stream` (`AsyncMapBatch`).
- `.apply_async(op: AsyncOneInputOperator, *, key_columns=None, parallelism=1) -> Stream` — the escape
  hatch mirroring `.apply` (deep-copied per subtask at parallelism > 1).
- `.sink(sink: AsyncSink, *, key_columns=None, parallelism=1) -> SinkHandle` — a terminal that exposes
  only `.run()`/`.run_async()`/`.collect()`. Chaining a combinator off a sink is a type error, not a
  runtime one.

`key_columns`/`parallelism` ride the edge through the same `_spec_for` path as `.map`/`.count_by`, so a
keyed async lookup co-partitions and a keyless async map round-robins (the I/O fan-out).
`max_in_flight`/`ordered`/`timeout` ride on the built instance (read by the actor), **not** the plan —
so they never enter the structural digest.

`api/graph.py` — add `_ASYNC_ONE_INPUT`/`_ASYNC_SINK` to `_KINDS`, both `_NUM_INPUTS == 1`; builders
`async_one_input(...)` and `async_sink(...)`. Validation: an `async_sink` vertex must be a leaf (no
outbound edge) in both `_validate_dag` and `_validate_linear`; reject `ordered=False` with a non-`None`
`key_columns`. The IR keeps treating a factory as opaque `Callable[[], object]`, so `api` still imports
no operator type (the import-linter contract holds).

`compile/plan.py` — `PhysicalOperator.kind` comment extends to include the two new kinds; no new field.

## The synthesized CollectSink stays, by conditional synthesis

`compile_graph` still finds the single leaf and appends the `CollectSink` (`SINK_ID`, `factory=None`,
the `leaf → sink` `ForwardSpec` edge) — **unless** that leaf's vertex kind is `async_sink`, in which
case the user sink *is* the plan's terminal and nothing is appended. Every existing graph (every test,
example, and benchmark) ends in a transform/join leaf, so it synthesizes the identical `CollectSink`;
`execute._collect_sink`, `RunResult.to_pylist()`, and the dashboard's sink node are byte-for-byte
unchanged. A graph that opts into a user sink correctly returns empty `sink_batches` — the data went to
the external store. `execute.py` keeps its `kind=="sink"` branch and gains a `kind=="async_sink"`
branch (instantiate the user sink via its factory, build a mailbox + an `Owner.AUTHOR` recorder, no
outputs, run `run_async_sink`) and a `kind=="async_one_input"` branch (`run_async_transform`). Both new
kinds are added to the set that gets an `Owner.AUTHOR` metrics recorder (`execute.py` currently lists
only `source`/`one_input`/`two_input`).

## Invariants (each verified against the code)

- **Single-writer keyed state** — `integrate`/`on_watermark`/`open`/`close` run only on the actor task,
  serially, never inside an `await`. The only concurrently-scheduled author code is `fetch`/`write`,
  which is handed no `Collector`, and whose state/metric access is blocked by the runtime guard.
- **Watermark + EOS ordering** — markers are deque slots carrying the advanced combined watermark
  computed in mailbox read order; ordered mode forwards a marker only at the head (after all lower-seq
  data is emitted), unordered drains the leading segment first, and terminal drain is a loop-level
  invariant so `WATERMARK_MAX`/`EOS` follow the last in-flight batch.
- **Backpressure** — `max_in_flight` bounds the in-flight set and the deque (decrement on pop); a full
  pool stalls reads and the bounded channel / credit window stalls upstream. Bounded memory.
- **Determinism + digest** — ordered default reproduces emission and fold order; unordered is rejected
  for keyed stages; `max_in_flight`/`ordered`/`timeout` are instance knobs absent from the plan; new
  graphs only, so every existing linear/join digest is byte-for-byte unchanged.
- **Import-linter** — new code lives in `core`/`runtime`/`operators`/`dsl`/`api`/`compile`/
  `telemetry.catalog`, all already in the data-path allowlists; the async loop imports neither
  `cluster` nor `telemetry.report`.
- **Telemetry attribution** — `_timed` returns the duration and writes nothing; the actor records
  `async.request_micros` on reap, so the recorder stays single-writer. `runtime.step_micros` for an
  async stage is the actor's coordination + `integrate`/`on_watermark` self-time only, never the
  awaited I/O — a cleaner split than the source's conflated `step_micros`. Summed `async.request_micros`
  exceeding wall is the overlap signal, with `async.in_flight` (MAX gauge) the peak concurrency.

## Distributed / cluster path

`execute.py` is the same code single-process and across workers, so the new branches run under
`cluster.deploy` unchanged. The plan must state, not just imply:

- **Placement.** A user `AsyncSink` at parallelism > 1 is N parallel external writers across workers,
  routed by `key_columns` like any stage. The at-least-once / idempotency contract (below) must hold
  with N writers.
- **Serialization.** Both new operator subclasses and their pooled clients must cloudpickle to a bare
  data-path worker. The client is acquired in `open()` per subtask (not `__init__`, which the
  compile-time double-build and the deep-copy-per-subtask path would replicate); a round-trip test
  pins this for both kinds.
- **Snapshots.** `async.*` metrics aggregate at the coordinator like every other metric; a write-only
  worker returns empty `sink_batches`.
- **Egress.** A sink on a worker daemon means external egress + credentials from a worker container.
  Functionally fine on the trusted Stage-4 network; securing it is Stage 5 (called out, not solved).

## At-least-once and idempotency for writes

`DESIGN.md` robustness is at-least-once + fail-fast whole job, with no checkpointing, so any failure
re-runs the whole job and the sink re-writes everything. The `AsyncSink.write` contract, stated in its
docstring and as a `DESIGN.md` amendment, is therefore: **writes must be idempotent under whole-job
re-run** — deterministic keys / upsert / overwrite-by-key, so a replay converges. `key_columns` on the
sink lets a keyed sink co-partition for per-key upsert/dedup. Partial writes committed before a sibling
failure remain after the abort (at-least-once, by definition); an `on_watermark` transactional commit
at event-time `t` commits a prefix that survives a restart. Exactly-once via the reserved `Barrier`
frame stays deferred.

## The NDVI example, reworked

The example is the proof, and it exposes two things the plan must resolve:

- **It cannot stay a linear `(source, transforms)` pipeline.** That shape (`pipelines.Pipeline`,
  `graph_from_pipeline`, `run_local_chain`) builds every transform as `kind="one_input"`; an
  `AsyncOneInputOperator` is not an `OneInputOperator` and a no-edge linear graph has no async kind. So
  the example becomes a `Stream`/`LogicalGraph` graph pipeline.
- **One leaf.** A graph has exactly one leaf, so it cannot both write NDVI to a sink and collect
  `MeanNdviByItem` for printing. Recommended resolution: the **default** example keeps
  `MeanNdviByItem → CollectSink` so `main()` and the `nautilus run` preview still print mean NDVI per
  scene, and demonstrates the **async transform** by moving COG open + range-read + decode out of the
  source into a new `AsyncOpenAndDecode(AsyncOneInputOperator)` (`fetch` does the awaited I/O,
  `integrate` emits the decoded red/nir tensor columns). The **async sink** is shown as an opt-in
  variant (`--write <uri>`) that terminates in an `AsyncSink` writing the means out; that variant
  returns empty `sink_batches`, and `main()`/the CLI present a write-only run as a row-count + telemetry
  summary rather than a table. Tests inject the reader and a fake sink so they run without network/S3.
  The source becomes a pure item-id/href lister, and its docstring loses "because an operator's
  `process` is synchronous".

## CLI / pipelines / dashboard / bench wiring

- **Registry.** `pipelines.EXAMPLES` builds `(source, transforms)`; `GRAPH_EXAMPLES`/`is_graph_pipeline`
  key on "more than one source". A single-source async/sink example fits neither — extend the registry
  so an async/sink graph example is runnable by name.
- **`nautilus run`.** It previews `result[0]`/`len(result)` and prints rows-out; define how a write-only
  (empty `sink_batches`) run is presented (row-count in + telemetry, no table).
- **Dashboard.** `nautilus dashboard` calls `serve_local_chain(source, transforms)` (linear only); add a
  graph path and render the new `async.*` metrics.
- **Bench.** Decide whether the async example is benchable; latency-driven nondeterminism must not reach
  the structural-digest gate (ordered mode keeps the digest stable, which is why keyed stages are forced
  ordered).

## Telemetry catalog additions

New `MetricSpec`s in `telemetry/catalog.py`, then regenerate `docs/telemetry-reference.md` (never
hand-edit the generated file):

- `async.requests` (counter) — fetch/write tasks completed.
- `async.request_micros` (counter, µs) — summed wall time tasks spent awaiting external I/O; exceeds
  wall under overlap.
- `async.in_flight` (gauge, MAX) — peak concurrent fetches/writes.
- `async.capacity` (gauge, LAST) — the configured `max_in_flight`.
- `async.timeouts` (counter) — requests that hit `timeout_micros`.

Meanings are written as facts, not cause-and-effect (the meaning lint), and stay out of
`STRUCTURAL_METRICS` (they are timing-dependent). The agent-prompt metric list surfaces them for free.

## Tests

The property/fuzz tests that would have caught the verified defects, beyond per-stage unit tests:

- EOS arriving with N slow in-flight fetches emits all N batches, **then** the `WATERMARK_MAX` flush,
  **then** EOS — asserting order, not just row count; the loop never `asyncio.wait`s an empty set.
- A mid-stream watermark arriving with a full in-flight window forwards after the data frontier.
- Ordered slow-head: the reorder deque stays ≤ `max_in_flight` (inflight decremented on pop).
- A non-head fetch failure is surfaced promptly with siblings cancelled-and-awaited (prompt release).
- A timeout increments `async.timeouts` and fails the job.
- Oracle: an ordered randomized-latency run equals the sync `MapBatch` baseline byte-for-byte and has a
  stable structural digest across trials; an unordered keyed conditional-emit pipeline is rejected.
- Cross-process sink + transform: row conservation, snapshot aggregation, cloudpickle round-trip for
  both new kinds.
- Digest oracle: every existing transform/join-leaf graph's plan and digest are byte-for-byte unchanged.

## Staging

Each sub-stage is independently shippable and green across pytest / mypy / ruff / black /
import-linter / bench-check, in repo style; docs are written with the code (CLAUDE.md: docs are
first-class). The example/CLI wiring and the at-least-once/timeout contracts are pulled *early* (into
6.2/6.3), because they are load-bearing for shipping a sink at all.

- **6.0 — ABCs + metrics, no loop.** `AsyncOneInputOperator` (fetch/integrate, the runtime-guard
  contract documented), `AsyncSink`, `AsyncMapBatch`; the `async.*` `MetricSpec`s + regenerate the
  reference; add both kinds to `execute.py`'s `Owner.AUTHOR` recorder set; the `OperatorContext`
  state-guard flag. Tests pin the ABC surface and that `fetch`/`write` take only the batch and that a
  guarded state access from outside `integrate` raises. Draft the `DESIGN.md` mechanism-5 amendment.
- **6.1 — the driver loop.** `_run_async_operator_loop` + the two wrappers, with **all four verified
  fixes baked in** (terminal-drain loop invariant; full-wake-set fail-fast with cancel-and-await;
  inflight-on-pop bound; state guard) plus the `wait_for` timeout and the ordered-for-keyed rule.
  Harness tests assert ordering/barriers (not just counts), the empty-wake-set guard, the slow-head
  bound, prompt cancellation, the timeout, and the ordered-vs-sync digest-stable oracle. `actor.py`
  module docstring gains the async-loop paragraph.
- **6.2 — wire the SINK end-to-end** (the simpler seam: no reorder/emit). `api/graph` kinds + leaf
  rule; `lower.py` conditional `CollectSink`; `plan.py` comment; `execute.py` `async_sink` branch; DSL
  `.sink()`/`SinkHandle`; the at-least-once/idempotency contract; the **distributed/placement path**;
  and the **CLI/pipelines/dashboard presentation of a write-only / single-source pipeline**. Tests: a
  sink-ending graph writes all rows, EOS drains in-flight, backpressure stalls upstream, the digest
  oracle, a cross-process sink, cloudpickle round-trip.
- **6.3 — wire the async TRANSFORM** (reorder/emit). `api/graph` `async_one_input`; `execute.py`
  branch; DSL `.map_async`/`.apply_async` + `Stream._extend(kind=)`; resolve the linear-shape question
  (`graph_from_pipeline`/`_validate_linear`). Tests in-proc + parallel + cross-worker: row
  conservation, ordered output equality vs a sync `MapBatch`, a keyed async enrich matching its sync
  equivalent.
- **6.4 — unordered mode (stateless only) + the NDVI rework + the design docs.** The unordered
  hard-barrier path restricted to stateless maps, with `async.in_flight` peak assertions; rework
  `examples/sentinel2_ndvi.py` per above; finalize `DESIGN.md` (corrected mechanism 4, extended
  mechanism 3 termination, the robustness/at-least-once row, new mechanism 8 "Async I/O stages"),
  `IMPLEMENTATION_PLAN.md` Stage 6, README run line if it changes, and glossary/reference entries
  (async stage, ordered vs unordered, in-flight bound).

## Open decisions (recommendations in parentheses)

1. **v1 scope of the keyed transform.** Ship the keyed `fetch` + state-mutating `integrate` in v1, or
   restrict v1 to the stateless `AsyncMapBatch` and add keyed integrate in 6.5? (*Recommend: ship the
   async **sink** first — it is the highest-value, simplest seam — and the **stateless** transform;
   land the keyed transform in a follow-up once the loop is proven.*)
2. **Unordered mode in v1 or defer.** Ordered-only is fully deterministic and covers both use cases;
   unordered is pure throughput and adds the hard-barrier drain path. (*Recommend: defer unordered to a
   follow-up; ship ordered-only.*)
3. **Timeout default.** Per-request `timeout_micros` defaulting off vs a sensible default. (*Recommend:
   off by default, fail-fast on expiry, with a documented default-value/skip seam later.*)
4. **Sink commit model.** At-least-once via `close()` at EOS is v1; also wire `async on_watermark`
   transactional commit in v1, or defer until a sink needs event-time commits? (*Recommend: defer the
   event-time commit; ship the EOS-time flush.*)
5. **`max_in_flight` default and shape** — 8 vs 32, global vs per-combinator, and how it interacts with
   the cross-process credit window so the in-flight bound, not the credit window, is the limiter.
   (*Recommend: 8, per-instance knob, document the interaction.*)
6. **`.sink()` return type** — a distinct `SinkHandle` (chaining is a type error) or a plain `Stream`
   (chaining rejected at compile by the leaf rule). (*Recommend: `SinkHandle`.*)
7. **Async two-input (join).** Out of scope — joins are CPU/state, not external I/O. (*Recommend:
   leave out.*)
