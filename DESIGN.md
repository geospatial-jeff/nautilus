# Nautilus — Design

A decentralized, entirely-streaming parallel compute framework. Conceptually "like Dask", but it
rejects Dask's centralized scheduler and batch task-graph in favor of a streaming actor dataflow
where the dataflow graph defines the computation and nothing central sits on the data path.

## Locked constraints

| Concern | Choice |
|---|---|
| Runtime | Python + asyncio; multicore via worker **processes** (one event loop each) |
| Model | **Unified streaming** — bounded data is a finite stream that terminates |
| Coordination | **Actor dataflow + backpressure**; routing is local, never centrally scheduled |
| Scope | Single-node multicore first; multi-node via seams (swappable transport, key-groups) |
| Data | **Arrow-first** — records flow as Arrow `RecordBatch`es, micro-batched |
| Robustness | At-least-once + fail-fast whole job; exactly-once deferred (`Barrier` slot reserved) |

## Layers

Each layer hands the next exactly one artifact; layers never reach across.

```
nautilus.api        LogicalGraph (frozen IR)      fluent DSL  (Stage 3)
nautilus.compile    PhysicalPlan                  one-time lowering  (Stage 2)
nautilus.runtime    actors, channels, mailboxes   the data path
nautilus.transport  framed Arrow-IPC + control    unix now, TCP later  (Stage 1/4)
nautilus.cluster    placement, launcher, …        CONTROL PLANE ONLY  (Stage 2)
```

"No central scheduler" is scoped precisely: a Compiler, Deployer, startup barrier, `CollectSink`,
and completion detector are deliberate, bounded centralizations confined to the one-time control
phase or to job boundaries; none grant data credits or gate per-record progress. The boundary is
enforced mechanically — `nautilus.runtime`/`core`/`transport` may not import `nautilus.cluster`
(an import-linter contract in CI).

## The Frame model (`nautilus.core.records`)

Every edge carries two kinds of frame:

* **data** frames — only `Batch` (an Arrow `RecordBatch`), routed by the edge's partitioner.
* **control** frames — `Watermark`, `EOS`, `StatusIdle`/`StatusActive`, and (reserved) `Barrier`,
  always **broadcast** to every downstream instance.

A `Batch` column may be an Arrow `fixed_shape_tensor` extension column for imagery and embeddings:
one tensor per row (row-major), the shape in the column type, and the column length as the batch
dimension. `nautilus.tensors` converts these to and from numpy.

Event time is integer microseconds. `EOS` means "advance this input's watermark to `WATERMARK_MAX`";
real event times are kept strictly below that sentinel so they can never collide.

## Core mechanisms

1. **Backpressure** — bounded channels suspend a fast producer until the consumer drains. Across
   process boundaries (Stage 1) this becomes credit-based flow control with a dedicated reader/writer
   split so control frames never stall behind saturated data.
2. **Non-reordering fan-in** (`runtime.mailbox.Mailbox`) — one outstanding `recv` per input channel,
   `FIRST_COMPLETED` merge that re-arms only the yielded channel, guaranteeing per-channel FIFO. This
   ordering invariant is what makes watermark correctness hold.
3. **Watermark combination** (`core.time.WatermarkTracker`) — operator watermark = minimum over
   **non-idle** inputs, monotonic. Idle inputs are excluded so a silent partition cannot freeze
   progress, and a rejoining input never regresses the combined watermark.
4. **Termination** — an operator forwards `EOS` only after receiving it on *all* inputs; reaching the
   final input advances the watermark to `WATERMARK_MAX`, which flushes every pending window. The job
   ends when all sinks see `EOS` — no central poller.
5. **Synchronous critical section** — `process`/`on_watermark` never `await`; they emit into an
   in-memory `Collector` and the actor performs all backpressured sends *between* steps. Each
   per-batch step is therefore race-free, matching GIL reality.
6. **Keyed state** (`nautilus.state`) — scoped by `(operator_id, name, key, namespace)` and accessed
   through a `KeyContext` captured by each handle (no shared mutable "current key" cursor).
   `snapshot`/`restore` are in the ABC from day one so a spilling/checkpointing backend is additive.

## Telemetry (a core principle: nautilus ships the data agents use to develop it)

Every run emits comprehensive, **self-describing** telemetry so an external agent can read the data,
identify perf issues / bugs / optimizations, change code, re-run, and compare. The system ships
*facts*, never verdicts — there is no built-in diagnostics engine. Two layers:

- **`nautilus.telemetry`** (data-path): `model` (Counter/Gauge/fixed-bucket Histogram/Event/
  Snapshot), `catalog` (the frozen `MetricSpec`/`EventSpec` source of truth — each metric carries a
  unit, labels, `meaning`, `relates_to`, `derivation`; a banned-word lint forbids causal language),
  `recorder` (single-writer, lock-free, tier-gated; zero-cost when off), `registry`.
- **`nautilus.telemetry.report`** (boundary, forbidden from the data path by import-linter): the
  versioned `RunReport` tree, deterministic `to_json` + a token-budgeted `to_markdown` agent digest,
  query helpers that sort/filter/project (never diagnose), and a generated
  `docs/telemetry-reference.md` (run `python -m nautilus.telemetry.report.reference`).

Instrumentation only ever writes to the recorder, so a future live scrape endpoint is
a new sink with zero instrumentation change. A run returns `RunResult` (a `Sequence[RecordBatch]` with
an additive `.telemetry`). `structural_digest()` covers only provably-deterministic facts, so report
tests are stable and a future benchmark/diff tool has a stable schema. See `docs/telemetry-reference.md`.

## Status

Stage 0 is complete: the semantics core runs in a single process over in-memory channels and proves
unified bounded/unbounded behavior (word-count, tumbling windows, idle handling) deterministically.
The telemetry subsystem is complete and on by default: every run ships a self-describing `RunReport`
(`result.telemetry`) with a JSON surface, a markdown agent digest, and a generated reference.
See `IMPLEMENTATION_PLAN.md` for the remaining engine stages (credit transport, multicore deploy,
full DSL, multi-node seams), which populate the telemetry catalog's reserved keys.
