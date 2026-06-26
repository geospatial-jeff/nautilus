# Nautilus — Design

A decentralized, entirely-streaming parallel compute framework. Like Dask, it runs parallel
computations, but it replaces Dask's central scheduler and batch task graph with a streaming actor
dataflow: the dataflow graph defines the computation, and no central component sits on the data path.

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

Each layer produces a single artifact for the next; no layer bypasses its neighbor to reach another.

```
nautilus.api        LogicalGraph (frozen IR)      fluent DSL  (Stage 3)
nautilus.compile    PhysicalPlan                  one-time lowering  (Stage 2)
nautilus.runtime    actors, channels, mailboxes   the data path
nautilus.transport  framed Arrow-IPC + control    TCP, loopback now / cross-host  (Stage 1/4)
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
   per-channel ordering is what keeps watermark handling correct.
3. **Watermark combination** (`core.time.WatermarkTracker`) — the operator watermark is the minimum
   over **non-idle** inputs, and is monotonic. Idle inputs are excluded so a silent partition cannot
   stop event-time progress; a rejoining input never moves the combined watermark backward.
4. **Termination** — an operator forwards `EOS` only after receiving it on *all* inputs; reaching the
   final input advances the watermark to `WATERMARK_MAX`, which flushes every pending window. The job
   ends when all sinks see `EOS` — no central poller.
5. **Synchronous critical section** — `process`/`on_watermark` never `await`; they emit into an
   in-memory `Collector` and the actor performs all backpressured sends *between* steps. Each
   per-batch step is therefore a race-free critical section under the GIL.
6. **Keyed state** (`nautilus.state`) — scoped by `(operator_id, name, key, namespace)` and accessed
   through a `KeyContext` captured by each handle (no shared mutable "current key" cursor).
   `snapshot`/`restore` are in the ABC from day one so a spilling/checkpointing backend is additive.

## Telemetry

Telemetry is a primary output: every run emits self-describing data so an external agent can read
it, find performance problems or bugs, change the code, re-run, and compare. nautilus reports *facts*
and draws no conclusions — there is no built-in diagnostics engine. Two layers:

- **`nautilus.telemetry`** (data path): `model` (Counter, Gauge, fixed-bucket Histogram, Event,
  Snapshot), `catalog` (the frozen `MetricSpec`/`EventSpec` source of truth — each metric carries a
  unit, labels, `meaning`, `relates_to`, and `derivation`; a lint forbids causal language),
  `recorder` (single-writer, lock-free, tier-gated, zero-cost when off), and `registry`.
- **`nautilus.telemetry.report`** (boundary, kept off the data path by import-linter): the versioned
  `RunReport` tree, a deterministic `to_json` and a token-budgeted `to_markdown` digest for agents,
  query helpers that sort, filter, and project (but never diagnose), and the generated
  `docs/telemetry-reference.md` (run `python -m nautilus.telemetry.report.reference`).

Instrumentation only ever writes to a recorder; every reader sits downstream of that. The in-process
`RunReport` is one reader; the live HTTP dashboard (`nautilus.telemetry.live`, served by `nautilus
dashboard`) is another, pulling `RecorderRegistry.snapshot_all()` between actor steps — neither needed
a change to instrumentation. A run returns a `RunResult`: the output batches plus an additive
`.telemetry` report. `structural_digest()` covers only provably-deterministic facts, so report tests
are stable and a future benchmark/diff tool has a fixed schema. See `docs/telemetry-reference.md`.

## Status

Complete: the Stage 0 semantics core (single process, in-memory channels, deterministic word-count,
tumbling windows, and idle handling), Stage 0.5 tensor columns, and the Stage 1 credit transport
(framed Arrow-IPC over TCP between two processes). The telemetry subsystem is complete and on by
default: every run ships a self-describing `RunReport` (`result.telemetry`) with a JSON surface, a
markdown digest for agents, and a generated reference. The remaining engine stages — parallel topology
and the keyed shuffle, compile-and-deploy across worker processes, the full DSL, and multi-node
validation — populate the telemetry catalog's reserved keys. See `IMPLEMENTATION_PLAN.md`.
