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

nautilus reports facts, never verdicts: every run emits self-describing measurements so an external
agent can read a run, find a problem, change the code, re-run, and compare. There is no built-in
diagnosis — deliberately, so the analysis can evolve outside the engine.

Telemetry is split in two, with a firewall between. Instrumentation on the hot path only records raw
numbers; it never assembles a report, so a run pays as little as possible for it. Building the report
is a separate, boundary-time concern, and import-linter forbids the per-record code from importing the
report layer, so report-building can never creep onto the hot path. Every reader sits downstream of the
recorders, which makes new readers additive: the returned `RunResult` and the live `nautilus dashboard`
read the same recordings, and neither required any change to instrumentation.

Every metric is declared once in a catalog — its name, unit, and a plain-language meaning — so a report
describes itself and its schema cannot drift from the code; a lint rejects any meaning written as
cause-and-effect. See `docs/telemetry-reference.md`.

## Status

The single-process semantics core, tensor columns, the credit transport, and the telemetry subsystem
run today; the cluster control plane, the full DSL, and multi-node validation are designed but not
built. `IMPLEMENTATION_PLAN.md` has the stage-by-stage detail.
