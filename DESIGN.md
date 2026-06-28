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
nautilus.dsl        Stream → LogicalGraph         fluent builder, value layer above the IR
nautilus.api        LogicalGraph (frozen IR)      explicit-edge DAG (linear chain or join)
nautilus.compile    PhysicalPlan                  one-time lowering  (Stage 2)
nautilus.runtime    actors, channels, mailboxes   the data path
nautilus.driver     RunResult                     the boundary: compile, run, build the report
nautilus.transport  framed Arrow-IPC + control    TCP, loopback now / cross-host  (Stage 1/4)
nautilus.cluster    placement, launcher, …        CONTROL PLANE ONLY  (Stage 2)
```

"No central scheduler" is scoped precisely: a Compiler, Deployer, startup barrier, `CollectSink`,
and completion detector are deliberate, bounded centralizations confined to the one-time control
phase or to job boundaries; none grant data credits or gate per-record progress. The boundaries are
enforced mechanically by import-linter contracts in CI: `nautilus.runtime`/`core`/`transport` may not
import `nautilus.cluster`; the report layer is assembled only in `nautilus.driver` (and the coordinator),
so the whole data-path package `nautilus.runtime` may not import it; and the IR (`nautilus.api`) imports
nothing else in nautilus, so a `LogicalGraph` stays a pure, serializable value.

The compiler's output, the `PhysicalPlan`, is the unit of distribution: a worker is handed a plan it
never compiled. So the plan is kept neutral — it carries operator factories and stateless partitioner
*specs* (not live partitioners) and a transport-free structural description (not a telemetry topology).
That is why `compile` imports neither the runtime nor the report layer (also an import-linter contract):
a plan must cloudpickle to a process that has only the data path, so a `RoundRobin`'s rotation cursor or
a report type must never ride along. A worker runs its slice of the plan and returns raw measurements;
the boundary — the single-process driver (`nautilus.driver`) today, the coordinator in a cluster —
translates the plan's neutral structure into the report topology and aggregates the workers' snapshots
into one report.

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
7. **Key groups and the rescale boundary** (`runtime.partition.KeyGroupPartitioner`) — a keyed edge
   hashes each key to one of a fixed number of key groups `G` (chosen once for the job, `G >= Q`) and
   routes by a static `group → instance` table the plan carries, rather than hashing straight to an
   instance. The indirection is the rescale seam: a key's group is fixed by the hash, and only the
   table maps groups to instances, so changing the instance count `Q` is a new table over the same
   groups — no key changes group. Stage 2 never moves live state: a rescale is a new job, not an online
   migration, so the table is computed once at compile and immutable for the run. At `G == Q` the table
   is the identity (the routing-level equivalence to a direct hash lives on `KeyGroupPartitioner`).
8. **Two-input join** (`core.operator.TwoInputOperator`) — the logical graph is an explicit-edge DAG, not
   only a linear chain, so an operator can have two inputs. A join is a two-input vertex fed by two keyed
   edges on distinct ports (port 0 left, port 1 right); both edges read the *join's* one parallelism and
   the run's one `G`, so their group tables are identical and an equal key co-partitions to the same join
   instance from either side. Its watermark is the same min-over-inputs combination as any fan-in
   (`min(left, right)`), and it forwards EOS only after *both* inputs close — the existing termination
   rule, unchanged for a second input. (A linear graph carries no edges; the compiler reads its positional
   adjacency, so it lowers byte-for-byte as before.) The built-in `HashJoin` is an inner equi-join whose
   result is independent of the order the two sides arrive; like the keyed aggregations it holds unbounded
   state until EOS — an accepted MVP tradeoff, since the inputs here are bounded. How it buffers and the
   `on_watermark` eviction seam for a future windowed variant are the operator's concern.

## Deployment (`nautilus.cluster`)

`deploy(graph, num_workers=W)` runs a graph across W spawned worker processes, coordinated by a control
plane that never touches the data path. The coordinator compiles once, computes placement, spawns the
workers, then only moves control messages and waits at the job boundary; it reads no data channel and
grants no credit, so "no central scheduler on the data path" still holds with a coordinator present.

**Placement** is per-operator round-robin over the workers: subtask *i* of every operator goes to worker
*i mod W*. Same-index subtasks co-locate, so a forward or diagonal edge stays a free in-process channel
and only a genuine shuffle crosses workers. Each worker therefore runs a *hybrid* connector — in-process
for co-located edges, a socket for cross-worker ones — wired by the same `execute` code as a
single-process run.

**Two-phase bootstrap** makes connection setup deadlock-free by construction. Every worker binds its
listener and registers its address (phase 1) *before* the coordinator broadcasts the address book that
lets anyone dial (phase 2). So by the time a worker dials, every destination listener exists; and because
a worker dials all of its outbound edges before it accepts any inbound — and a dial completes once the
peer's listener is bound, never waiting on the peer's accept — even a bidirectional mesh cannot
circular-wait. A connection that arrives before its consumer accepts is parked by the listener, so no
global "go" barrier is needed: credit and parking absorb startup skew, and a mailbox is always built with
its full input set before its actor starts.

**Teardown is symmetric.** On a clean stop each worker drains its outbound edges and closes its inbound
edges in one `gather`, so every worker emits its FIN at once; sequential finish-then-close would
circular-wait on a bidirectional mesh and make both workers eat the full drain timeout. On failure a
worker skips the drain and abortively closes, so a peer's `recv` raises promptly, and the coordinator
re-raises the child's traceback and reaps every worker. The coordinator is also the telemetry boundary:
workers return raw snapshots, and it translates the plan into the report topology and aggregates the one
`RunReport`.

## Telemetry

nautilus reports facts, never verdicts: every run emits self-describing measurements so an external
agent can read a run, find a problem, change the code, re-run, and compare. There is no built-in
diagnosis — deliberately, so the analysis can evolve outside the engine.

Telemetry is split in two, with a firewall between. Instrumentation on the hot path only records raw
numbers; it never assembles a report, so a run pays as little as possible for it. Building the report
is a separate, boundary-time concern, and import-linter forbids the per-record code from importing the
report layer, so report-building can never creep onto the hot path. Every reader sits downstream of the
recorders, which makes new readers additive: the returned `RunResult` and the live `nautilus dashboard`
read the same recordings, and neither required any change to instrumentation. That single-registry reader
model is per-process: a distributed run has one registry per worker, each worker ships its raw snapshots
to the coordinator, and the coordinator builds the one report at the job boundary (the live dashboard
stays single-process).

Every metric is declared once in a catalog — its name, unit, and a plain-language meaning — so a report
describes itself and its schema cannot drift from the code; a lint rejects any meaning written as
cause-and-effect. See `docs/telemetry-reference.md`.

## Status

The single-process semantics core, tensor columns, the credit transport, the telemetry subsystem, the
compiler + cluster control plane (compile a graph and deploy it across worker processes), the fluent
`Stream` DSL, and the two-input inner equi-join run today. Multi-node validation (Stage 4) is designed
but not built. `IMPLEMENTATION_PLAN.md` has the stage-by-stage detail.
