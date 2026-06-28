# Nautilus — Implementation Plan

What's built and what's next; each stage lands with a demo and property/fuzz tests. See `DESIGN.md`
for the architecture and rationale.

## Engine stages

### Stage 0 — Semantics core, single process · **Done**

The frame model and operator contracts run in one process over in-memory channels, with deterministic
bounded and unbounded behavior: word-count, keyed tumbling windows that fire on watermark advance and
flush at EOS, and idle inputs that don't stall event time.

### Stage 0.5 — Tensor columns · **Done**

N-D imagery and 1-D embeddings ride as Arrow `fixed_shape_tensor` columns, with numpy conversion
helpers and no changes to the core.

### Stage 1 — Credit backpressure across a process boundary · **Done**

A `SocketChannel` carries framed Arrow-IPC over TCP, gating data frames with a credit window while
control frames stay credit-exempt; a fast producer stalls with bounded memory, and graceful shutdown
loses no in-flight data. It is the cross-process edge a cluster uses between nodes (Stage 2 dials it
through the `Connector`); on its own it does not yet exercise operator parallelism or the keyed shuffle.

### Stage 1.5 — Parallel topology and the keyed shuffle · **Done**

An operator runs as N instances, each owning a key range, with the keyed shuffle (`HashPartitioner`,
generalized to `KeyGroupPartitioner` in Stage 2) routing each batch to the owning instance and `Mailbox`
fan-in conserving rows. The per-instance report groups by `(operator_id, subtask_index, node)`. Stage 2
subsumed the original single-process channel mesh into the compiler and executor, and `nautilus run
--parallelism`/`--workers` drives parallelism from the CLI; Stage 3 made the fluent `Stream` DSL
(`.run(parallelism=, workers=)`) the way to express it.

### Stage 2 — Compile, deploy, decentralized control plane · **Done**

A `LogicalGraph` (`nautilus.api`) compiles to a serializable `PhysicalPlan` (`nautilus.compile`) and runs
through a per-worker executor over an injected `Connector` (`nautilus.runtime`), single-process or across
W worker processes via a coordinator — placement, two-phase bootstrap, launcher, key-group partitioning,
and symmetric EOS-draining teardown (`nautilus.cluster.deploy`). Co-located edges stay in-process; only a
true shuffle crosses workers, over the Stage 1 `SocketChannel` reached by a node address. `nautilus run
--workers/--parallelism` drives it; telemetry aggregates at the coordinator with per-worker attribution.

### Stage 3 — Full DSL, two-input join, Arrow hot path · **Done**

A fluent graph-building API, an inner streaming equi-join, and a columnar shuffle. Landed in
independently-shippable sub-stages, each green across pytest / mypy / ruff / black / import-linter:

- **3.0 — Arrow hot path · Done.** The keyed shuffle (`runtime.partition._route_keyed`) routes by
  `dictionary_encode` → per-distinct-key bucket → `take`/`filter` instead of a per-row Python loop,
  byte-for-byte identical to the old routing (a fuzz oracle pins the rid→instance map; the structural
  digest is unchanged). +24% on `bench-keyed` at 1000 keys, 2.3× at 50 keys. (`PERFORMANCE_CHANGELOG`.)
- **3.1 — DAG IR + DAG-aware compiler · Done.** `LogicalGraph` now carries an explicit `LogicalEdge`
  list with per-edge input ports and keying, so a two-input join is expressible; `compile_graph` lowers
  the DAG (deterministic topological order, position-derived ids, the join's two edges sharing one
  group table). A linear graph carries no edges and compiles byte-for-byte as before.
- **3.2 — Two-input actor + executor wiring · Done.** `run_transform` and a new `run_two_input` share one
  loop core; the two-input one dispatches each batch to `process_left`/`process_right` by its input's
  side, combines watermarks as `min(left, right)`, and forwards EOS after both ports close. The executor
  wires a port-ordered mailbox and one Output per outbound edge (list-valued edge maps — also the latent
  fan-out edge-loss fix).
- **3.3 — `HashJoin` operator · Done.** The concrete inner symmetric-hash equi-join: buffers both sides
  by key, emits each match as the later side arrives (order-independent), drops the right's key columns
  and rejects an output column-name collision, clears at EOS. Verified in-process, parallel (co-partition),
  and across worker processes (distributed result + digest match single-process).
- **3.4 — Fluent `Stream` DSL · Done.** `nautilus.dsl.Stream` (`source(...)` → `map`/`filter`/`tokenize`/
  `count_by`/`tumbling_sum`/`apply`/`join` → `.run(workers=, parallelism=)`/`.collect()`) is the public
  surface for building a pipeline — immutable, join-capable, the same graph in-process and across workers.
  The boundary runners moved into a new `nautilus.driver` package (making the report-layer firewall a
  package-level import-linter contract, plus a fifth contract enforcing IR purity), and the redundant
  Stage-2 builder path (`Stage`/`graph_from_stages`/`run_parallel_chain`) is retired.

See `CODE_REVIEW.md` for the design forks these settled (join semantics, DSL surface, the hot path) and
the Stage-3 API-consolidation note.

### Stage 4 — Validate multi-node seams · Planned

The same `PhysicalPlan` runs over loopback and a real node-to-node connection with no operator or
channel changes — validating cross-host addressing, connection setup, and security.

## Telemetry · **Done**

Self-describing telemetry, with analysis left outside the engine (see `DESIGN.md`). Every run ships a
versioned `RunReport` — JSON, a token-budgeted markdown digest for agents, and the generated
`docs/telemetry-reference.md` — and a live HTTP dashboard serves the same report mid-run. Reserved
catalog keys fill in as Stages 1.5–4 land.

## CLI · **Done**

`nautilus` (also `python -m nautilus`): `run`, `examples`, `catalog`, `reference`, `dashboard` and
`serve`, `version`, `task` (prints a ready-to-paste agent prompt), and the benchmarking pair `bench` /
`bench-check` (median-of-trials throughput vs. a baseline; the CI regression gate). A pipeline is a
built-in name or `module:function`.
