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
subsumed the original single-process channel mesh into the compiler and executor; `run_parallel_chain` is
now a thin wrapper over them, and `nautilus run --parallelism`/`--workers` drives parallelism from the CLI.

### Stage 2 — Compile, deploy, decentralized control plane · **Done**

A `LogicalGraph` (`nautilus.api`) compiles to a serializable `PhysicalPlan` (`nautilus.compile`) and runs
through a per-worker executor over an injected `Connector` (`nautilus.runtime`), single-process or across
W worker processes via a coordinator — placement, two-phase bootstrap, launcher, key-group partitioning,
and symmetric EOS-draining teardown (`nautilus.cluster.deploy`). Co-located edges stay in-process; only a
true shuffle crosses workers, over the Stage 1 `SocketChannel` reached by a node address. `nautilus run
--workers/--parallelism` drives it; telemetry aggregates at the coordinator with per-worker attribution.

### Stage 3 — Full DSL, two-input join, Arrow hot path · Planned

The fluent graph-building API, a columnar performance path, and debuggability.

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
