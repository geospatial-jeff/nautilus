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
loses no in-flight data. `run_two_process` joins two processes over one TCP edge — the channel a
cluster will use between nodes. It does not yet exercise operator parallelism or the keyed shuffle.

### Stage 1.5 — Parallel topology and the keyed shuffle · **Done**

An operator runs as N instances, each owning a key range, with a `HashPartitioner` routing each batch
to the owning instance and `Mailbox` fan-in conserving rows. `run_parallel_chain` wires the P×Q channel
mesh from a `ChannelFactory`, so the same graph runs unchanged over in-process channels or a TCP
`SocketChannel` (`SocketPairFactory`), and the per-instance report groups by `subtask_index`. A
`nautilus run --parallelism` CLI surface is deferred to Stage 2.

### Stage 2 — Compile, deploy, decentralized control plane · Planned

Lower a `LogicalGraph` to N worker processes via a coordinator (placement, launcher, membership),
replacing the hard-coded `run_two_process` split, with decentralized EOS termination and key-group
partitioning. Cross-host reuses the Stage 1 `SocketChannel` with a node address.

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
`serve`, `version`, and `task` (prints a ready-to-paste agent prompt). A pipeline is a built-in name
or `module:function`.
