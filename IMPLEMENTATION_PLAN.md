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

### Stage 4 — Multi-node via docker-compose · Planned

The same `PhysicalPlan` runs across separate containers addressed by service DNS — only how a worker is
*started* changes, not an operator or a channel. The data plane is already cross-host (the Stage 1
`SocketChannel`/`EdgeListener` over TCP, dialed through the `AddressBook`); what is single-machine is the
control plane — local process spawn, the `multiprocessing` queues carrying control messages, and
exit-code crash detection. Stage 4 networks those three primitives behind one seam and leaves the data
path untouched. The model is a long-lived worker daemon the coordinator dials, chosen as the foundation
for the eventual Kubernetes deployment (each worker a Pod behind a stable Service DNS name, the
coordinator a Job that dials them; the bind-vs-advertise split below maps straight onto a Pod that binds
`0.0.0.0` and advertises its Pod DNS); local spawn stays the default for a single-machine run. Security
is **out of scope** here — Stage 4 is correct only on an isolated, trusted network (see Stage 5).
Landing in independently-shippable sub-stages, each green across pytest / mypy / ruff / black /
import-linter:

- **4.0 — Worker-cohort seam · Planned.** A `WorkerCohort` ABC abstracts the three machine-specific
  control primitives behind `send` / `next_event(watch=)` / `reap`. `LocalCohort` wraps today's spawn +
  `multiprocessing.Queue` + exit-code path unchanged, so `deploy`'s body and every `test_cluster_*` stay
  byte-for-byte — a pure refactor that introduces the seam the remote path plugs into.
- **4.1 — Bind-vs-advertise + bounded dials · Planned.** A worker binds all interfaces but *registers* a
  separate routable advertised address, because `getsockname()` on a `0.0.0.0` bind returns the
  undialable `0.0.0.0`. The data dial gains a connect timeout and the data sockets gain TCP keepalive, so
  a misadvertised peer or a mid-job partition becomes a bounded error instead of an indefinite hang. The
  local default keeps advertise == bind == loopback, so existing runs are unaffected.
- **4.2 — Control link + daemon + RemoteCohort · Planned.** `nautilus worker` is a long-lived daemon the
  coordinator dials; `cluster.control_link` frames `Launch`/`Abort` down and `Register`/`Done`/`Failed`
  up one TCP control connection per worker. A control-connection drop before `Done` aborts the job (the
  network replacement for a missing exit code); a wedged abort self-terminates the daemon out-of-band
  (the replacement for the local SIGKILL); a normal job end returns the daemon to idle for the next
  `Launch`. A hermetic loopback test runs `deploy(daemons=…)` against subprocess daemons, so the
  multi-node *control* path is green without Docker.
- **4.3 — docker-compose harness · Planned.** A Dockerfile and `docker-compose.yml` run N worker daemons
  plus a coordinator on one bridge network, addressed by service DNS, with healthcheck/`depends_on`
  ordering so the coordinator dials only bound daemons. Telemetry gains a physical-host attribute (sourced
  from each daemon's identity, the k8s Pod name later) alongside the logical `worker-{id}` node, so a
  multi-node report shows *which container* an operator ran on — without it the report collapses every host
  to its worker id, blinding the development loop. A Docker-marked, skipped-by-default integration test
  forces a cross-container keyed shuffle and asserts the distributed result matches a single-process run by
  multiset and structural digest, that the keyed operator ran on more than one worker node, and that the
  per-host attribute holds the distinct container names. The repo's first CI workflows land here (the base
  gates, plus a separate Docker job).

### Stage 5 — Security · Planned

Stage 4 runs across a trusted compose network; Stage 5 makes it safe on an untrusted one — scoped here,
not yet designed in depth:

- **Schema the control wire.** The plan (cloudpickle) and the rest of the control messages (pickle) are
  arbitrary-code-execution on receipt over TCP. Move the structured fields to a schema'd codec and the
  snapshots to a typed path; the kind-tagged framer already leaves room.
- **Authenticate both planes.** A shared secret or mTLS gates the control `Launch`/`Abort` path and the
  data-edge handshake, so an unidentified peer can neither run a plan nor inject frames into an edge.
- **Authorize the control port.** Restrict who may submit or abort a job, even once authenticated.
- **Encrypt both planes (TLS).** Plan bytes, Arrow batches, and a failed worker's traceback cross in
  clear today.
- **Contain the `0.0.0.0` bind and the dashboard.** Harden the all-interfaces bind and the
  `dashboard`/`serve --host 0.0.0.0` telemetry HTTP exposure.
- **DoS hardening.** Rate-limit and cap connections on the control and data listeners; the frame-length
  guards bound only a single allocation, and the liveness timeouts are not a security boundary.

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
