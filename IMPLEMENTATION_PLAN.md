# Nautilus — Implementation Plan

Staged build of a decentralized, entirely-streaming compute framework. See `DESIGN.md` for the
architecture and the approved plan for full rationale. Each stage compiles, runs, and has a demo
plus a property/fuzz-test gate.

## Stage 0: Skeleton + semantics core, single process (no IPC)
**Goal**: Frame model + operator contracts running in one process with in-memory channels, proving
unified bounded/unbounded semantics deterministically.
**Success Criteria**: bounded word-count returns a deterministic result; keyed tumbling windows fire
on watermark advance and flush at EOS; an idle input does not freeze event-time progress; all gates
(ruff/black/mypy/import-linter) clean.
**Tests**: `test_records`, `test_watermark_tracker`, `test_mailbox`, `test_wordcount`,
`test_windowing`.
**Status**: **Complete** — 21 tests passing; gates green.

Delivered: `core.records` (sealed Frame union, Arrow-first `Batch`), `core.time`
(Clock/TestClock, watermark generation, `WatermarkTracker` with idle exclusion), `state`
(KeyContext-threaded handles, `InMemoryStateBackend` with snapshot/restore ABC), `core.operator`
(Source/OneInput/TwoInput families, synchronous `Collector`), `windows` (TimeWindow, tumbling
assigner), `runtime` (`InProcChannel`, non-reordering `Mailbox`, `Forward`/`Broadcast` partitioners,
`run_source`/`run_transform` actors, `run_local_chain`), `operators` (built-ins), `testing` helpers,
and the import-linter "no central scheduler on the data path" contract.

## Stage 0.5: Tensor columns (imagery + embeddings)
**Goal**: numpy-array columns for N-D imagery and 1-D embeddings, carried as Arrow
`fixed_shape_tensor` extension columns, with numpy conversion helpers and no core changes.
**Success Criteria**: `tensor_array`/`embedding_array`/`to_numpy` round-trip `(N,H,W,C)` and `(N,dim)`
arrays; tensor columns pass through `MapBatch`/`FilterRows` and `FULL`-tier byte accounting unchanged;
an embedding's `.storage` is `fixed_size_list<float32, dim>`; gates (ruff/black/mypy/import-linter)
clean.
**Tests**: `test_tensors` (round-trip, operator pass-through, sliced `to_numpy`, byte accounting,
storage type, error cases); the `image-embed` example runs end to end.
**Status**: **Complete**.

## Stage 1: Credit backpressure + cross-process transport (2 processes, TCP)
**Goal**: correct, non-deadlocking credit-based backpressure across a process boundary.
**Success Criteria**: fast producer / slow consumer stalls with bounded memory (conservation
invariant holds); control frames (watermark/EOS) delivered while the data channel is saturated; diamond makes
progress; stray/duplicate credit cannot exceed the window; shutdown drops no in-flight data.
**Tests**: framing round-trip; credit conservation fuzz test; control-during-saturation;
diamond no-deadlock; graceful-shutdown no-loss over TCP; 2-process row conservation at high capacity.
**Status**: **Complete** — `nautilus.transport` (`framing`, `socket_channel`, `process`): a
`SocketChannel` (drop-in `Channel`) with a credit window for data frames and a credit-exempt path for
control frames; framed Arrow-IPC data + msgpack control over a TCP connection (loopback between two
local processes — the same channel a cluster uses node-to-node). The producer ends a stream with
`finish()`: half-close the write side and drain the returning credits to the consumer's end before
`close()`, so teardown never RSTs away in-flight data. `run_two_process` runs the source and the
transforms + sink in two processes joined by one TCP edge. 16 tests (framing incl. tensor columns,
credit conservation, control-not-blocked-by-saturation, diamond no-deadlock, stray-credit rejected,
peer-death robustness, graceful-shutdown no-loss over real TCP, 2-process word-count matches
single-process, 2-process row conservation at capacity 16). Import-linter contract extended to
`nautilus.transport`.

The transport began as a Unix-domain-socket proof of one process boundary; it now runs over TCP
because TCP is what a cluster uses between nodes, so this channel ports to multi-node unchanged (only
the connect address differs). A single linear edge does not exercise the topology a real deployment
runs — operator parallelism and the keyed shuffle — which is Stage 1.5.

## Stage 1.5: Parallel topology over TCP (the shape a cluster runs)
**Goal**: prove operator parallelism and the keyed shuffle (all-to-all repartition by key) behind the
`Channel` ABC — in-process first, then with the cross-worker edges as TCP `SocketChannel`s — because a
worker runs N parallel instances with shuffles between them, which the single linear Stage 1 edge
never tested. Defer the coordinator (placement/launcher/membership) to Stage 2.
**Success Criteria**: an operator runs as N instances, each owning a key range; a hash partitioner
routes each upstream batch to the owning downstream instance; `Mailbox` fan-in conserves rows; the
same graph runs unchanged whether an edge is in-process or a TCP `SocketChannel`.
**Tests**: hash-partition routing (each key to exactly one instance); row conservation across a
shuffle; in-process vs TCP edge parity on the same graph.
**Status**: Not Started.

## Stage 2: Compile + deploy + decentralized placement/termination (cluster control plane)
**Goal**: `LogicalGraph` → N worker processes via a coordinator (placement, launcher, membership),
replacing the hard-coded `run_two_process` split; no central scheduler on the data path; decentralized
EOS termination; key-group partitioning. Cross-host is the Stage 1 `SocketChannel` with a node address
instead of loopback — no channel changes.
**Status**: Not Started.

## Stage 3: Full DSL, two-input join, Arrow hot path, observability, robustness
**Goal**: complete the fluent API + geospatial/columnar performance path + debuggability.
**Status**: Not Started.

## Stage 4: Validate multi-node seams (design-only)
**Goal**: same `PhysicalPlan` runs over loopback and a real node-to-node TCP connection with zero
operator/channel changes (the transport is already TCP as of Stage 1; this validates cross-host
addressing, connection setup/retry, and security).
**Status**: Not Started.

---

# Telemetry workstream (core principle: nautilus ships the data agents use to develop it)

Decoupled, self-describing telemetry — analysis lives outside the system. See `DESIGN.md` and the
approved plan for the architecture. Threads into Stages 1–4 (which populate reserved catalog keys).

## Telemetry S1: model + catalog + recorder + sink + report skeleton + contracts
**Goal**: data-path model + boundary report types, with the data-path/cluster and report firewalls
enforced.
**Status**: **Complete** — `nautilus.telemetry` (`model`, `catalog`, `recorder`, `registry`) +
`nautilus.telemetry.report` (`report`, `serialize`, `sink`); `REPORT_SCHEMA_VERSION=1`; 3 import-linter
contracts KEPT; 19 telemetry tests (banned-word lint, name↔unit, histogram bucketing, tier gating,
pickle round-trip, build_report, structural-digest stability). All gates green.

## Telemetry S2: instrument the data path + boundary assembly + `RunResult`
**Goal**: every run ships telemetry end-to-end, no behavior change, no broken callers.
**Status**: **Complete** — `run_source`/`run_transform`/`Output` instrumented (throughput, per-op
timing histograms, backpressure/input-wait, watermark/EOS accounting, lifecycle/error events);
`run_local_chain` builds the `RecorderRegistry`, assembles the `RunReport`, and returns `RunResult`
(a `Sequence[RecordBatch]` subclass, so existing callers are unchanged). `bytes_*` gated FULL-only
behind a no-op-counter guard (no buffer-size walk otherwise). 7 new tests: rows-conserved-across-edges,
topology-edges-resolve, 50×-stable structural digest, OFF-tier zero-catalog-lookup guard, bytes FULL-only.
All 47 tests + gates green.

## Telemetry S3: agent-facing serialization + Python API + author-metric API
**Goal**: make the telemetry readable by an agent and extensible by operator authors.
**Status**: **Complete** — `to_markdown(token_budget)` (RunSummary first, axis-explicit rankings by
self-time / send-wait, errors never dropped); query helpers `.operator()/.edge()/.ranked_by()/
.by_self_time()/.by_send_wait()/.by_rows_per_sec()` (project/sort only, no diagnosis); `build_report`
merges multiple snapshots per `(operator_id, subtask)`; `ctx.metrics` wired so `KeyedCount`/
`KeyedTumblingSum` emit `window.fires`. 5 tests incl. markdown-numbers ⊆ JSON.

## Telemetry S4: generated self-describing reference + docs
**Goal**: ship the offline self-description an agent reads, locked against drift.
**Status**: **Complete** — `nautilus.telemetry.report.reference.render_reference()` generates
`docs/telemetry-reference.md` from the catalog; `python -m nautilus.telemetry.report.reference`
regenerates it; a no-drift test asserts the committed file matches; DESIGN.md telemetry section added.

---

# CLI (`nautilus`)

**Status**: **Complete** — a Typer + Rich CLI (`nautilus.cli`, console script `nautilus`, also
`python -m nautilus`). Commands: `run` (run a pipeline → output preview + telemetry summary/markdown/
json, `--save`), `examples`, `catalog` (telemetry cheat-sheet, `--md`), `reference` (`--write`),
`version`, and `task <desc> --on <pipeline>` — prints a ready-to-paste agent prompt (task + run
telemetry + per-metric definitions + file pointers). Pipelines resolve by built-in name or
`module:function`. 9 CLI tests via Typer's `CliRunner`.
