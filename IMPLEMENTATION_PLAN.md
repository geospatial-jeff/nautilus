# Nautilus — Implementation Plan

Staged build of a decentralized, entirely-streaming compute framework. See `DESIGN.md` for the
architecture and rationale. Each stage compiles, runs, and has a demo plus a property/fuzz-test gate.

## Stage 0: Skeleton + semantics core, single process (no IPC)
**Goal**: Frame model + operator contracts running in one process with in-memory channels, proving
unified bounded/unbounded semantics deterministically.
**Success Criteria**: bounded word-count returns a deterministic result; keyed tumbling windows fire
on watermark advance and flush at EOS; an idle input does not stop event-time progress; all gates
(ruff/black/mypy/import-linter) clean.
**Tests**: `test_records`, `test_watermark_tracker`, `test_mailbox`, `test_wordcount`,
`test_windowing`.
**Status**: **Complete**. Gates green.

Delivered: `core.records` (the sealed Frame union and Arrow-first `Batch`); `core.time`
(`Clock`/`TestClock`, watermark generation, and `WatermarkTracker` with idle exclusion); `state`
(`KeyContext`-threaded handles and `InMemoryStateBackend` with a snapshot/restore ABC); `core.operator`
(the source, one-input, and two-input families and the synchronous `Collector`); `windows`
(`TimeWindow` and the tumbling assigner); `runtime` (`InProcChannel`, the non-reordering `Mailbox`, the
`Forward`/`Broadcast` partitioners, the `run_source`/`run_transform` actors, and `run_local_chain`);
the built-in `operators`; `testing` helpers; and the import-linter contract that keeps the control
plane off the data path.

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
**Status**: **Complete** — `nautilus.transport` (`framing`, `socket_channel`, `process`). A
`SocketChannel` is a drop-in `Channel` with a credit window for data frames and a credit-exempt path
for control frames, carrying framed Arrow-IPC data and msgpack control over a TCP connection (loopback
between two local processes, the same channel a cluster uses between nodes). The producer ends a stream
with `finish()`, which half-closes the write side and drains the returning credits to the consumer
before `close()`, so teardown does not discard in-flight data with a TCP reset. `run_two_process` runs
the source in one process and the transforms plus sink in another, joined by a single TCP edge. The
import-linter contract is extended to `nautilus.transport`.

The transport began over a Unix-domain socket to prove one process boundary; it now runs over TCP,
which is what a cluster uses between nodes, so the channel ports to multi-node unchanged (only the
connect address differs). A single linear edge does not exercise the topology a real deployment runs —
operator parallelism and the keyed shuffle — which is Stage 1.5.

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

# Telemetry workstream

Decoupled, self-describing telemetry: the analysis happens outside nautilus. See `DESIGN.md` for the
architecture. Threads into Stages 1–4, which populate the reserved catalog keys.

## Telemetry S1: model + catalog + recorder + sink + report skeleton + contracts
**Goal**: data-path model + boundary report types, with the data-path/cluster and report firewalls
enforced.
**Status**: **Complete** — `nautilus.telemetry` (`model`, `catalog`, `recorder`, `registry`) and
`nautilus.telemetry.report` (`report`, `serialize`, `sink`), with a versioned `RunReport` schema and
the three import-linter firewalls (data path vs. cluster, data path vs. report, telemetry data path vs.
report) in place. Gates green.

## Telemetry S2: instrument the data path + boundary assembly + `RunResult`
**Goal**: every run ships telemetry end-to-end, no behavior change, no broken callers.
**Status**: **Complete** — `run_source`, `run_transform`, and `Output` are instrumented (throughput,
per-operator timing histograms, backpressure and input-wait, watermark/EOS accounting, and
lifecycle/error events). `run_local_chain` builds the `RecorderRegistry`, assembles the `RunReport`,
and returns a `RunResult` that wraps the output batches and adds `.telemetry`, so existing callers are
unchanged. The `bytes_*` metrics are gated to the FULL tier behind a no-op-counter guard, so a run does
no buffer-size walk otherwise.

## Telemetry S3: agent-facing serialization + Python API + author-metric API
**Goal**: make the telemetry readable by an agent and extensible by operator authors.
**Status**: **Complete** — `to_markdown(token_budget)` leads with the run summary, then axis-explicit
rankings by self-time and send-wait, and never drops errors. Query helpers (`.operator()`, `.edge()`,
`.ranked_by()`, `.by_self_time()`, `.by_send_wait()`, `.by_rows_per_sec()`) sort and project but do not
diagnose. `build_report` merges multiple snapshots per `(operator_id, subtask)`, and `ctx.metrics` is
wired so `KeyedCount` and `KeyedTumblingSum` emit `window.fires`.

## Telemetry S4: generated self-describing reference + docs
**Goal**: ship the offline self-description an agent reads, locked against drift.
**Status**: **Complete** — `nautilus.telemetry.report.reference.render_reference()` generates
`docs/telemetry-reference.md` from the catalog; `python -m nautilus.telemetry.report.reference`
regenerates it; a no-drift test asserts the committed file matches; DESIGN.md telemetry section added.

---

# CLI (`nautilus`)

**Status**: **Complete** — a Typer + Rich CLI (`nautilus.cli`, console script `nautilus`, also
`python -m nautilus`). Commands: `run` (run a pipeline, then show an output preview plus a telemetry
summary, markdown, or JSON, with `--save`), `examples`, `catalog` (the metric reference, `--md`),
`reference` (`--write`), `dashboard` and `serve` (a live or saved telemetry dashboard over HTTP),
`version`, and `task <desc> --on <pipeline>`, which prints a ready-to-paste agent prompt (the task, the
run's telemetry, the definition of each metric, and the files to read). A pipeline resolves by built-in
name or `module:function`.
