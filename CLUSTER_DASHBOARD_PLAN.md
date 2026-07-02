# Nautilus — the live dashboard across workers · Plan

The live `nautilus dashboard` runs a pipeline in one process and serves that process's telemetry. This
feature makes it aggregate **every worker's telemetry live** while a distributed run is in flight, so a
cluster run is as observable as a single-process one. `DESIGN.md` carries the rationale; this file stages
the build and is retired when the stages land (as `ASYNC_IO_PLAN.md` was).

**The decision.** The coordinator already merges every worker's raw snapshots into one `RunReport` with
`build_report` — but only once, from the terminal `Done`. This feature ships those snapshots
*periodically* over the same control plane, so the coordinator rebuilds and serves the report mid-run.
Workers push; the coordinator merges and serves with the `LiveServer` the single-process dashboard
already uses. No new aggregation and no new server — the missing piece is a mid-run snapshot channel.

**The invariant it must not break.** Telemetry crosses **only the control plane**, as one new message
beside `Register`/`Done`/`Failed`. The coordinator still reads no data channel and grants no credit, so
the data path stays scheduler-free (`DESIGN.md`). Each worker snapshots *itself* on its own event loop —
the single-writer rule — exactly as it already does for `Done` and for the `SystemSampler`. The snapshot
push is off unless a dashboard is attached, so a plain `nautilus run` pays nothing and its report is
byte-for-byte unchanged.

**Placement (import-linter).** `nautilus.telemetry` may not import `nautilus.cluster`, so the orchestration
lives in `cluster` (a new `cluster/dashboard.py`), which may import the serving layer; the reusable
`LiveAggregator` lives in `telemetry.live` and imports only `telemetry.report`. `deploy` gains a plain
callback, not a `telemetry.live` import.

Each stage is green across pytest / mypy / ruff / black / import-linter, and **lands its own docs** — the
docstrings, comments, and DESIGN/CLI updates that stage's code needs are part of that stage, never a
later pass. The default single-process `dashboard`, `run`, and `deploy` paths stay unchanged throughout.

## Stage 1 — the periodic snapshot channel · Planned

A worker ships a point-in-time snapshot on an interval, over the path its `Done` already travels.

- `Heartbeat(worker_id, snapshots)` in `cluster.protocol` — a frozen dataclass beside `Done`, so it
  pickles over the local `mp.Queue` and cloudpickles over the control link with no framer change. Its
  docstring states it carries telemetry for the dashboard and is **not** a liveness signal (crash
  detection stays the exit code / control-connection close).
- `execute` gains an optional `heartbeat` callback + `heartbeat_interval_micros` and runs it as a periodic
  task *outside* the data `TaskGroup` — the `SystemSampler` pattern — calling
  `heartbeat(registry.snapshot_all())` between actor steps (safe: same loop, one synchronous reader). The
  task is cancelled with the sampler in the same `finally`. `execute` never names `Heartbeat` (the
  callback does), so the report-layer and cluster import contracts are untouched.
- `run_worker_slice` backs the callback with its existing `send_event` (queue or control socket), gated on
  a new `TelemetryConfig.heartbeat_interval_micros` (`None` = off). The field is digest-excluded for free —
  `config_digest` hashes only capacity/tier/topology.
- Tests: `execute` invokes the callback about once per interval with this worker's snapshots; a run with
  the field unset invokes it never.

## Stage 2 — the coordinator aggregates live · Planned

`deploy` keeps each worker's latest snapshot and rebuilds the report as heartbeats arrive.

- `deploy` gains an optional `on_report: Callable[[RunReport], None]` and a `heartbeat_interval_micros`.
  When `on_report` is given it sets the workers' interval, seeds `latest: {worker_id: snapshots}` from the
  roster (so an early report already lists every worker, the pending ones empty), and in the completion
  loop handles `Heartbeat` by updating `latest` and calling
  `on_report(build_report(⋃ latest, meta=<live meta>, topology))`. `Done` updates `latest` too, so a
  finished worker's final numbers persist while its peers run on. With no `on_report`, the loop and the
  returned report are byte-for-byte today's.
- Tests: a two-worker bounded run with an `on_report` collector sees at least one intermediate report
  whose process rows cover both nodes; the returned final report equals the non-live run's; default
  `deploy` is unchanged.

## Stage 3 — the coordinator serves the dashboard · Planned

- `LiveAggregator` in `telemetry.live` — a `Snapshotter` holding the last `RunReport` behind an atomic
  reference plus a status; `render_json` serializes it with `status`/`sampled_at_micros`. The HTTP thread
  reads lock-free with no loop hop, because (unlike the in-process `SnapshotSource`) the report handed in
  is already frozen.
- `serve_cluster(graph, *, num_workers, daemons, capacity, host, port, linger, max_seconds)` in
  `cluster/dashboard.py` — starts a `LiveServer` over a `LiveAggregator`, runs
  `deploy(on_report=aggregator.update, …)`, then (the bounded run finished) marks the aggregator completed
  and lingers on the final aggregated report until cancelled. The cluster counterpart to `serve_graph`.
- `DESIGN.md`: replace the parenthetical "the live dashboard stays single-process" (lines 192–194) with
  the multi-process live path and the control-plane-only invariant — the durable rationale lands here.
- Tests: `serve_cluster` over a bounded two-worker pipeline serves `/api/telemetry.json` carrying two
  process rows and both workers' operator rows; it lingers after completion and frees the port on cancel.

## Stage 4 — the dashboard renders the cluster · Planned

`dashboard.html` draws one hardware panel; a cluster run has W.

- Render one hardware panel (CPU / RSS MB / loop-lag sparklines) per worker node, keyed by the node label
  the report already carries; add a node column to the operators table; add a cluster-summary header
  (worker count, aggregate rows in → out, errors). The Stage-3 JSON already carries per-node rows, so this
  is presentation only — no server change.
- Tests: the served JSON exposes the per-node fields the page groups by (covered in Stage 3); a smoke load
  of the page returns 200 with every worker's panel.

## Stage 5 — the CLI drives it · Planned

- `nautilus dashboard` gains `--workers N` and `--daemons host:port,…`, mirroring `run`: `N == 1` and no
  daemons keeps the in-process `serve_graph`; otherwise `serve_cluster`. Reuses `_parse_daemons`.
- `docs/cli-reference.md` gains the two options; `IMPLEMENTATION_PLAN.md`'s Telemetry and CLI notes gain
  the multi-worker live path. Updated in this stage, with the code.
- Tests: `dashboard --workers 2 --max-seconds` over a tiny built-in pipeline serves an aggregated report
  and exits cleanly.

## Future work (out of this slice)

- **Parity hardware metrics** — network I/O, disk I/O, GIL contention, per-core CPU. New `SystemSampler`
  readings + catalog entries + a `telemetry-reference.md` regen; they ride the same `Heartbeat` unchanged.
  This is the intended next increment.
- **Per-operator task timeline** — a per-worker execution timeline needs cross-node clock normalization
  (one reference clock, a per-worker offset carried on the heartbeat). The seam is left open here; the
  panel and the offset are the added work.
- **Unbounded cluster dashboards** — `serve_cluster` lingers after a *bounded* run; an unbounded source
  under `--max-seconds` needs a cancellable `deploy`, not built here (batch-first runs are bounded).
- **Exposure hardening** — a dashboard bound to `0.0.0.0` is already the Stage 5 (security) surface in
  `IMPLEMENTATION_PLAN.md`; this feature adds no new exception to it.
