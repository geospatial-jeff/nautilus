# Stage 4 working plan — multi-node via docker-compose

This is the design and build sheet for Stage 4 (the terse "what's next" list lives in
`IMPLEMENTATION_PLAN.md`; this file is the *why* and the *how* you implement each sub-stage from). It is
a working document: as each sub-stage lands, its rationale moves into `DESIGN.md` and the module
docstrings per the funnel, and this file shrinks. Every load-bearing claim below was checked against the
code at the line cited.

**Goal.** Run the *same* `PhysicalPlan` across separate containers addressed by service DNS, proving a
keyed shuffle genuinely crosses a container boundary and the result matches a single-process run. Only
how a worker is *started* changes — no operator or channel change.

**Locked decisions.**

- **Launch model: a long-lived worker daemon the coordinator dials, static membership from a roster.**
  Chosen as the foundation for the eventual Kubernetes deployment: each worker becomes a Pod running the
  daemon (a Deployment/StatefulSet behind a headless Service for stable per-pod DNS), the coordinator a
  Job that dials them by Service DNS. The roster generalizes from a compose service list to the Service's
  endpoints; the bind-vs-advertise split (§A.5) maps straight onto a Pod that binds `0.0.0.0` and
  advertises its Pod DNS / `POD_IP`. Compose is the lower-cost rehearsal of that topology.
- **Daemon lifecycle: stays up, serves many jobs** (§A.4).
- **Control payload codec: cloudpickle/pickle now** — the smallest correct change, mirroring today's
  implicit `mp.Queue` pickling; the resulting remote-code-execution surface is owned entirely by Stage 5.
- **Data-plane keepalive + connect timeout: included in Stage 4** (§A.5) — a few `setsockopt`/timeout
  calls that close the worst genuine multi-host hang.
- **Security is out of scope** — Stage 4 is correct only on an isolated, trusted, non-published network.
  Everything security-shaped is descoped to Stage 5 (`IMPLEMENTATION_PLAN.md`).

---

## A. Architecture

The data plane is already multi-node: `SocketChannel`, `EdgeListener`, the handshake, and
`SocketConnector` carry framed Arrow-IPC over TCP and address peers by `(host, port)` (`transport/*`,
`cluster/membership.py`). What is single-machine lives in exactly three control-plane primitives across
four files:

- **process spawn** — `mp.get_context("spawn")` + `ctx.Process` start children on the local host only
  (`launcher.py:33,39-52`);
- **the two `mp.Queue` channels** — a shared `events` queue (workers→coordinator: `Register`/`Done`/
  `Failed`) and a per-worker `commands` queue (coordinator→worker: the `AddressBook`), each a local OS
  pipe (`launcher.py:34-35`, `worker_main.py:115,119`);
- **crash detection** — `recv_event` reads liveness from `proc.exitcode` (`rendezvous.py:53,59`).

Stage 4 networks those three behind one seam and leaves the data path untouched.

### A.1 Worker-launch and membership

Each container runs `nautilus worker`: a process that binds a control port and serves one job at a time.
`deploy()` is handed a roster of `(host, port)` daemon control addresses; it dials the first `effective`
of them, assigns `worker_id = roster index`, and sends each daemon a `Launch` carrying what are spawn
arguments today (`plan_bytes`, `placement`, `capacity`, `worker_config`, its `worker_id`).

The coordinator cannot spawn across hosts, and an `mp.Queue` is a local pipe, so the coordinator must
*connect to* something already running. Dialing a known roster preserves the existing invariant that the
coordinator knows the worker count and identities up front (`worker_ids = list(range(effective))`,
`coordinator.py:74`) and avoids the identity-collision failure mode of self-registering daemons (two
daemons claiming one `worker_id`, or one outside the plan). It matches `membership.py:5-6` ("membership
is fixed; a rescale is a new job"); compose service DNS names are known up front, so a static roster
(`worker-1:9000,worker-2:9000,…`) is the natural source.

**Roster length is an upper bound, capped exactly like `--workers` today.** `deploy` still computes
`effective = min(num_workers, max_parallelism(plan))` (`coordinator.py:69`) and the completion loop
watches only `range(effective)` (`coordinator.py:74,104`). Surplus daemons (index ≥ `effective`) are
never dialed, never watched, and never treated as crashed — otherwise `deploy` would hang awaiting a
`Done` from an idle daemon, or misread its control socket as a crash.

### A.2 The control-plane abstraction: `WorkerCohort`

`WorkerCohort` (an ABC in `nautilus.cluster`) is the seam that lets local-spawn and remote-connect share
`deploy`'s body. It exists because `recv_event` today *fuses* transport (`events.get`) with liveness
(`proc.exitcode`) in one function (`rendezvous.py:48-66`), and that fusion is exactly what is
machine-specific. Three operations:

- `send(worker_id, message)` — replaces the `AddressBook` broadcast `commands[wid].put(book)`
  (`rendezvous.py:88-89`).
- `next_event(timeout, watch: set[int] | None) -> message` — replaces `recv_event`; raises
  `WorkerCrashed`. `watch` carries the remaining-workers narrowing the completion loop already does
  (`coordinator.py:104-116`): a worker whose whole contribution is in its `Done` may exit or close during
  teardown and must not fail a finished run, so it leaves `watch`. This invariant moves into the cohort
  verbatim.
- `reap()` — replaces `launcher.reap` (`launcher.py:60-69`); unconditional in `deploy`'s `finally`
  (`coordinator.py:133-134`).

Two implementations:

- **`LocalCohort`** wraps today's `(procs, events, commands)` and **calls the existing `recv_event`/
  `bind_barrier`/`reap` unchanged** — it does not delete or reshape them. This keeps both the local code
  path and the direct-call tests green: `tests/test_cluster_rendezvous.py` drives `recv_event(events,
  [_FakeProc(exitcode)], timeout)` with a stdlib `queue.Queue`, so that function's signature and module
  location must survive.
- **`RemoteCohort`** holds one persistent, non-blocking TCP control socket per dialed daemon, each with
  its own read-reassembly buffer.

`deploy`'s body — `bind_barrier`'s two-phase logic, the one-`Done`-per-worker loop, report aggregation,
the `finally: reap` guarantee — is unchanged; it calls only cohort methods. `bind_barrier`
(`rendezvous.py:69-89`) is already backend-agnostic and is reused as-is: `LocalCohort` runs it directly;
`RemoteCohort` runs the same collect-every-`Register`-then-broadcast logic over its sockets.

**`RemoteCohort.next_event` must not head-of-line-block.** A `selectors` readiness event means "bytes are
available," not "a whole framed message arrived"; a control message can span TCP segments, and a `Done`
carrying the sink's full Arrow-IPC output is large. So `next_event` runs a blocking
`selectors.DefaultSelector` over the watched sockets; on readability it reads available bytes into that
socket's buffer and returns a message only when a full `[magic][kind][len][payload]` is assembled,
otherwise re-arms and loops. A socket is deregistered when its `Done` arrives (the watch-set narrowing).
A blocking selectors loop, **not** an asyncio loop, keeps `deploy` synchronous — the coordinator has no
event loop today and must not grow one.

### A.3 The network control transport: `cluster.control_link`

**Where it lives: a new self-contained module `nautilus.cluster.control_link`, not `nautilus.transport`.**
Contract 1 in `pyproject.toml` makes `nautilus.transport` a source and forbids it from importing
`nautilus.cluster`, so the wire that serializes cluster messages must live in cluster. Contract 1's
whole-package forbiddance also firewalls the new submodule from the data path for free — **no new
contract is required.** The control wire is kept in cluster rather than factored into `transport` because
the coordinator needs a *synchronous* framed read/write (its blocking selectors loop) while the daemon
needs an *asynchronous* one (it runs asyncio); `transport.framing.read_message` is async-only and coupled
to `core.records` frames (`framing.py:20-31,80`), so it is not reusable here. Duplicating a sync+async
opaque-bytes framer into a data-path package for a need transport itself never has is worse than the ~20
lines of self-contained framing in cluster, mirroring `handshake.py:38-79`.

**Layering trap (review blocker, resolved).** `control_link` must *not* import `cluster.membership`:
`membership.py:20` does `from nautilus.runtime.connector import ChannelId`, and an import-linter forbidden
contract expands packages, so a future defensive contract on the wire module forbidding `nautilus.runtime`
would find `control_link → membership → runtime.connector` and fail the gate. Instead the coordinator
sends the address book as a plain `dict[int, [host, port]]` and the daemon reconstructs the `AddressBook`
locally (it already imports `membership` via `worker_main.py:28`). The existing `cluster → runtime` edge
is legitimate and stays (`ChannelId` is a value type; `execute` is in `runtime.execute`). The simplest
decision is to add **no** within-cluster contract and rely on Contract 1.

**Wire format.** `[4-byte magic][1-byte kind][4-byte big-endian length][payload]`, mirroring
`handshake.py:49` and `framing.py:93`, with a `_MAX` length guard checked *before* allocating, like
`framing.py:34-36` / `handshake.py:31`. The guard is sized for the full sink output, because
`Done.sink_batches` carries the entire Arrow-IPC result of the sink-hosting worker inline
(`protocol.py:34-40`, `coordinator.py:115`).

**Payloads.** Every control payload is **cloudpickled**, except `Done.sink_batches`, which stays opaque
Arrow-IPC bytes nested inside `Done` (`encode_batches`, `protocol.py:51-59`) so canonical extension types
like `fixed_shape_tensor` survive. `placement` (tuple-keyed dict), `worker_config` (a `TelemetryConfig`
dataclass holding a clock), `capacity`, and `worker_id` all ride in one cloudpickled `Launch` bundle,
mirroring today's implicit `mp.Queue` pickling. The plan must be cloudpickle regardless (lambda operator
factories, `coordinator.py:82`), and cloudpickle is a superset of pickle, so one codec covers
`Register`/`Done`/`Failed`/`Launch`/`Abort`. **This pickle-over-TCP path is the Stage-5 RCE surface**; the
kind-tagged framing leaves room to swap any payload to a schema'd codec without touching the framer.

**Messages.** `Launch` and `Abort` (coordinator→daemon); `Register`, `Done`, `Failed` (daemon→
coordinator, reused from `protocol.py`); the address book as a plain dict inside the phase-2 broadcast.

**Crash detection without `proc.exitcode`.** The per-worker control connection is the liveness signal. An
EOF / `ConnectionReset` / incomplete read on a worker's control socket *before* its `Done` is
`WorkerCrashed` — the same rule the data plane already uses (`socket_channel.py:190-194`). One ordered TCP
connection makes this cleaner than the racy `mp.Queue`: `Done`-then-FIN is ordered, so the 1.0 s in-flight
grace window (`rendezvous.py:54-57`) is unnecessary. A connection-refused at dial time means the daemon is
not up yet (bounded by `bootstrap_timeout`; `RemoteCohort` retries the dial with backoff until then).

**Liveness bounds (review majors).** The completion wait stays `timeout=None` (`coordinator.py:96`,
`rendezvous.py:42-46`), so silence from a *partitioned* host (no FIN) must still surface. `SO_KEEPALIVE`
alone is a ~2-hour bound on Linux (`tcp_keepalive_time` 7200 s). So the control sockets set
`TCP_KEEPIDLE=10`, `TCP_KEEPINTVL=5`, `TCP_KEEPCNT=3` (≈25 s to detect a dead peer) — the interval is
stated in `DESIGN.md` so the partition bound is explicit. The data plane has the same gap: `SocketChannel`'s
read loop blocks on `readexactly` with no timeout (`socket_channel.py:176`), so a worker↔worker partition
mid-shuffle hangs both sides forever while each stays reachable from the coordinator. The same keepalive
triple is set on the `SocketChannel` sockets (a few `setsockopt` calls in `SocketConnector.outbound` and on
the accepted socket) so a mid-job data-plane partition becomes a bounded `ConnectionError`.

### A.4 No-orphan guarantee and the daemon state machine (review blockers)

Locally, `reap()` does terminate→join→kill — a SIGKILL that lands even on a worker wedged in a
non-yielding loop or a blocking C call (`launcher.py:60-69`). The coordinator cannot SIGKILL a non-child
PID across a machine, so the **daemon self-terminates its own PID**, and that self-kill must survive a
wedged event loop.

**The abort is job-scoped, not process-scoped.** The control connection is per-job. The daemon runs this
state machine, documented in `daemon.py`'s module docstring:

1. **Idle** — accept one coordinator control connection.
2. **Running** — read `Launch`, run a fresh `_run_worker` on a new event loop binding an ephemeral data
   port (no module or loop state persists between jobs), reporting `Register`/`Done`/`Failed` over that
   same socket. A background read watches the control socket.
3. **Control EOF/reset *before* this job's `Done`** = the coordinator is gone or aborting → cancel
   `execute()`, run the existing failure-path teardown (`execute.py:321-322` runs `connector.close()` in
   `finally`; the `CancelledError` skips `worker_main.py:140`'s `except Exception` and reaches the listener
   `close()` at `:143-147`), then return to **Idle**. The daemon does **not** exit.
4. **Control close *after* `Done`** = the normal job boundary (the coordinator's `reap()` closed the
   connection after reading `Done`) → return to **Idle**. The daemon does **not** exit.

So a normal job end leaves the daemon up for the next `Launch`, and a coordinator crash mid-job aborts the
job rather than orphaning `execute()`.

**The out-of-band backstop** is the network replacement for SIGKILL. asyncio cancellation is cooperative —
it only lands at an `await`, and a wedged operator (infinite loop or blocking C call inside the `TaskGroup`
at `execute.py:306`) never reaches one, and an asyncio-task "lease" never fires because a wedged
single-threaded loop runs no timers. So when an abort is requested, the daemon arms a genuinely
out-of-band watchdog — a `threading.Timer` (or `SIGALRM`) on a dedicated thread — that calls `os._exit()`
if `execute()` has not unwound within a bound. A wedged abort therefore kills the daemon process;
compose's `restart` policy brings a fresh daemon back. Stays-up holds for every normal job; hard-exit is
the fault path only, exactly as the local SIGKILL kills a wedged worker.

### A.5 Data-plane bind-vs-advertise fix

A worker registers `EdgeListener.address`, which is `getsockname()` — the *bind* result
(`listener.py:75`). Binding `0.0.0.0` to accept on the bridge interface makes `getsockname()` return
`0.0.0.0`, which no peer can dial (`connect(0.0.0.0)` hits the dialer's own loopback). The fix decouples
bind from advertise, with no protocol or wire change — `Register` already has separate `host`/`port`
fields (`protocol.py:24-31`):

- **Thread an `advertise_host`** `deploy → spawn_workers → worker_main → _run_worker`, replacing the
  single `host` with `(bind_host, advertise_host)`. `_run_worker` keeps `EdgeListener(bind_host, 0, …)`
  binding all interfaces, but registers `Register(worker_id, advertise_host, listener.address[1])` — the
  concrete ephemeral port from `getsockname`, the host from advertise. The local path defaults
  `advertise_host == bind_host == 127.0.0.1`, so every existing test stays green (`test_cluster_scale.py:137`
  registers only `127.0.0.1`).
- **In remote mode the daemon supplies its own `--bind` (default `0.0.0.0`) and `--advertise`**, and
  `deploy(host=)` is unused. **`--advertise` is required in the remote path** — there is no
  `socket.gethostname()` default, because in a compose container `gethostname()` returns the container
  short-ID, not a peer-resolvable name. The reliable advertise value is the compose service name / network
  alias (`--advertise worker-i`), resolved by `open_connection` at dial time (`connector.py:51-52`). A
  missing advertise fails fast.
- **Reject `0.0.0.0` and `""` as advertise values** in the registration path (`bind_barrier`, shared by
  both cohorts), so the misconfiguration fails fast with a clear message instead of silently dialing
  loopback. No current test registers such an address.
- **Add a connect timeout** to `asyncio.open_connection` (`connector.py:52`), which has none today, so a
  misadvertised peer hangs at OS TCP connect for minutes while crash detection never fires. The value is
  generous (e.g. 30 s) — sized for cross-host RTT, SYN-retransmit backoff, and listen-backlog drain when
  many producers dial one listener during a wide shuffle — but well below `bootstrap_timeout`. Add a small
  retry on a transient `getaddrinfo` failure for a worker→worker data dial (no retry today makes a
  transient DNS miss fatal).
- **Reword `listener.py:69-76`'s `address` docstring**: it is the bind result whose port is concrete; it
  is no longer "the address producers dial" once advertise differs from bind.

No change to `protocol.py` fields, `membership.py`'s map shape, `edge_resolver`, `SocketConnector`'s logic,
`handshake.py`, or `socket_channel.py`'s framing. The two-phase bootstrap's deadlock-freedom
(`DESIGN.md:119-126`) is pure connection-graph ordering and survives, but its premise shifts from
"`getsockname` is always dialable on loopback" to "every advertised address routes to its own listener,"
which `bind_barrier` does not validate — it is established only by config + the `0.0.0.0` reject + the
connect timeout. `DESIGN.md` states this precondition.

### A.6 Code availability in worker containers

Operators pickled **by value** travel inside `plan_bytes` and need no import in the daemon; operators
pickled **by reference** must be importable in every container. (Spawn does not inherit the parent's
modules — it re-imports fresh, which is why `test_cluster_deploy.py:36` calls
`cloudpickle.register_pickle_by_value(sys.modules[__name__])`.) The integration-test pipeline is therefore
an installed module in the image (a built-in under `nautilus.pipelines`), so it loads by reference in every
container; the same image runs every role, so the classes the factories name are importable wherever the
plan lands.

### A.7 Invariants and contracts preserved

The coordinator stays control-plane only (Contract 1); only the framing changes (an `mp.Queue`'s implicit
pickle becomes an explicit framed message over a socket), and report aggregation at
`coordinator.py:114,130` is identical. The data path never imports `nautilus.cluster`, `transport` never
imports `cluster`, `compile` imports only `api`, and `api` imports nothing else — all unchanged, because
the new code is all under `nautilus.cluster` (firewalled by Contract 1) plus inert parameter additions in
`transport` (a connect timeout and keepalive `setsockopt` calls, no new imports).

### A.8 Physical-host telemetry attribution

`Deployment.node` is the single attribution label every operator instance and the per-worker process row
is tagged with (`worker_main.py:133` sets `node=f"worker-{worker_id}"`; it flows through `execute.py:149,
297,300` → `recorder.py:151,233,287` → the report rows at `serialize.py:79,189` and `report.py:345,471`).
For single-machine multiprocess that label *is* the node — each worker is just a local process. Across
machines it is not: it names the logical worker, never which physical container/Pod the work ran on, so a
multi-node report cannot answer "which machine is hot/slow/OOMing" — the question the telemetry loop
exists for. Collapsing every host to `worker-{id}` is therefore a regression Stage 4 must not ship.

The fix separates logical identity from physical placement. `node` stays `worker-{id}` as the stable
correlation key (placement, the address book, `cross_worker_inbound`, and the existing process-row
assertions all key on it), and a new **`host`** attribute carries the physical identity. The daemon
sources `host` from what it already knows — its `--advertise` value / container hostname, and in k8s the
downward-API Pod name or `POD_IP` — and threads it into `Deployment`; the local-spawn and single-process
paths default `host` to the `node` label, so their reports and tests are byte-for-byte. Adding a `str`
field to `Deployment` (`runtime/connector.py`) is inert on the data path: no new import, and the recorder
change stays within the telemetry data-path layer (it never reaches `telemetry.report`), so no
import-linter contract moves. `host` is an attribution dimension like `node`, not a catalog metric, so it
needs no new catalog entry — but if the catalog does change, regenerate `docs/telemetry-reference.md` with
`nautilus reference` rather than hand-editing it.

*Open choice for 4.3:* a separate `host` field (recommended — keeps `node` stable and queryable
independently) vs. enriching the label to `worker-{id}@{host}` (one field, no model change, but rewrites
the string the distributed tests assert on). Settle when 4.3 is built.

---

## B. Sub-stage breakdown

Each sub-stage is independently shippable and green across pytest / mypy / ruff / black / import-linter,
and keeps `bench-check`'s 2-worker baseline entries green (they run on the `LocalCohort` path).

### 4.0 — Cohort seam (refactor, zero behavior change)

- **Deliverable:** `WorkerCohort` ABC + `LocalCohort` wrapping today's `(procs, events, commands)`;
  `deploy`/`bind_barrier`/the completion loop call only cohort methods.
- **Files:** add `src/nautilus/cluster/cohort.py` (`WorkerCohort`, `LocalCohort`); change
  `src/nautilus/cluster/coordinator.py` (build a `LocalCohort`, call `send`/`next_event`/`reap`);
  `src/nautilus/cluster/rendezvous.py` functions unchanged (`LocalCohort` calls them).
- **Test/demo:** all `test_cluster_*` pass unchanged; `test_cluster_rendezvous.py`'s direct `recv_event`
  calls stay valid. Add `tests/test_cluster_cohort.py`: `LocalCohort.next_event` narrows `watch` and
  raises `WorkerCrashed` on a bad exitcode, through the seam.
- **Docs:** `WorkerCohort` docstring states the three-operation contract and the `watch`-set crash
  invariant; `rendezvous.py` module docstring → "the shared receiver behind the cohort"; `DESIGN.md`
  bootstrap/teardown paragraphs note the loop is cohort-driven.

### 4.1 — Bind-vs-advertise + connect timeout + data-plane keepalive

- **Deliverable:** advertise/bind split; bounded data dial; a data-plane partition becomes a bounded error.
- **Files:** `worker_main.py` (`_run_worker` takes `bind_host, advertise_host`; registers advertise host +
  concrete port); `launcher.py` + `coordinator.py` (thread `advertise_host`, default `== host ==
  127.0.0.1`); `rendezvous.py` (`bind_barrier` rejects `0.0.0.0`/`""`); `transport/connector.py` (connect
  timeout; `getaddrinfo` retry; keepalive `setsockopt` on the dialed socket); `transport/socket_channel.py`
  or `listener.py` (keepalive on the accepted socket); `transport/listener.py` (`address` docstring).
- **Test/demo:** local default `advertise == bind == 127.0.0.1` keeps every test green. Add
  `tests/test_advertise.py`: a `0.0.0.0`/empty advertise is rejected; an advertise host ≠ bind host routes
  a real edge; dialing a black-hole address raises a bounded `TimeoutError` rather than hanging.
- **Docs:** `listener.py` `address` docstring; `DESIGN.md` bootstrap paragraph gains "every registered
  address must be dialable, established by config not by construction," plus the keepalive interval and its
  partition bound; `membership.py` docstring notes the book holds *advertised* addresses; glossary
  "Bind/Advertised address" + "Address book" entries (§F.4).

### 4.2 — Control link + daemon + RemoteCohort + CLI

- **Deliverable:** the full coordinator-dials-daemons path, with hermetic loopback coverage.
- **Files:** add `src/nautilus/cluster/control_link.py` (sync + async framer, `_MAX` guard sized for full
  sink output, `Launch`/`Abort` messages, cloudpickle payload codec, address book as a plain dict);
  `src/nautilus/cluster/daemon.py` (the job-scoped state machine of §A.4, the out-of-band watchdog, a
  `--healthcheck` TCP-probe mode); `src/nautilus/cluster/cohort.py` (`RemoteCohort`: non-blocking sockets,
  per-connection reassembly, selectors multiplex, deregister-on-`Done`, control keepalive, dial
  retry-with-backoff bounded by a new `connect_timeout`); change `coordinator.py` (`deploy` gains `daemons:
  list[tuple[str,int]] | None = None` and `connect_timeout`; `daemons` selects `RemoteCohort`,
  `num_workers` inferred from roster length, connect/`Launch`/await only the first `effective`); `cli.py`
  (`nautilus worker` command; `nautilus run --daemons`); thread `daemons` through `dsl.py:243-245` and
  `bench.py:214,229` via the existing lazy `from nautilus.cluster import deploy`.
- **Test/demo (default suite, no Docker):** `tests/test_cluster_remote.py` launches one or two `nautilus
  worker --listen 127.0.0.1:0` daemons as localhost subprocesses and runs `deploy(daemons=[...])` over
  loopback — the remote analogue of `test_cluster_deploy.py`, asserting `multiset(result) ==
  multiset(serial)` and digest equality. `tests/test_control_link.py`: round-trip every message kind; the
  `_MAX` guard rejects an oversized declared length; a truncated read surfaces `WorkerCrashed`; a
  non-trivial sink result crosses the wire. Add a `roster_len > effective` test: surplus daemons get no
  `Launch`, are not awaited, are not treated as crashed.
- **Docs:** `daemon.py` module docstring (the job-scoped abort state machine, self-kill-on-wedged-abort,
  advertise-vs-bind, by-reference code-availability); `control_link.py` docstring (wire format, `_MAX`
  rationale, cloudpickle = Stage-5 RCE surface); `protocol.py` docstring updated from "these cross an
  `mp.Queue`" to "an `mp.Queue` locally, the `control_link` framed wire remotely"; `DESIGN.md` "Teardown
  is symmetric" gains the control-loss abort + out-of-band self-terminate; glossary gains daemon, worker
  cohort, control link, roster (§F.4); `docs/cli-reference.md` gains a `worker` section and a `--daemons`
  row on `run`; `docs/dsl-reference.md` notes `Stream.run` forwards `daemons`.

### 4.3 — docker-compose harness + physical-host telemetry + integration test + CI

- **Deliverable:** the compose stack, real physical-host attribution in the report, the genuine cross-host
  test, and the first CI workflows.
- **Physical-host attribution (see §A.8):** the report must distinguish *which container/Pod* an instance
  ran on, not just its logical worker id — telemetry is the development loop here, so a multi-node run that
  collapses every host to `worker-{id}` is a regression. Add a `host` attribute to `Deployment`
  (`runtime/connector.py`), sourced by the daemon from its identity (`--advertise` / container hostname;
  in k8s the downward-API Pod name/IP), carried through `execute.py` → `recorder.py` → the process and
  operator rows (`report/serialize.py`, `report/report.py`). `node` stays `worker-{id}` as the correlation
  key, so existing reports/tests read unchanged; `host` defaults to the node label off the remote path.
- **Files:** add `Dockerfile`; `docker-compose.yml`; `tests/integration/test_compose.py` (marked
  `@pytest.mark.docker`, skipped by default); `.github/workflows/ci.yml` (base gates: `pytest -m "not
  docker"`, mypy, ruff, black, import-linter, `bench-check`); `.github/workflows/compose.yml` (the Docker
  job). Change `runtime/connector.py` (`Deployment.host`), `cluster/worker_main.py` + `cluster/daemon.py`
  (thread the host identity in), `runtime/execute.py`, `telemetry/recorder.py`,
  `telemetry/report/{serialize,report}.py` (carry/emit `host`). `pyproject.toml` registers the `docker`
  marker and adds `addopts = "-m 'not docker'"`.
- **Test/demo:** see §D. Plus: the compose test asserts the report's `host` values are the *distinct*
  container names (not all `worker-0`/`worker-1`), proving real host attribution; a single-process /
  local-spawn report is unchanged.
- **Docs:** `README.md` "running multi-node"; `IMPLEMENTATION_PLAN.md` Stage 4 → **Done**; `DESIGN.md`
  Status + the telemetry section's node-attribution note (node = logical worker, host = physical
  container/Pod); the layer-table `nautilus.transport` comment → "loopback and cross-host"; glossary
  "node vs host"; regenerate `docs/telemetry-reference.md` via `nautilus reference` **only** if a catalog
  entry changes (never hand-edit it — `host` is an attribution dimension like `node`, not a new metric, so
  likely no catalog change).

---

## C. docker-compose harness

**Dockerfile.** One image, `python:3.12-slim` base (dev uses 3.12; `requires-python >= 3.11`), `pip
install .` (non-editable; pyarrow ships a manylinux wheel, no build). The *same* image runs every role,
because the daemon `cloudpickle.loads` the plan and by-reference operators must be importable wherever it
lands (§A.6). Default entrypoint `nautilus`.

**docker-compose.yml.** N worker services `worker-1..worker-N` on one user-defined bridge network, each:

```yaml
command: worker --listen 0.0.0.0:9000 --advertise worker-i --bind 0.0.0.0
healthcheck:
  test: ["CMD", "nautilus", "worker", "--healthcheck", "127.0.0.1:9000"]
  interval: 1s
  retries: 30
restart: unless-stopped     # revives a daemon after a wedged-abort self-kill
```

A short-lived `coordinator` service (or the host test harness) runs `nautilus run <pipeline> --parallelism
P --daemons worker-1:9000,worker-2:9000,…` with `depends_on: { worker-1: { condition: service_healthy },
… }`. The coordinator is **never** a worker — it is control-plane only (Contract 1), so co-locating it
would blur the plane boundary.

**Healthcheck.** `depends_on: condition: service_healthy` needs a real probe, and the daemon exposes only a
raw TCP control port (no HTTP). `nautilus worker --healthcheck HOST:PORT` opens a TCP connection to the
control port and exits 0/1 — it proves the accept loop is listening (and DNS is warm), not merely that the
process is up. Same image, no extra dependency.

**Dial-race defense.** Even with `service_healthy`, a host-run `nautilus run` harness has no `depends_on`,
and a daemon can be milliseconds slow to bind. `RemoteCohort` retries the control dial with backoff until
`connect_timeout` before failing, with a clear "daemon at host:port never accepted within Ns" message.

**Compose hygiene (a Stage 4 correctness requirement, not a Stage 5 feature).** The compose file must
**not** `ports:`-publish the worker control or data ports to the host. Binding `0.0.0.0` with no auth is
contained only while the ports stay on the internal bridge.

**Env-var config.** `--listen`/`--advertise`/`--bind` are flags; `--daemons` also reads
`$NAUTILUS_DAEMONS`. No silent `gethostname()` default for advertise (§A.5).

---

## D. Test & CI strategy

**The assertion.** Force a non-co-located shuffle by setting `--parallelism` greater than the worker count
so a keyed shuffle crosses containers, then assert the multi-container result matches a single-process run
of the same pipeline: `multiset(distributed) == multiset(single_process)` (a shuffle reorders batches, so
equality is multiset, matching `test_cluster_scale.py:87,105`) **and** `structural_digest` equality
(`test_cluster_scale.py:122`). This exercises a real TCP shuffle edge opened by service DNS, carrying
Arrow-IPC over the unchanged `SocketChannel`/handshake path. To prove a keyed op actually ran on multiple
nodes, assert the report's process rows are `{"worker-0", …, "worker-{N-1}"}` (the `Deployment.node` set,
`test_cluster_scale.py:131,174`) **and** that the new `host` attribute (§A.8) holds the *distinct*
container names — proving the run genuinely spanned separate hosts, not just logical worker ids.

**Harness lifecycle.** Because daemons stay up, the coordinator's run completing does not stop them, so the
test owns the stack: bring it up (`docker compose up -d` via subprocess, the no-new-dependency choice),
wait for `service_healthy`, run the coordinator, assert, then `docker compose down` in a `finally`.

**Docker gating.** `tests/integration/test_compose.py` is marked `@pytest.mark.docker`; `pyproject.toml`
registers the marker and sets `addopts = "-m 'not docker'"`, so the unit suite stays hermetic and fast and
the marked test is opt-in (`pytest -m docker`). The base CI workflow runs `pytest -m "not docker"`, mypy,
ruff, black, import-linter, and `bench-check`; a separate compose workflow runs the marked test, so a
Docker-less contributor's local run is unaffected. These are the repo's first CI definitions (no
`.github/workflows`, `Makefile`, or `noxfile` exists today); 4.2's hermetic `test_cluster_remote.py` is
what makes the multi-node *control* path green in the default suite without Docker.

---

## E. Settled (not forks)

- `deploy` stays synchronous — `RemoteCohort` multiplexes its control sockets with a blocking `selectors`
  loop, not an asyncio loop.
- Membership is a static roster the coordinator dials, not self-registering daemons.
- The coordinator is never a worker.
- Daemon lifecycle is stays-up; a wedged abort self-terminates and compose `restart` revives it.
- Control payloads are cloudpickle/pickle now; the RCE surface is owned entirely by Stage 5.

---

## F. Exact doc edits (land with the sub-stage that introduces the behavior)

`IMPLEMENTATION_PLAN.md` Stage 4/5 prose is **already written** (the sub-stage list). The edits below are
the `DESIGN.md` / glossary / CLI-reference deltas that land *with each sub-stage's implementation* (not
now — `DESIGN.md` describes what is built).

### F.1 DESIGN.md — Deployment section (lands across 4.0–4.2)

Insert after the **Two-phase bootstrap** paragraph:

> **Workers are started one of two ways, behind one seam.** A `WorkerCohort` abstracts the three
> machine-specific control primitives — start a worker, move a control message, detect a crash — so
> `deploy`'s body is identical local or remote. `LocalCohort` spawns worker processes and moves
> `Register`/`Done`/`Failed` and the address book over `multiprocessing` queues, detecting a crash from a
> child's exit code — the single-machine path. `RemoteCohort` instead dials a roster of long-lived
> `nautilus worker` daemons (one per container, addressed by service DNS), carries the same messages down
> one framed TCP control connection per worker (`cluster.control_link`), and detects a crash from that
> connection closing before a worker's `Done`. The roster is fixed membership: the coordinator dials the
> first `min(num_workers, max-parallelism)` daemons, assigns `worker_id = roster index`, and leaves any
> surplus daemon idle.
>
> **A worker advertises where peers should dial it, which need not be where it binds.** A daemon binds its
> listener on all interfaces (`0.0.0.0`) to accept on the container's bridge interface, but registers a
> separate routable advertised address (its compose service name), because `getsockname()` on a `0.0.0.0`
> bind returns `0.0.0.0`, which no peer can dial. Deadlock-freedom now carries the precondition that every
> advertised address routes to its own listener — enforced by configuration, the rejection of a `0.0.0.0`
> advertise, and a connect timeout that turns a bad address from a hang into a bounded error, not by
> construction. Control and data sockets set TCP keepalive (idle 10 s, interval 5 s, 3 probes) so a silent
> partition during a job — which sends no FIN — surfaces as a bounded connection error rather than
> indefinite silence, since the completion wait is otherwise unbounded.

Extend the **Teardown is symmetric** paragraph with:

> Across machines the coordinator cannot SIGKILL a non-child worker, so the daemon enforces no-orphan
> itself: its control connection is per-job, and a control drop *before* this job's `Done` cancels
> `execute()` and runs the failure-path teardown, returning the daemon to idle. A normal job end (control
> closed *after* `Done`) leaves the daemon up for the next job. Only a wedged abort — one asyncio
> cancellation cannot unwind because the loop is blocked — trips an out-of-band watchdog that hard-exits
> the daemon's own process, the network replacement for the local SIGKILL.

Status paragraph: change "Multi-node validation (Stage 4) is designed but not built." → "Multi-node runs
across separate containers addressed by service DNS (Stage 4); securing it on an untrusted network is
Stage 5." Layer-table `nautilus.transport` comment: "loopback now / cross-host (Stage 1/4)" → "loopback
and cross-host (Stage 1/4)".

### F.2 docs/glossary.md — rewrite Node/Address-book, add four

Rewrite **Node address** and **Address book**:

> - **Bind address** — The `(host, port)` a worker's `EdgeListener` actually binds (`getsockname()`).
>   Binding `0.0.0.0` accepts on every interface but is not itself dialable.
> - **Advertised address** — The routable `(host, port)` a worker registers for peers to dial; the *same*
>   concrete port as the bind but a host that resolves from other containers (the compose service name).
>   It differs from the bind address whenever a worker binds all interfaces.
> - **Address book** — The `AddressBook` (`cluster.membership`) mapping each worker id to its *advertised*
>   address, built once after every worker has bound and broadcast unchanged. `edge_resolver` turns it into
>   the resolver a `SocketConnector` dials: the address for an edge is the advertised listener of the
>   worker hosting the edge's destination instance.

Add:

> - **Worker daemon** — A long-lived `nautilus worker` process (one per container) that binds a control
>   port, waits, and runs one job per `Launch` from a coordinator on a fresh event loop, then returns to
>   idle. The remote replacement for a spawned worker process.
> - **Worker cohort** — The `WorkerCohort` seam abstracting how the coordinator starts workers, moves
>   control messages, and detects a crash. `LocalCohort` uses spawn + `mp.Queue` + exit codes;
>   `RemoteCohort` dials daemons over a framed TCP control connection.
> - **Control link** — `cluster.control_link`: the framed TCP wire (`[magic][kind][len][payload]`)
>   carrying `Launch`/`Abort` down and `Register`/`Done`/`Failed` up between a coordinator and a daemon,
>   the remote replacement for the control `mp.Queue`s.
> - **Roster** — The fixed list of daemon control addresses a coordinator dials, from
>   `--daemons`/`$NAUTILUS_DAEMONS`. The coordinator assigns `worker_id = roster index`; roster length is
>   an upper bound on workers, capped like `--workers`.

### F.3 docs/cli-reference.md — `--daemons` row on `run`, a `worker` section

Add to the `run` options table:

> | `--daemons` | none | `host:port,…` of worker daemons to dial (remote mode); unset = spawn locally |

Add a `worker` command section (options `--listen 0.0.0.0:9000`, `--advertise HOST` (required in remote),
`--bind 0.0.0.0`, `--healthcheck HOST:PORT`). Add to `docs/dsl-reference.md`'s `run()` entry:
"`daemons=[(host,port),…]` runs across worker daemons instead of spawning locally."
