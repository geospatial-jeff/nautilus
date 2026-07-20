# Nautilus glossary

The vocabulary and data model of nautilus. These are mostly standard
stream-processing terms (as used by systems like Apache Flink); where nautilus has a specific class
for a term, it is named in parentheses, e.g. (`Batch`).

A nautilus job is a directed graph: data enters at a **source**, passes through a chain of
**operators**, and leaves at a **sink**. Records travel as **frames** over bounded **channels**.
The same machinery handles bounded data (a finite stream that ends) and unbounded data (a stream
that runs indefinitely).

Items marked **(planned)** or **(reserved)** are designed for but not yet built; they are included
so the names you will meet in `DESIGN.md` and the source are explained.

---

## The job graph

*Source: `nautilus.core.operator`, `nautilus.runtime.partition`.*

- **Dataflow graph (topology)** — The directed graph of operators connected by edges that specifies
  the computation. It is fixed before the run starts and does not change while running.
- **Operator** — A unit of computation in the graph. It reads records from its inputs, may keep
  state, and emits records to its output. Nautilus has three operator shapes (below).
- **Source (source operator)** — An operator with no inputs that produces a stream. It generates the
  data records and the end-of-stream marker (`SourceOperator`; e.g. `InMemorySource`). A source may
  `await` between batches, which is how unbounded inputs work.
- **Transform (one-input operator)** — An operator with exactly one input stream and one output
  stream (`OneInputOperator`; e.g. `MapBatch`, `FilterRows`, `Tokenize`, `KeyedCount`).
- **Two-input operator** — An operator that combines two input streams, such as a join, with one stream
  on each **input port** (port 0 = left, port 1 = right). It ends only after *both* inputs reach EOS
  (`TwoInputOperator`).
- **Input port** — Which input of an operator an edge feeds: port 0 for a one-input transform (and a
  join's left side), port 1 for a join's right side. The port is what lets a two-input operator tell its
  two inbound edges apart.
- **Equi-join (hash join)** — The built-in two-input operator (`HashJoin`): emits a row for every left and
  right row whose join keys are equal. The output is the left row's columns plus the right's non-key
  columns; both inputs are co-partitioned on the join key, so a key's rows meet on one instance. `how`
  selects whether unmatched rows are also kept — `"inner"` (default) drops them, `"left"`/`"right"`/
  `"outer"` keep one or both sides with the other side null. (How it buffers, matches, and flushes the
  unmatched rows — and why an outer join is parallelism-1 only — is on `HashJoin`.)
- **Sink** — The end of the graph: it consumes the final stream. Stage 0's sink collects the output
  batches into a list (it appears as `CollectSink` in the topology and telemetry). An **async sink**
  (`AsyncSink`) is an authored terminal that instead *writes* each batch to an external store.
- **Async stage** — An operator that does its I/O by *awaiting* it, so the pipeline can enrich a record
  from an external lookup (`AsyncOneInputOperator`) or write to an external store (`AsyncSink`) inside the
  streaming model. It is split into an awaiting half — `fetch`/`write`, run as bounded concurrent tasks so
  I/O overlaps, handed no keyed state — and a synchronous half — a transform's `integrate`, run on the
  actor task, the only place that touches state or emits. This keeps keyed state single-writer while I/O
  overlaps (`DESIGN.md` mechanism 8).
- **In-flight bound (`max_in_flight`)** — How many `fetch`/`write` tasks an async stage may run at once.
  It doubles as the stage's backpressure: once reached, the actor stops reading, so a slow external store
  stalls upstream with bounded memory rather than buffering without limit.
- **Ordered vs unordered emission** — How an async transform orders its output. **Ordered** (the default)
  integrates and emits strictly in input order — reproducible emission, keyed-state fold order, and
  structural digest — while later fetches still overlap behind the in-order frontier. **Unordered**
  (`ordered=False`) emits each result as its fetch finishes (completion order, lower latency); it is
  stateless-only, because a keyed stage's counts would then depend on fetch timing. EOS is a hard barrier
  either way — it never overtakes the data read before it.
- **Edge** — A directed connection from one operator's output to a downstream operator's **input port**.
  Every edge carries both data and control frames and has a partitioner that decides how data is routed
  to the downstream instances.
- **Channel** — The concrete one-directional, in-order (FIFO) transport behind an edge between two
  instances. It has a fixed capacity; a full channel blocks the sender (see **backpressure**).
  In-process it is an `asyncio.Queue` (`InProcChannel`); across processes it is a socket
  (`SocketChannel`), which uses credit-based flow control (below).
- **Instance (subtask)** — One parallel copy of an operator. An operator with a given parallelism runs
  as that many independent instances, each handling a subset of the data, numbered from zero by
  `subtask_index`.
- **Parallelism** — The number of instances an operator is split into.
- **Partitioner** — A pure function on the sending side that decides which downstream instance each
  row of a batch goes to. The kinds the runtime builds are `Forward` (co-located: sender `i` to the
  same-index downstream instance, or to instance 0 for a single owner), `KeyGroupPartitioner` (the keyed
  shuffle — see below), and `RoundRobin` (rotates whole batches for keyless rebalancing). Control frames
  skip the partitioner and are always broadcast.
- **Keyed shuffle** — Routing that sends every row with a given key to the same downstream instance,
  so a key's rows and state are never split across instances. The hash (`stable_bucket`) is process-,
  seed-, and platform-stable, so the same key maps to the same instance in any process. The runtime
  uses `KeyGroupPartitioner`, which adds a `group → instance` indirection over the hash; `HashPartitioner`
  (which hashes each key straight to an instance, modulo the parallelism) is the form it generalizes,
  kept only as the test oracle for when the key-group count equals the parallelism.
- **Key group** — One of a fixed number of buckets a key hashes to (`hash(key)` modulo the key-group
  count) before being mapped to an instance through a static `group → instance` table the plan carries
  (`KeyGroupPartitioner`). A key never moves between groups, so rescaling the instance count is a table
  swap, not a re-hash of state.
- **Max parallelism (the key-group count)** — The fixed number of key groups a keyed edge hashes into,
  chosen once for the job (the `key_groups` argument to `compile_graph` / `run_plan`, or the `--key-groups`
  CLI flag, defaulting to the stage parallelism). Because each group maps to one instance,
  it is the most instances the edge can be rescaled to without re-hashing, so it must be at least the
  stage parallelism.
- **Stage parallelism** — The number of instances of the operator an edge feeds (its parallelism). The
  `group → instance` table maps the key groups onto these instances.
- **Key range** — The set of keys one instance owns: the union of the key groups the `group → instance`
  table assigns to it. When the key-group count equals the parallelism (the identity table) this is the
  direct-hash range — the keys whose hash, modulo the parallelism, equals the instance's own index; with
  more key groups than instances, an instance owns one or more groups. Each key belongs to exactly one
  instance.

## From graph to run (compile)

*Source: `nautilus.api`, `nautilus.compile`.* How a described job becomes a runnable artifact.

- **Stream** — The fluent builder (`nautilus.dsl.Stream`): an immutable handle on a dataflow under
  construction. Each *combinator* (`map`, `tokenize`, `count_by`, `join`, `apply`, …) returns a new
  Stream that adds one operator; a *terminal* (`run`, `run_async`, `collect`) executes it. It produces a
  **logical graph** and nothing more — the readable, join-capable way to build one. Start one with
  `source(...)`.
- **Combinator / terminal** — A combinator is a `Stream` method that adds an operator and returns a new
  stream; a terminal (`run`/`collect`) is the method that compiles and runs the stream. `run(workers=…)`
  deploys the same graph across that many processes.
- **Logical graph** — The job as you describe it: operators with their parallelism and keying wired into
  a dataflow, and nothing physical — no instances, no channels, no operator ids (`LogicalGraph`). With no
  explicit edges it is the linear shape (a source then a chain, built with `linear_graph`); with explicit
  edges it is any DAG — the shape a join needs (two sources into one two-input vertex). It is the input to
  the compiler, and what the fluent `Stream` DSL produces.
- **Vertex** — One operator in a logical graph: the factory that builds it, its kind (source, one-input,
  or two-input), its parallelism, and (on a one-input vertex, for the linear shape) its key columns
  (`LogicalVertex`).
- **Logical edge** — A directed edge from one vertex to a downstream vertex's input port, carrying the
  columns that edge is co-partitioned on (`LogicalEdge`). Keying lives on the edge, not the vertex, so a
  join's two inputs can shuffle on differently-named columns yet land equal keys together. A linear graph
  carries no edges — the compiler reads its positional adjacency.
- **Compile (lowering)** — The one-time step that turns a logical graph into a physical plan
  (`compile_graph`): it orders the operators topologically and names them by position (`source`, `op0`…,
  `sink`), chooses a partitioner spec for each edge, and synthesizes the collecting sink. It runs once,
  before the data path starts — never per record.
- **Physical plan** — The runnable, serializable result of compiling: the operators with their
  parallelism, the edges between them, and a partitioner spec per edge (`PhysicalPlan`). It is inert
  data plus the operator factories, so it can be cloudpickled to a worker that never saw the original
  graph — it is the unit of serialization for distributing a job.
- **Partitioner spec** — A stateless description of how one edge routes, selected by the compiler
  (`ForwardSpec`, `RoundRobinSpec`, `KeyGroupSpec` — the last carrying the `group → instance` table).
  The runtime builds a fresh `Partitioner` from it when it wires each output. The spec deliberately
  carries no live state — a `RoundRobin`'s rotation cursor, for instance — so it is safe to serialize
  and never shared between workers.

## Frames — what moves on a channel

*Source: `nautilus.core.records`.* Every frame is either a **data frame** or a **control frame**;
the set of frame types is fixed.

- **Frame** — The unit that travels on a channel (`Frame`).
- **Batch (data frame)** — The only data frame. It carries one Arrow record batch — a chunk of rows
  (`Batch`). Data frames are routed by the partitioner.
- **RecordBatch** — Apache Arrow's columnar container: a set of equal-length, typed columns. Nautilus
  moves data as Arrow record batches end to end ("Arrow-first"), so data is columnar and cheap to
  pass between processes.
- **Micro-batch** — Carrying many rows per frame instead of one row at a time, to reduce per-record
  overhead. One `Batch` is one micro-batch.
- **Tensor column** — An Arrow `fixed_shape_tensor` column whose rows are fixed-shape multidimensional
  arrays (for example an `H×W×C` image), stored row-major as a `fixed_size_list`; the shape is type
  metadata and
  the column length is the batch dimension. `nautilus.tensors` builds these from numpy and reads them
  back (`tensor_array` / `embedding_array` / `to_numpy`).
- **Embedding** — A 1-D float vector per row, held as a tensor column of shape `(dim,)`. Its
  `.storage` is `fixed_size_list<float32, dim>` — the layout vector-search indexes operate on.
- **Control frame** — A frame that carries a coordination signal rather than data. Control frames are
  broadcast to every downstream instance.
- **EOS (end of stream)** — The terminal control frame (`EOS`). An operator forwards EOS downstream
  only after it has received EOS on *every* input; reaching the last input runs the operator's
  end-of-stream flush (`on_eos`) first, so a keyed aggregation emits its totals. The job is done once
  all sinks have seen EOS.
- **Barrier** — A control frame for checkpoint-based exactly-once processing. **(reserved — the type
  exists so adding it later is not a breaking change to the wire format.)**

## Time

*Source: `nautilus.core.time`.*

- **Processing time** — Wall-clock time on the machine running the operator, read from a `Clock`.
  Used for timing and telemetry. Injectable so tests are deterministic (`TestClock`).

## State

*Source: `nautilus.state`.*

- **Keyed state** — Per-key memory an operator keeps across records (for example, a running count per
  word). Addressed by `(operator_id, state name, key, namespace)`.
- **Key** — The value, or tuple of values, that partitions the stream (for example, the word in
  word-count). Rows with the same key are handled by the same instance and share state.
- **Namespace** — A sub-division of one key's state. It remains a `StateBackend` capability but no
  built-in operator sets one today.
- **State backend** — The pluggable store behind keyed state (`StateBackend`). The default is an
  in-memory dictionary (`InMemoryStateBackend`); the interface includes `snapshot`/`restore` so a
  persistent or checkpointing backend can be added without changing operators.
- **State handles** — Typed accessors for one piece of keyed state: `ValueState` (a single value) and
  `ReducingState` (a value folded by a reducer as items are added). To enumerate or clear all of an
  operator's keyed state at end of stream, use `OperatorContext.entries` / `clear_state`.

## Execution and flow control

*Source: `nautilus.runtime` (the data path), `nautilus.driver` (the boundary that runs it).*

- **Actor** — The loop that drives one operator instance: it pulls frames from the inputs, calls the
  operator's `process` / `on_eos`, and pushes results to the outputs. One actor per instance,
  single-threaded (one asyncio task).
- **Mailbox** — The fan-in on an instance that merges its several input channels into one ordered
  sequence of `(input_index, frame)`, preserving each channel's order (`Mailbox`).
- **Backpressure** — Flow control in which a slow consumer slows its producers. Because channels are
  bounded, a full channel suspends the sender until the consumer drains it; this propagates upstream
  all the way to the source.
- **Credit-based flow control** — How backpressure works across a process boundary, where there is no
  shared queue. The consumer grants the producer a fixed number of **credits** — the **window**, equal
  to the channel capacity. The producer may send a data frame only while it holds a credit, and the
  consumer returns one credit for each data frame it receives, so the number of data frames in flight
  never exceeds the window. Control frames (EOS) are sent without credit, so a full data
  window never delays them (`SocketChannel` in `nautilus.transport`).
- **Collector** — The in-memory buffer an operator emits into during a single `process` /
  `on_eos` call (`Collector`). The actor drains it and performs the (awaiting) sends
  afterward, so operator code stays synchronous and each step is a self-contained critical section.
- **Operator context** — The object handed to an operator at `open` time holding its dependencies: its
  id, subtask index and count, state backend, clock, and a metrics recorder (`OperatorContext`).
- **Runner (driver)** — The `nautilus.driver` component that executes a job single-process — the
  boundary that compiles, runs, and builds the report. `run_local_chain` runs a `(source, transforms)`
  chain in-memory (`run()` is its synchronous one-line wrapper; this is what the CLI's single-process path
  uses, including at `--parallelism > 1 --workers 1`); `run_plan` compiles a `LogicalGraph` and runs it —
  the shared engine beneath `run_local_chain` and the direct target of the `Stream` DSL's `.run()`
  terminal; and `run_compiled` runs an already-compiled `PhysicalPlan` (e.g. a cloudpickle round-trip).
  All go through the same compiled executor.
- **RunResult** — What a run returns: the final output batches plus the run's telemetry report
  (`nautilus.driver.RunResult`; `result.telemetry`).
- **Worker process** — One spawned OS process running a slice of the plan, with its own event loop and
  no shared memory. `deploy` runs several of them on a single machine; routing within and between them is local
  to each sender.
- **Worker daemon** — A long-lived `nautilus worker` process (one per container) that binds a control
  port, waits, and runs one job per coordinator connection on a fresh event loop, then returns to idle.
  The multi-node replacement for a spawned worker process — the coordinator *dials* it instead of spawning
  it.
- **Coordinator** — The control plane behind `deploy`: it compiles the graph, computes placement, starts
  the workers (or dials the daemons), drives the bootstrap, and aggregates one report at the job boundary.
  It reads no data channel and grants no credit, so there is still no central scheduler on the data path.
- **Worker cohort** — The `WorkerCohort` seam abstracting how the coordinator reaches its workers: hand
  one a control message, take the next event with crash detection, reap them. `LocalCohort` spawns
  processes and uses `multiprocessing` queues and exit codes; `RemoteCohort` dials daemons over a framed
  TCP control connection and reads a crash from the connection closing before a worker's `Done`.
- **Roster** — The fixed list of daemon control addresses a coordinator dials (`--daemons` /
  `$NAUTILUS_DAEMONS`). The coordinator assigns `worker_id = roster index`; the roster length is the
  worker count, capped at the plan's max parallelism (a surplus daemon is left idle).
- **Placement** — The map from each operator instance to the worker that hosts it (`cluster.placement`):
  per-operator round-robin over the workers, so same-index subtasks co-locate and only a real shuffle
  crosses workers.
- **Compile and deploy** — Lowering the graph to a physical plan (`compile`) and running it across
  workers (`deploy`), driven from the CLI by `nautilus run --workers <count> --parallelism <count>`.

## Cross-worker connections

*Source: `nautilus.transport`, `nautilus.cluster`.* How an edge that crosses workers is established.

- **Bind address** — The `(host, port)` a worker's `EdgeListener` actually binds (`getsockname()`).
  Binding `0.0.0.0` accepts on every interface but is not itself dialable.
- **Advertised address** — The routable `(host, port)` a worker registers for peers to dial: the *same*
  concrete bound port, but a host that resolves from other containers (its service/DNS name). It differs
  from the bind address whenever a worker binds all interfaces; on a single-machine run the two are equal
  (both loopback).
- **Edge handshake** — A one-shot preamble a producer writes right after connecting, naming the
  `ChannelId` of the edge it is opening, before any frame. The accepting `EdgeListener` reads it to route
  the socket to the right consumer, so connections may arrive in any order without the wires crossing.
- **ChannelId** — The identifier of one directed instance-to-instance edge: the source operator id and
  subtask, and the destination operator id and subtask. It is both the key the in-process connector maps
  to a queue and the value a socket announces in its handshake, so an edge is named the same way whatever
  the transport.
- **Address book** — The `AddressBook` (`cluster.membership`) mapping each worker to its *advertised*
  `(host, port)`, built once after every worker binds. The socket connector takes a resolver over it (the
  address a producer dials is the advertised listener of the worker hosting the edge's destination), which
  is why `transport` never imports `cluster`.
- **Control link** — `cluster.control_link`: the framed TCP wire (`[magic][length][cloudpickle payload]`)
  carrying `Launch`/`Abort` down and `Register`/`Done`/`Failed` up between a coordinator and a daemon — the
  multi-node replacement for the control `multiprocessing` queues.
- **Rendezvous** — The two-phase startup that makes connection setup deadlock-free: every worker binds
  its listener (so all destinations exist) and registers before the coordinator broadcasts the address
  book that lets anyone dial. A connection arriving before its consumer accepts is parked, so no global
  "go" barrier is needed.

## Processing guarantees

- **At-least-once** — Every record is processed at least once; after a failure and recovery, some
  records may be reprocessed (duplicates possible). This is nautilus's current target.
- **Fail-fast** — On an unhandled error the whole job stops and the error is surfaced, rather than
  being silently swallowed or partially retried.
- **Exactly-once** — Each record affects state exactly once even across failures, using aligned
  checkpoints (see **Barrier**). **(deferred.)**
- **Checkpoint** — A consistent snapshot of all operator state, taken so a job can recover after a
  failure. **(deferred.)**

## Telemetry

Every run emits telemetry — metrics and events about what it did. The terms below are summarized here;
the full catalog of metrics is in [`telemetry-reference.md`](telemetry-reference.md).

- **Recorder** — The single object instrumentation writes metrics and events to (`Recorder`).
  One writer per actor, and it is a no-op (zero cost) when telemetry is off.
- **Metric** — A measured number: a counter (running total), gauge (latest value), or histogram
  (bucketed distribution). Each metric is described in the catalog with its unit, labels, and
  meaning.
- **Tier** — How much telemetry is collected: `OFF`, `COUNTERS`, `COUNTERS_PLUS_EVENTS`, or `FULL`
  (the byte-accounting metrics that require walking Arrow buffers are `FULL`-only). The CLI names
  these `off`/`counters`/`events`/`full`.
- **RunReport** — The structured telemetry a run produces, with a JSON form and a markdown digest
  meant to be read by a coding agent (`result.telemetry`).
