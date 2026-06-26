# Nautilus glossary

The vocabulary and data model of nautilus, in plain language. These are mostly standard
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
  data records, the watermarks, and the end-of-stream marker (`SourceOperator`; e.g.
  `InMemorySource`). A source may `await` between batches, which is how unbounded inputs work.
- **Transform (one-input operator)** — An operator with exactly one input stream and one output
  stream (`OneInputOperator`; e.g. `MapBatch`, `FilterRows`, `Tokenize`, `KeyedCount`,
  `KeyedTumblingSum`).
- **Two-input operator** — An operator that combines two input streams, such as a join. Its
  watermark is the minimum of the two inputs. **(reserved — `TwoInputOperator`.)**
- **Sink** — The end of the graph: it consumes the final stream. Stage 0's sink collects the output
  batches into a list (it appears as `CollectSink` in the topology and telemetry).
- **Edge** — A directed connection from one operator to the next. Every edge carries both data and
  control frames and has a partitioner that decides how data is routed to the downstream instances.
- **Channel** — The concrete one-directional, in-order (FIFO) transport behind an edge between two
  instances. It has a fixed capacity; a full channel blocks the sender (see **backpressure**).
  In-process it is an `asyncio.Queue` (`InProcChannel`); across processes it is a socket
  (`SocketChannel`), which uses credit-based flow control (below).
- **Instance (subtask)** — One parallel copy of an operator. An operator with parallelism *N* runs
  as *N* independent instances, each handling a subset of the data, numbered by `subtask_index`
  (0…*N*−1).
- **Parallelism** — The number of instances an operator is split into.
- **Partitioner** — A pure function on the sending side that decides which downstream instance each
  row of a batch goes to. The kinds are `Forward` (1:1), `Broadcast` (every instance gets a copy),
  `HashPartitioner` (the keyed shuffle: routes each row by `hash(key) mod Q`), and `RoundRobin`
  (rotates whole batches for keyless rebalancing). Control frames skip the partitioner and are
  always broadcast.
- **Keyed shuffle** — The `HashPartitioner` routing that sends every row with a given key to the same
  downstream instance, so a key's rows and state are never split across instances. The hash
  (`stable_bucket`) is process-, seed-, and platform-stable, so the same key maps to the same instance
  in any process.
- **Key range** — The set of keys one instance owns under the keyed shuffle: instance *i* of a stage
  with parallelism *Q* handles every key *k* where `hash(k) mod Q == i`. Each key belongs to exactly
  one instance.

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
- **Tensor column** — An Arrow `fixed_shape_tensor` column whose rows are fixed-shape N-D arrays (for
  example an `H×W×C` image), stored row-major as a `fixed_size_list`; the shape is type metadata and
  the column length is the batch dimension. `nautilus.tensors` builds these from numpy and reads them
  back (`tensor_array` / `embedding_array` / `to_numpy`).
- **Embedding** — A 1-D float vector per row, held as a tensor column of shape `(dim,)`. Its
  `.storage` is `fixed_size_list<float32, dim>` — the layout vector-search indexes operate on.
- **Control frame** — A frame that carries a coordination signal rather than data. Control frames are
  broadcast to every downstream instance.
- **Watermark** — A control frame carrying an event-time value `t`, meaning "no later record on this
  channel will have an event time below `t`" (`Watermark`). Watermarks only move forward. They are
  how the system knows event time has advanced enough to close windows.
- **EOS (end of stream)** — The terminal control frame (`EOS`). An operator forwards EOS downstream
  only after it has received EOS on *every* input. The job is done once all sinks have seen EOS.
- **StatusIdle / StatusActive** — Control frames marking an input as temporarily silent (`StatusIdle`)
  or speaking again (`StatusActive`). An idle input is left out of the watermark minimum so a quiet
  partition does not stall event-time progress.
- **Barrier** — A control frame for checkpoint-based exactly-once processing. **(reserved — the type
  exists so adding it later is not a breaking change to the wire format.)**

## Event time and watermarks

*Source: `nautilus.core.time`.*

- **Event time** — The time an event actually occurred, read from the data itself (a timestamp
  column). Represented as an integer number of microseconds since the Unix epoch. Windows and
  watermarks are defined over event time.
- **Processing time** — Wall-clock time on the machine running the operator, read from a `Clock`.
  Used for timing and telemetry, not for windowing. Injectable so tests are deterministic
  (`TestClock`).
- **Timestamp assigner** — The source-side component that reads each row's event time from a batch
  (`TimestampAssigner`; e.g. `ColumnTimestampAssigner` reads a named column).
- **Watermark strategy** — The source-side rule that turns the largest event time seen so far into
  the watermark to emit (`WatermarkStrategy`): `MonotonicTimestamps` for in-order data, or
  `BoundedOutOfOrder` which subtracts an allowed lateness.
- **Watermark combination** — On an operator with several inputs, the operator's watermark is the
  **minimum of its inputs' watermarks** (excluding idle inputs), and it never moves backward
  (`WatermarkTracker`).
- **Idle input** — An input currently marked silent (see `StatusIdle`); it is excluded from the
  watermark minimum until it becomes active again.

## State and windows

*Source: `nautilus.state`, `nautilus.windows`.*

- **Keyed state** — Per-key memory an operator keeps across records (for example, a running count per
  word). Addressed by `(operator_id, state name, key, namespace)`.
- **Key** — The value, or tuple of values, that partitions the stream (for example, the word in
  word-count). Rows with the same key are handled by the same instance and share state.
- **Namespace** — A sub-division of one key's state, used mainly to hold a separate entry per window
  (for example, the running sum for key `sensor-7` in the 10:00–10:05 window).
- **State backend** — The pluggable store behind keyed state (`StateBackend`). The default is an
  in-memory dictionary (`InMemoryStateBackend`); the interface includes `snapshot`/`restore` so a
  persistent or checkpointing backend can be added without changing operators.
- **State handles** — Typed accessors for one piece of keyed state: `ValueState` (a single value),
  `ReducingState` (a value folded by a reducer as items are added), `ListState` (an appendable list),
  and `MapState` (a dictionary).
- **Window** — A finite slice of the stream, defined over event time, that a result is computed over.
  A `TimeWindow` is a half-open interval `[start, end)`.
- **Tumbling window** — Fixed-size, non-overlapping, back-to-back windows, e.g. every 5 minutes
  (`TumblingEventTimeWindows`).
- **Window assigner** — Maps a record's event time to the window(s) it belongs to (`WindowAssigner`).
- **Trigger** — The rule for when a window's result is emitted. In Stage 0 this is implicit: a window
  fires when the operator watermark passes its end.

## Execution and flow control

*Source: `nautilus.runtime`.*

- **Actor** — The loop that drives one operator instance: it pulls frames from the inputs, calls the
  operator's `process` / `on_watermark`, and pushes results to the outputs. One actor per instance,
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
  never exceeds the window. Control frames (watermark, EOS) are sent without credit, so a full data
  window never delays them (`SocketChannel` in `nautilus.transport`).
- **Collector** — The in-memory buffer an operator emits into during a single `process` /
  `on_watermark` call (`Collector`). The actor drains it and performs the (awaiting) sends
  afterward, so operator code stays synchronous and each step is a self-contained critical section.
- **Operator context** — The object handed to an operator at `open` time holding its dependencies: its
  id, subtask index and count, state backend, clock, config, and a metrics recorder
  (`OperatorContext`).
- **Runner (runtime)** — The component that executes a graph. Stage 0's runner is `run_local_chain`
  (single process, in-memory channels); `run()` is the synchronous one-line wrapper around it.
- **RunResult** — What a run returns: the final output batches plus the run's telemetry report
  (`RunResult`; `result.telemetry`).
- **Worker process** — For multicore, nautilus runs one operating-system process per core, each with
  its own event loop and no shared memory. **(planned; Stage 0 is a single process.)**
- **Compile and deploy** — The one-time step that lowers the dataflow graph to a physical plan and
  starts the worker processes. Routing during the run is decided locally by each sender; there is no
  central scheduler on the data path. **(planned.)**

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

Every run emits self-describing telemetry. The terms below are summarized here; the full catalog of
metrics is in [`telemetry-reference.md`](telemetry-reference.md).

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
