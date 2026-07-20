<!-- Generated from nautilus.telemetry.catalog — do not edit by hand; regenerate with `python -m nautilus.telemetry.report.reference`. -->
# Nautilus telemetry reference

Every metric Nautilus records, generated from the metric catalog (report schema v3, catalog v1).

Each entry gives what the metric measures, its unit and tier, and the metrics it relates to.

## Metrics

| name | kind | unit | tier | reduction | labels | meaning | relates_to | derivation |
|---|---|---|---|---|---|---|---|---|
| `async.capacity` | gauge | count | COUNTERS | last | operator_id, subtask_index | The configured max_in_flight bound on concurrent async I/O tasks for an async stage — the ceiling async.in_flight rises to before the actor stops reading (the stage's backpressure). | async.in_flight |  |
| `async.in_flight` | gauge | count | COUNTERS | max | operator_id, subtask_index | High-water number of async I/O tasks in flight at once on one instance. At most async.capacity. | async.capacity, async.requests |  |
| `async.request_micros` | counter | microseconds | COUNTERS | sum | operator_id, subtask_index | Summed wall time an async stage's I/O tasks spent awaiting external I/O (each write's or fetch's perf_counter span). Several tasks run at once, so this sum can exceed the run's wall time; the gap to wall is the overlap. Distinct from runtime.step_micros, which for an async stage counts only the actor's own coordination (a transform's integrate/on_eos self-time), never the awaited I/O. | async.requests, async.in_flight, runtime.step_micros |  |
| `async.requests` | counter | count | COUNTERS | sum | operator_id, subtask_index | Number of async I/O tasks an async stage completed — one per batch an async sink writes or an async transform fetches. Recorded by the actor when it reaps the task, not by the awaiting code. | async.request_micros, async.in_flight |  |
| `async.timeouts` | counter | count | COUNTERS | sum | operator_id, subtask_index | Number of async I/O tasks cancelled for exceeding the stage's per-request timeout_micros. Zero unless a timeout is configured. | async.requests |  |
| `edge.batches_sent` | counter | batches | COUNTERS | sum | operator_id, edge_src, edge_dst, channel_index | Data batches pushed by the producer. |  |  |
| `edge.credit_wait_micros` | counter | microseconds | COUNTERS | sum | operator_id, edge_src, edge_dst, channel_index | Time the producer awaited flow-control credit on a channel. |  |  |
| `edge.frames_sent` | counter | count | COUNTERS | sum | operator_id, edge_src, edge_dst, channel_index, frame_type | Frames pushed by the producer. | operator.rows_out |  |
| `edge.input_wait_micros` | counter | microseconds | COUNTERS | sum | operator_id | Time the actor was suspended in mailbox.get awaiting any input. | edge.send_wait_micros |  |
| `edge.queue_capacity` | gauge | count | COUNTERS | last | operator_id, edge_src, edge_dst, channel_index | Configured channel capacity. | edge.queue_depth |  |
| `edge.queue_depth` | gauge | count | COUNTERS | max | operator_id, edge_src, edge_dst, channel_index | Channel.depth() sampled by the producer after each send (high-water). | edge.queue_capacity | queue_depth / queue_capacity = saturation |
| `edge.queue_depth_hist` | histogram | count | COUNTERS | sum | operator_id, edge_src, edge_dst, channel_index | Distribution of Channel.depth() sampled by the producer after each send. Where edge.queue_depth gives the high-water level, this gives how often each level occurred — the share of sends near capacity. In-process channels only (a socket channel reports no depth). | edge.queue_depth, edge.queue_capacity |  |
| `edge.rows_sent` | counter | rows | COUNTERS | sum | operator_id, edge_src, edge_dst, channel_index | Rows pushed by the producer. | operator.rows_out |  |
| `edge.send_wait_micros` | counter | microseconds | COUNTERS | sum | operator_id, edge_src, edge_dst, channel_index | Time the sending actor was suspended inside channel.send awaiting capacity. | edge.input_wait_micros, edge.queue_depth, edge.queue_capacity | send_wait_micros > 0 = the send awaited |
| `eos.expected` | gauge | count | COUNTERS | last | operator_id | Number of input channels (mailbox.num_inputs). | eos.received |  |
| `eos.received` | counter | count | COUNTERS | sum | operator_id, input_index | Number of EOS frames received, written as each one arrives. | eos.expected |  |
| `host.cpu_percent` | gauge | percent | COUNTERS | last |  | psutil.cpu_percent(): host-wide CPU utilization since the previous sample. Per OS host; not summed across processes sharing a host. |  |  |
| `host.mem_percent` | gauge | percent | COUNTERS | last |  | psutil.virtual_memory().percent: fraction of host physical memory in use at the sample. Per OS host; not summed across processes sharing a host. |  |  |
| `host.net_bytes_recv` | gauge | bytes | COUNTERS | last |  | Bytes received across all host network interfaces since the previous sample (psutil.net_io_counters delta). Host-wide, not summed across co-located workers. |  |  |
| `host.net_bytes_sent` | gauge | bytes | COUNTERS | last |  | Bytes sent across all host network interfaces since the previous sample (psutil.net_io_counters delta). Host-wide, not summed across co-located workers; the OS-level counterpart to the per-edge transport.bytes_sent, which counts application payload. | transport.bytes_sent |  |
| `io.wait_micros` | counter | microseconds | COUNTERS | sum | operator_id | Wall time a source spent awaiting external I/O, recorded by the source itself via ctx.io_wait(). A source is the one operator that may await inside its own code, so its runtime.step_micros counts both its on-CPU frame construction and the awaits it performs between frames; subtracting this from step_micros leaves the on-CPU time, so a source whose io.wait_micros is most of its step_micros is I/O-bound, not compute-bound. Zero unless a source brackets its awaits. | runtime.step_micros |  |
| `operator.batch_rows` | histogram | rows | COUNTERS | sum | operator_id, subtask_index | num_rows of each inbound batch. | operator.process_micros |  |
| `operator.batches_in` | counter | batches | COUNTERS | sum | operator_id, subtask_index | Number of data batches received. |  |  |
| `operator.batches_out` | counter | batches | COUNTERS | sum | operator_id, subtask_index | Number of non-empty data batches emitted. |  |  |
| `operator.bytes_in` | counter | bytes | FULL | sum | operator_id, subtask_index | Approximate Arrow buffer size of received batches (get_total_buffer_size proxy). | operator.rows_in |  |
| `operator.bytes_out` | counter | bytes | FULL | sum | operator_id, subtask_index | Approximate Arrow buffer size of emitted batches (get_total_buffer_size proxy). | operator.rows_out |  |
| `operator.errors` | counter | count | COUNTERS | sum | operator_id, exc_type | Number of exceptions raised in an operator lifecycle method. |  |  |
| `operator.on_eos_calls` | counter | calls | COUNTERS | sum | operator_id, subtask_index | Number of op.on_eos invocations. |  |  |
| `operator.on_eos_micros` | histogram | microseconds | COUNTERS | sum | operator_id, subtask_index | Wall time of the op.on_eos(out) end-of-stream flush. |  |  |
| `operator.process_calls` | counter | calls | COUNTERS | sum | operator_id, subtask_index | Number of op.process invocations. |  |  |
| `operator.process_micros` | histogram | microseconds | COUNTERS | sum | operator_id, subtask_index | Wall time of one op.process(batch) call, measured with perf_counter_ns. | operator.batch_rows |  |
| `operator.rows_in` | counter | rows | COUNTERS | sum | operator_id, subtask_index | Sum of num_rows across received batches. | operator.rows_out |  |
| `operator.rows_out` | counter | rows | COUNTERS | sum | operator_id, subtask_index | Sum of num_rows across emitted batches. | operator.rows_in | rows_out / rows_in = selectivity |
| `partition.route_micros` | histogram | microseconds | COUNTERS | sum | operator_id, edge_dst | Wall time of one partitioner.route(batch) call on the sending actor, measured with perf_counter_ns. Spans key extraction, per-key assignment, and the take into sub-batches; sits between the operator's process and the downstream send. | edge.rows_sent, edge.send_wait_micros |  |
| `placement.instances_per_worker` | gauge | count | COUNTERS | last | node | Number of operator instances placed on a worker. |  |  |
| `process.cpu_percent` | gauge | percent | COUNTERS | last |  | psutil.Process.cpu_percent() over the interval since the previous sample, where 100 equals one fully used CPU core. | runtime.loop_lag_micros |  |
| `process.num_fds` | gauge | count | COUNTERS | last |  | psutil.Process.num_fds(): open file descriptors at the sample (POSIX; omitted elsewhere). |  |  |
| `process.num_threads` | gauge | count | COUNTERS | last |  | psutil.Process.num_threads(): OS threads in this process at the sample. |  |  |
| `process.rss_bytes` | gauge | bytes | COUNTERS | last |  | psutil.Process.memory_info().rss: resident set size of this process at the sample. |  |  |
| `runtime.await_count` | counter | count | COUNTERS | sum | operator_id, subtask_index | Number of awaits the actor performed. | runtime.step_micros |  |
| `runtime.gil_percent` | gauge | percent | FULL | last |  | Fraction of the sampling interval the interpreter's global interpreter lock was held under contention, from the gilknocker monitor thread (100 means fully contended). Per process; recorded only at the FULL tier, and omitted when gilknocker is not installed. | runtime.loop_lag_micros, runtime.step_micros |  |
| `runtime.loop_lag_micros` | histogram | microseconds | COUNTERS | sum |  | Difference between the requested asyncio.sleep interval and the monotonic time that actually elapsed before the sampler resumed, measured with perf_counter_ns. | runtime.step_micros |  |
| `runtime.step_micros` | counter | microseconds | COUNTERS | sum | operator_id, subtask_index | Summed wall time the actor spent producing output: a transform's process and on_eos critical sections, or a source's frame generation (which includes any await a self-pacing source performs between frames). Accumulated in nanoseconds and reduced to microseconds once, so a step shorter than a microsecond still counts. | runtime.await_count |  |
| `state.entries` | gauge | count | COUNTERS | max | operator_id, state_name | Count of (key, namespace) entries held in a named state. | state.keys | entries / keys = entries-per-key |
| `state.keys` | gauge | count | COUNTERS | max | operator_id, state_name | Count of distinct keys held in a named state. | state.entries |  |
| `transport.bytes_sent` | counter | bytes | FULL | sum | operator_id, edge_src, edge_dst, channel_index | Bytes written to a cross-process channel. |  |  |
| `transport.decode_micros` | counter | microseconds | COUNTERS | sum | operator_id | Wall time this instance's inbound socket reader spent deserializing frames from the wire (Arrow IPC for a batch, msgpack for a control frame). Runs in the background read loop, so it overlaps the actor's own work; recorded once when the instance closes. No cross-process inbound edge means zero. | transport.encode_micros |  |
| `transport.encode_micros` | counter | microseconds | COUNTERS | sum | operator_id, edge_src, edge_dst, channel_index | Wall time the producer spent serializing frames to the wire (Arrow IPC for a batch, msgpack for a control frame) on a cross-process edge. A component of edge.send_wait_micros, separated out so serialization is distinguishable from flow-control and network waiting. | transport.bytes_sent, edge.send_wait_micros, transport.decode_micros |  |

## Events

| name | tier | fields | meaning |
|---|---|---|---|
| `eos.forwarded` | COUNTERS_PLUS_EVENTS | operator_id, wall_micros | An instance received EOS on all inputs and broadcast EOS downstream. |
| `operator.error` | COUNTERS | operator_id, op_class, phase, exc_type, message, traceback, frame_kind, input_index, batch_rows, source_location | An exception was raised in a lifecycle method (recorded, then re-raised unchanged). |
| `operator.lifecycle.close` | COUNTERS | operator_id, rows_in, rows_out, wall_micros | An instance closed, with its end-of-life counts. |
| `operator.lifecycle.open` | COUNTERS | operator_id, op_class, source_location, num_inputs | An instance opened. Carries the source location anchoring it to code. |
