# Nautilus

A decentralized, entirely-streaming parallel compute framework. Comparable to Dask, but with no
central scheduler.

- **Decentralized.** The computation is a dataflow graph of operators that run as actors and route
  data to each other locally — no central component sits on the data path.
- **Entirely streaming.** Bounded data is just a finite stream that ends, so the same operators handle
  bounded and unbounded inputs.
- **Backpressure end to end.** Operators are joined by bounded channels with credit-based flow
  control, so a slow sink slows the source instead of growing memory without bound.
- **Arrow-first.** Records move as Arrow `RecordBatch`es — columnar and micro-batched, passed by
  reference in-process and serialized once to Arrow IPC across a socket.

For the vocabulary and data model (operators, frames, watermarks, …) see `docs/glossary.md`; for the
architecture and the reasons behind it, `DESIGN.md`; for what's built and what's next,
`IMPLEMENTATION_PLAN.md`.

## Status

Early development. A single-process streaming engine runs today, plus the compiler and a multicore
deployer that runs a graph across worker processes over a mix of in-process and socket edges. Multi-node
validation is designed but not yet built. See `IMPLEMENTATION_PLAN.md`.

## Python

The fluent `Stream` DSL builds and runs a pipeline; each combinator returns a new stream.

```python
import pyarrow as pa
from nautilus import source

lines = pa.record_batch({"line": ["the quick brown fox", "the lazy dog"]})
result = source(lines).tokenize("line", "word").count_by("word").run()
print(result.to_pylist())              # [{'word': 'the', 'count': 2}, ...]
```

`.run(workers=N)` deploys the *same* graph across N worker processes — the only change from the
single-process run. `.join` combines two streams into an inner equi-join (both sides shuffled on the key,
so equal keys meet on one instance):

```python
joined = source(orders).join(source(customers), on="customer_id").run(workers=2)
```

For the simplest case the one-liner `run(src, [op, ...])` takes a source and a list of operators
directly. See [`docs/dsl-reference.md`](docs/dsl-reference.md) for every combinator.

## CLI

```bash
nautilus examples                 # list runnable example pipelines
nautilus run wordcount            # run one; prints its output and a telemetry summary
nautilus run wordcount --parallelism 3      # run each operator as 3 instances (keyed ops shuffle by key)
nautilus run wordcount --workers 2 --parallelism 2   # spread across 2 worker processes
nautilus run wordcount --show markdown   # the telemetry digest formatted for an AI agent
nautilus run wordcount --save report.json
nautilus catalog                  # every metric nautilus records, with its meaning

# Print a ready-to-paste prompt for an AI coding agent: your task plus the run's
# telemetry, what each metric means, and the relevant source files.
nautilus task "make Tokenize faster" --on wordcount

# Performance work: the bench-* pipelines generate millions of rows (the examples above are tiny) and
# model real-stream stressors — bench-skew (hot keys), bench-late (out-of-order events), bench-backpressure
# (a slow stage). Scale from the environment; vary --parallelism / --workers to stress shuffle and transport.
NAUTILUS_BENCH_ROWS=2000000 nautilus run bench-skew --parallelism 4 --save report.json
nautilus bench bench-keyed        # measure throughput over many trials: median ± IQR, vs the baseline
nautilus bench-check              # re-run benchmarks/baseline.json (incl. a 2-worker TCP run); CI gate
```

Run your own pipeline with `nautilus run mymodule:builder`, where `builder()` returns
`(source, transforms)`. Also available as `python -m nautilus`. Full command reference:
[`docs/cli-reference.md`](docs/cli-reference.md).

## Development

```bash
uv venv --python 3.12
uv pip install -e ".[dev,fast]"
uv run pytest -q
```
