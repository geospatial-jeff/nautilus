# Nautilus

A decentralized, entirely-streaming parallel compute framework. Comparable to Dask, but with no
central scheduler.

- **Decentralized.** The computation is a dataflow graph of operators that run as actors and route
  data to each other locally — no central component sits on the data path.
- **Entirely streaming.** Bounded data is just a finite stream that ends, so the same operators handle
  bounded and unbounded inputs.
- **Backpressure end to end.** Operators are joined by bounded channels with credit-based flow
  control, so a slow sink slows the source instead of growing memory without bound.
- **Arrow-first.** Records move as Arrow `RecordBatch`es — columnar, micro-batched, and zero-copy
  across processes.

For the vocabulary and data model (operators, frames, watermarks, …) see `docs/glossary.md`; for the
architecture and the reasons behind it, `DESIGN.md`; for what's built and what's next,
`IMPLEMENTATION_PLAN.md`.

## Status

Early development. A single-process streaming engine runs today, along with a two-process credit
transport; multicore deploy and multi-node are designed but not yet built. See
`IMPLEMENTATION_PLAN.md`.

## CLI

```bash
nautilus examples                 # list runnable example pipelines
nautilus run wordcount            # run one; prints its output and a telemetry summary
nautilus run wordcount --show markdown   # the telemetry digest formatted for an AI agent
nautilus run wordcount --save report.json
nautilus catalog                  # every metric nautilus records, with its meaning

# Print a ready-to-paste prompt for an AI coding agent: your task plus the run's
# telemetry, what each metric means, and the relevant source files.
nautilus task "make Tokenize faster" --on wordcount
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
