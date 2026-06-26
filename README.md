# Nautilus

A decentralized, entirely-streaming parallel compute framework. Comparable to Dask, but with no
central scheduler.

- **You define the computation as a dataflow graph.** Build an immutable graph with a fluent DSL; a
  one-time compile-and-deploy step turns it into a physical plan and starts one worker process per
  core. After startup, no central component is on the data path.
- **Entirely streaming.** Bounded data is a finite stream that terminates. The same operators handle
  bounded and unbounded inputs.
- **Backpressure.** Operators are actors connected by channels with credit-based flow control; a
  slow sink propagates backpressure upstream to the source.
- **Arrow-first.** Records flow as Arrow `RecordBatch`es, micro-batched and zero-copy across
  processes.

For the core terms and data model (operators, frames, watermarks, …), see `docs/glossary.md`. See
`DESIGN.md` for the architecture and `IMPLEMENTATION_PLAN.md` for staged progress.

## Status

Early development. Single-node multicore is the current target; multi-node is accounted for in the
design (swappable transport endpoints, key-group addressing) but not yet built.

## CLI

```bash
nautilus examples                 # list runnable example pipelines
nautilus run wordcount            # run one; prints its output and a telemetry summary
nautilus run wordcount --show markdown   # the agent-readable telemetry digest
nautilus run wordcount --save report.json
nautilus catalog                  # the telemetry cheat-sheet: every number nautilus records and what it means

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
