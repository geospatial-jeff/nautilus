# Nautilus CLI reference

The `nautilus` command-line interface runs pipelines and reads their telemetry.

> This file mirrors `src/nautilus/cli.py`. Update it when commands, arguments, or options change.

## Invocation

- `nautilus <command> [options]` — the installed console script.
- `python -m nautilus <command> [options]` — equivalent.
- In this repository, prefix with `uv run` so the project environment is used:
  `uv run nautilus run wordcount`.
- Running `nautilus` with no command prints help.

## The PIPELINE argument

Several commands take a `PIPELINE`, which is either:

- a built-in example name (list them with `nautilus examples`), or
- `module:function` — an importable, zero-argument function that returns `(source, transforms)`.

## Telemetry tiers

The `--telemetry` option accepts one of:

| value | meaning |
|---|---|
| `off` | no telemetry |
| `counters` (default) | counters, gauges, histograms, and lifecycle/error events |
| `events` | everything in `counters` plus verbose events |
| `full` | everything in `events` plus byte-accounting metrics (walks Arrow buffers) |

## Commands

### run

Run a pipeline and print its output and telemetry.

```
nautilus run PIPELINE [options]
```

| argument / option | default | description |
|---|---|---|
| `PIPELINE` | required | built-in name or `module:function` |
| `--telemetry` | `counters` | `off` / `counters` / `events` / `full` |
| `--show` | `summary` | `summary` / `markdown` / `json` / `none` |
| `--save PATH` | none | write the full JSON report to `PATH` |
| `--capacity` | `16` | channel capacity (backpressure bound) |
| `--head` | `5` | rows of pipeline output to preview |
| `--workers` | `1` | worker processes to deploy across (`>1` spawns and distributes) |
| `--parallelism` | `1` | instances per operator (keyed operators shuffle by key) |

Example: `uv run nautilus run wordcount --show markdown --save report.json`
Distributed: `uv run nautilus run wordcount --workers 2 --parallelism 2`

### examples

List the built-in example pipelines. No arguments.

```
nautilus examples
```

### catalog

Print the telemetry metrics: each metric's unit, tier, and meaning.

```
nautilus catalog [--md]
```

| option | default | description |
|---|---|---|
| `--md` | `false` | print the full markdown reference instead of the table |

### task

Print a prompt for an AI coding agent: the task you describe, plus — when `--on` is given — the
telemetry from a run and the definitions of the metrics it produced.

```
nautilus task DESCRIPTION [options]
```

| argument / option | default | description |
|---|---|---|
| `DESCRIPTION` | required | what you want the agent to do |
| `--on PIPELINE` | none | run this pipeline first and include its telemetry |
| `--telemetry` | `counters` | `off` / `counters` / `events` / `full` |
| `--capacity` | `16` | channel capacity (backpressure bound) |
| `--save PATH` | none | write the prompt to `PATH` instead of stdout |

Example: `uv run nautilus task "make Tokenize faster" --on wordcount`

### reference

Print, or regenerate, the telemetry reference document (`docs/telemetry-reference.md`).

```
nautilus reference [--write]
```

| option | default | description |
|---|---|---|
| `--write` | `false` | regenerate `docs/telemetry-reference.md` |

### dashboard

Run a pipeline and serve a live telemetry dashboard over HTTP.

```
nautilus dashboard PIPELINE [options]
```

| argument / option | default | description |
|---|---|---|
| `PIPELINE` | required | built-in name or `module:function` |
| `--port` | `8787` | port to serve on (`0` = OS-assigned) |
| `--host` | `127.0.0.1` | bind host (`0.0.0.0` exposes it; add authentication) |
| `--telemetry` | `counters` | `off` / `counters` / `events` / `full` |
| `--capacity` | `16` | channel capacity (backpressure bound) |
| `--linger` / `--no-linger` | `--linger` | keep serving after a bounded run completes |
| `--max-seconds` | none | stop after N seconds (caps unbounded runs) |
| `--open` | `false` | open the dashboard in a browser |

Example: `uv run nautilus dashboard image-embed --open`

### serve

Serve a saved JSON report in the dashboard, without running a pipeline.

```
nautilus serve --report PATH [options]
```

| option | default | description |
|---|---|---|
| `--report PATH` | required | a saved JSON report (from `run --save`) |
| `--port` | `8787` | port to serve on (`0` = OS-assigned) |
| `--host` | `127.0.0.1` | bind host |

Example: `uv run nautilus serve --report report.json`

### version

Print the nautilus version. No arguments.

```
nautilus version
```

### bench

Measure a pipeline's throughput over repeated trials (median + IQR, not best-of-N), compare to a
baseline if one exists, and optionally update it. Telemetry must be at least `counters` (the structural
digest is the correctness anchor). This produces the before/after numbers a `PERFORMANCE_CHANGELOG.md`
entry records.

```
nautilus bench PIPELINE [options]
```

| option | default | description |
|---|---|---|
| `--trials` | `5` | measured runs (median + IQR over these) |
| `--warmup` | `1` | discarded warmup runs |
| `--rows` / `--batch` / `--keys` / `--wm-every` | per env | synthetic-source scale |
| `--parallelism` | `1` | instances per operator |
| `--workers` | `1` | worker processes (>1 deploys) |
| `--capacity` | `16` | channel capacity |
| `--telemetry` | `counters` | tier (must be ≥ `counters`) |
| `--json` | `false` | print the result as JSON |
| `--baseline PATH` | `benchmarks/baseline.json` | baseline to compare against / update |
| `--update` | `false` | write this result into the baseline (refused if nondeterministic) |
| `--label` | the pipeline | baseline entry name (keep a `--workers` variant alongside the serial one) |

### bench-check

Re-run every pipeline in the baseline at its recorded scale and fail on any regression or output change
— the CI regression gate. A change counts only when it clears both the threshold and twice the run-to-run
noise.

```
nautilus bench-check [options]
```

| option | default | description |
|---|---|---|
| `--baseline PATH` | `benchmarks/baseline.json` | baseline to check against |
| `--threshold` | `0.07` | floor (fraction) a change must clear to count |
| `--update` | `false` | rewrite the baseline from this run instead of checking |

## Exit codes

| code | meaning |
|---|---|
| `0` | success |
| `1` | could not bind `host:port` (`dashboard`, `serve`); a regression / output change (`bench-check`); or a nondeterministic `--update` (`bench`) |
| `2` | could not load the pipeline (`run`, `task`, `dashboard`, `bench`); or no baseline file exists (`bench-check`) |
