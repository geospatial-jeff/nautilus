# DSL reference (`nautilus.dsl`)

The fluent `Stream` DSL is the readable way to build a pipeline. A `Stream` is an **immutable** handle on
a dataflow under construction: every combinator returns a *new* `Stream` that adds one operator, so a
stream value is reusable and side-effect-free. You build one with `source(...)`, extend it with
combinators, and execute it with a terminal (`run` / `collect`). It produces a
[`LogicalGraph`](glossary.md) and nothing more — the runners are imported only when you call a terminal,
so building a stream never starts the engine.

```python
from nautilus import source            # also: from nautilus.dsl import source

result = source(lines).tokenize("line", "word").count_by("word").run()
```

## Starting a stream

| Function | What it does |
|---|---|
| `source(src)` | Start a stream. `src` is a `SourceOperator`, a bare `pyarrow.RecordBatch` (wrapped with a terminal EOS), or a sequence of batches/frames. |

## Combinators

Each returns a new `Stream`. `parallelism` (default 1) runs that operator as that many instances; a keyed
combinator shuffles its input on the key so each key's rows meet on one instance.

| Method | What it adds |
|---|---|
| `.map(fn)` | A pure batch → batch function (`MapBatch`). |
| `.filter(mask_fn)` | Keep rows where `mask_fn(batch)` (a boolean Arrow array) is true (`FilterRows`). |
| `.tokenize(in_col, out_col="word", lowercase=True)` | Split a string column into one row per whitespace token (`Tokenize`). |
| `.count_by(key_col, count_col="count")` | Count occurrences per key, emitted at end of stream; shuffled on `key_col` (`KeyedCount`). |
| `.agg_by(key_cols, **aggs)` | Grouped `sum`/`count`/`mean`/`min`/`max` per key, emitted at end of stream; shuffled on `key_cols` (`KeyedAgg`). Each keyword names an output column as `(input_col, func)`, e.g. `.agg_by("lat", mean=("temp", "mean"), hi=("temp", "max"))`. Its `count` is `COUNT(col)` — non-null values of that column; `.count_by` is the separate `COUNT(*)`, rows per key. |
| `.apply(operator, key_columns=None)` | The escape hatch: append any `OneInputOperator` instance. Keyed by `key_columns` if given, else the operator's own `key_columns()`. At parallelism > 1 the instance is deep-copied per subtask. |
| `.join(other, on=…)` | Inner equi-join with another stream — see below. |

## Joining

`.join(other, on="k")` (or `left_on=…, right_on=…` for differently-named keys) produces the inner
equi-join (`HashJoin`): a row for every left and right row whose join keys are equal. Both inputs are
shuffled on their join keys, so equal keys meet on one instance. The output is this stream's columns
followed by `other`'s non-key columns; a colliding non-key column name is rejected. A stream cannot be
joined to itself, and the two sides must name the same number of key columns.

```python
joined = source(orders).join(source(customers), on="customer_id")
```

## Terminals

| Method | What it does |
|---|---|
| `.run(workers=None, parallelism=None, key_groups=None, daemons=None, …)` | Synchronous: compile and run to completion, returning a `RunResult`. `workers > 1` deploys the *same* graph across that many spawned worker processes; `daemons=[(host, port), …]` deploys it across long-lived worker daemons instead (the multi-node path, worker count taken from the roster); `parallelism` sets every operator's instance count uniformly. |
| `.run_async(…)` | The `await`-able single-process form, for use inside a running event loop. |
| `.collect()` | Run and return the rows as `{column: value}` dicts (a convenience over `run().to_pylist()`). |
| `.to_graph(parallelism=None)` | The `LogicalGraph` this stream describes, without running it. |

## When to use what

- The **`Stream` DSL** is the default — readable, supports joins, and the one scale-up knob (`workers=` /
  `parallelism=`) is the only thing that changes between a single-process and a distributed run.
- The one-liner **`run(src, [op, ...])`** is the shortest path for the simplest case (a source and a flat
  list of operators, uniform parallelism).
- Drop to the explicit **`LogicalGraph`** ([`nautilus.api`](glossary.md)) only to build a shape the DSL
  doesn't express.
