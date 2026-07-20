# Getting started

From install to a running pipeline.

## Install

```bash
pip install "nautilus @ git+https://github.com/geospatial-jeff/nautilus"
```

## Count some words

Read a table, split each line into words, tally them:

```python
import pyarrow as pa
from nautilus import source

lines = pa.record_batch({"line": ["the cat sat", "the dog ran the cat"]})

result = (
    source(lines)              # (1)!
    .tokenize("line", "word")  # (2)!
    .count_by("word")          # (3)!
    .collect()                 # (4)!
)
for row in result:
    print(row)
```

1.  Start a stream from an Arrow batch.
2.  Split `line` into words — one row in, many out.
3.  Group by `word` and count.
4.  Run it. Nothing executes until a terminal like `.collect()`.

```text
{'word': 'the', 'count': 3}
{'word': 'cat', 'count': 2}
{'word': 'sat', 'count': 1}
{'word': 'dog', 'count': 1}
{'word': 'ran', 'count': 1}
```

Three things worth noticing:

- **It streams.** The data flows through in Arrow *batches* — columnar chunks of rows — never gathered
  into one table in the middle. Memory stays bounded no matter how large the input is.
- **It's lazy.** `source(...).tokenize(...).count_by(...)` only describes the pipeline. Nothing runs until
  a terminal — `.collect()` here, or `.run()` — asks for a result.
- **Keys converge.** `count_by` shuffles rows by key, so every `"cat"` meets on one instance to be
  counted. You get the same answer whether it runs in one process or across many.

## Scale out

The same pipeline runs across worker processes by passing `workers` — the graph is unchanged:

=== "One process"

    ```python
    source(lines).tokenize("line", "word").count_by("word").run()
    ```

=== "Four workers"

    ```python
    source(lines).tokenize("line", "word").count_by("word").run(workers=4)
    ```

`run(workers=4)` spawns the workers and shuffles across a socket; each operator runs at four instances,
and your result is identical. In a script, wrap the call in `if __name__ == "__main__":` — the workers
are spawned as fresh processes.

## See what it did

Every run records telemetry. From the CLI, `--show markdown` prints a summary of where the time went:

```bash
nautilus run wordcount --show markdown
```

```text
summary   rows_in=26 rows_out=26 errors=0   wall 5721us
operators — by self-time
 operator  class            rows_out  busy_us  send_wait_us
 op1       KeyedCount              8      377             3
 op0       Tokenize               14      125             1
 source    InMemorySource          4        6             7
```

The `KeyedCount` did most of the work, `Tokenize` turned 4 rows into 14, and nothing spent time blocked
waiting to send. Reading telemetry like this is how you tune a pipeline; the
[telemetry reference](telemetry-reference.md) lists every metric.

## Next

- [Examples](examples.md) — joins, aggregations, reshaping, async, distributed.
- [Concepts](design.md) — design.
- [DSL reference](dsl-reference.md) — every combinator.
- [Performance](performance.md) — benchmark numbers and how to run them.
