# Examples

Runnable snippets for common tasks. Each starts from an Arrow batch via `source(...)`; swap in your own
`SourceOperator` for real data.

## Transform, filter, reshape

Chain per-batch operations — add a computed column, keep the rows you want, and project down to the
columns you need. Each step runs on a batch at a time, using Arrow compute functions.

```python
import pyarrow as pa, pyarrow.compute as pc
from nautilus import source

orders = pa.record_batch({"item": ["a", "b", "c"], "qty": [2.0, 5.0, 1.0], "price": [3.0, 1.5, 3.0]})

result = (
    source(orders)
    .with_column("total", lambda b: pc.multiply(b["qty"], b["price"]))
    .filter(lambda b: pc.greater(b["total"], 4.0))
    .select("item", "total")
    .collect()
)
# [{'item': 'a', 'total': 6.0}, {'item': 'b', 'total': 7.5}]
```

## Group and aggregate

`agg_by` groups by one or more keys and reduces each group. Each keyword names an output column as
`(input_column, function)`, where the function is one of `sum`, `count`, `mean`, `min`, or `max`. A null
key — for example from an unmatched row after a `how="left"` join — forms its own group.

```python
readings = pa.record_batch({"city": ["A", "A", "B"], "temp": [10.0, 20.0, 30.0]})

source(readings).agg_by("city", avg=("temp", "mean"), n=("temp", "count")).collect()
# [{'city': 'A', 'avg': 15.0, 'n': 2}, {'city': 'B', 'avg': 30.0, 'n': 1}]
```

## Join two streams

Join on a shared key. `how` decides which non-matching rows survive: `"inner"` (the default) drops them,
`"left"` keeps unmatched left rows with the right columns null, and `"right"` / `"outer"` do the mirror
and both. Both sides are shuffled on the key, so equal keys meet on one instance.

```python
orders    = source(pa.record_batch({"cust": [1, 2], "item": ["x", "y"]}))
customers = source(pa.record_batch({"cust": [1, 3], "name": ["Ada", "Lin"]}))

orders.join(customers, on="cust").collect()
# [{'cust': 1, 'item': 'x', 'name': 'Ada'}]

orders.join(customers, on="cust", how="left").collect()   # also keep unmatched orders
# [{'cust': 1, 'item': 'x', 'name': 'Ada'}, {'cust': 2, 'item': 'y', 'name': None}]
```

!!! note
    Outer joins (`how="left"`, `"right"`, `"outer"`) run at **parallelism 1 only** and are rejected with
    `workers`/`parallelism` > 1 — a wider shuffle can route an instance no rows on a side, leaving that
    side's null columns untypable. Inner joins run at any width.

## Concatenate streams

`union` appends one stream to another — every row from both, duplicates kept, like SQL `UNION ALL`. The
two sides must share a schema, and neither is shuffled: batches flow straight through.

```python
clicks = source(pa.record_batch({"id": [1, 2], "kind": ["click", "click"]}))
views  = source(pa.record_batch({"id": [3], "kind": ["view"]}))

clicks.union(views).collect()
# [{'id': 1, 'kind': 'click'}, {'id': 2, 'kind': 'click'}, {'id': 3, 'kind': 'view'}]
```

## Explode a list column

`explode` turns a list column into one row per element, repeating the other columns — the inverse of
collecting values into a list. Empty and null lists produce no rows.

```python
posts = pa.record_batch({"post": [1, 2], "tags": [["a", "b"], ["c"]]})

source(posts).explode("tags").collect()
# [{'post': 1, 'tags': 'a'}, {'post': 1, 'tags': 'b'}, {'post': 2, 'tags': 'c'}]
```

## Enrich with async I/O

`map_async(fn)` runs an `async def fn(batch) -> batch` — one call **per batch**, not per row — and keeps
several calls in flight at once (`max_in_flight`, default 8). It's stateless, so it can emit in completion
order for lower latency (`ordered=False`; the default `True` keeps input order).

```python
async def enrich(batch):
    location = await lookup(batch["device_id"])   # your awaited call, e.g. an HTTP request
    return batch.append_column("location", location)

source(readings).map_async(enrich, max_in_flight=16).run()
```

## Count the whole stream, sort, take top N

There is no whole-stream `count`, `sort`, or `limit` combinator — those aren't streaming operations. Get
the rows with a terminal, then use Python:

```python
sales = source(pa.record_batch({"item": ["a", "b", "c"], "amount": [6.0, 7.5, 3.0]}))

rows  = sales.collect()
total = len(rows)                                                   # count every row -> 3
top2  = sorted(rows, key=lambda r: r["amount"], reverse=True)[:2]   # -> b (7.5), a (6.0)
```

## Run across workers

Any pipeline runs across worker processes by passing `workers` — the graph is unchanged, and each
operator runs at that many instances. Because workers are spawned, wrap the call in a `__main__` guard
when it's the top level of a script:

```python
def main():
    events = source(pa.record_batch({"user": [1, 2, 1], "action": ["view", "buy", "view"]}))
    result = events.agg_by("user", n=("action", "count")).run(workers=4)
    print(result.to_pylist())   # [{'user': 1, 'n': 2}, {'user': 2, 'n': 1}]

if __name__ == "__main__":
    main()
```

For every combinator and its arguments, see the [DSL reference](dsl-reference.md).
