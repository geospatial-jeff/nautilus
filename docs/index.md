---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Nautilus

<p class="hero-tagline">A decentralized, entirely-streaming parallel compute framework — write a dataflow once, run it unchanged in one process or across a cluster.</p>

[Get started](getting-started.md){ .md-button .md-button--primary }
[Design](design.md){ .md-button }

</div>

<div class="hero-code" markdown>

```python
from nautilus import source

# same graph; workers=4 is the only change:
source(lines).tokenize("line", "word").count_by("word").run()
source(lines).tokenize("line", "word").count_by("word").run(workers=4)
```

</div>

---

<div class="grid cards" markdown>

-   :material-vector-triangle:{ .lg .middle } __Decentralized__

    ---

    Operators run as actors and route data to each other locally — no central scheduler between them.

    [:octicons-arrow-right-24: Concepts](design.md)

-   :material-transit-connection-variant:{ .lg .middle } __Entirely streaming__

    ---

    The same pipeline runs over a fixed dataset or a live stream, with bounded channels carrying backpressure end to end.

    [:octicons-arrow-right-24: Getting started](getting-started.md)

-   :material-table:{ .lg .middle } __Arrow-first__

    ---

    Records move as columnar `RecordBatch`es — by reference in-process, serialized once across processes.

    [:octicons-arrow-right-24: Glossary](glossary.md)

-   :material-chart-line:{ .lg .middle } __Built-in telemetry__

    ---

    Every run records rows, timings, and queue depths per operator — read them to see where time goes.

    [:octicons-arrow-right-24: Telemetry](telemetry-reference.md)

</div>
