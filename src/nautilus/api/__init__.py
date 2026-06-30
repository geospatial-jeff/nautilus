"""The logical IR: a frozen :class:`LogicalGraph` describing *what* to compute, not how.

This is the top of the layer stack — the artifact the fluent DSL builds and the compiler
(:mod:`nautilus.compile`) lowers to a runnable :class:`~nautilus.compile.plan.PhysicalPlan`. It is
deliberately a pure value layer: it names operators, their parallelism, and their keying, and imports
nothing else in nautilus. Everything physical — operator-id naming, the channel mesh, partitioner
selection — is the compiler's job, so the IR stays a description that an agent (or a test) can build,
diff, and serialize without dragging in the runtime.
"""

from nautilus.api.graph import (
    LogicalEdge,
    LogicalGraph,
    LogicalVertex,
    async_sink,
    linear_graph,
    one_input,
    source,
    two_input,
)

__all__ = [
    "LogicalGraph",
    "LogicalVertex",
    "LogicalEdge",
    "linear_graph",
    "one_input",
    "two_input",
    "async_sink",
    "source",
]
