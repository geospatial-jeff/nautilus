"""The logical IR: a frozen :class:`LogicalGraph` describing *what* to compute, not how.

This is the top of the layer stack — the artifact the fluent DSL (Stage 3) will build and the compiler
(:mod:`nautilus.compile`) lowers to a runnable :class:`~nautilus.compile.plan.PhysicalPlan`. It is
deliberately a pure value layer: it names operators, their parallelism, and their keying, and imports
nothing else in nautilus. Everything physical — operator-id naming, the channel mesh, partitioner
selection — is the compiler's job, so the IR stays a description that an agent (or a test) can build,
diff, and serialize without dragging in the runtime.
"""

from nautilus.api.graph import LogicalGraph, LogicalVertex, linear_graph

__all__ = ["LogicalGraph", "LogicalVertex", "linear_graph"]
