"""The compiler: lower a :class:`~nautilus.api.LogicalGraph` to a runnable, serializable
:class:`~nautilus.compile.plan.PhysicalPlan`.

This is the one-time control-phase step between *what to compute* (the IR) and *how to run it* (the
runtime). It expands each vertex into its parallel instances, names the physical operators, selects a
partitioner for every edge, and synthesizes the collecting sink — emitting a plan that is plain data
plus operator factories, so it can be cloudpickled to a worker that has never seen the original graph.

It imports only :mod:`nautilus.api`. The plan describes routing as stateless *specs*, never live
:class:`~nautilus.runtime.partition.Partitioner` objects, so the compiler never reaches into the
runtime; an import-linter contract enforces that direction.
"""

from nautilus.compile.lower import compile_graph
from nautilus.compile.plan import (
    ForwardSpec,
    KeyGroupSpec,
    PartitionerSpec,
    PhysicalEdge,
    PhysicalOperator,
    PhysicalPlan,
    RoundRobinSpec,
)

__all__ = [
    "compile_graph",
    "PhysicalPlan",
    "PhysicalOperator",
    "PhysicalEdge",
    "PartitionerSpec",
    "ForwardSpec",
    "RoundRobinSpec",
    "KeyGroupSpec",
]
