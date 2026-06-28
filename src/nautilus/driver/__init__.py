"""The boundary: turn a graph into a result.

``nautilus.driver`` is where a :class:`~nautilus.api.LogicalGraph` becomes a
:class:`~nautilus.driver.result.RunResult`. It compiles the graph, runs it through the executor
(:mod:`nautilus.runtime`), and assembles the telemetry report — the single-process counterpart to the
cluster coordinator (the distributed boundary). It is the one layer above the data path that may import
the telemetry-report layer; :mod:`nautilus.runtime` (the data path) may not, which an import-linter
contract now enforces at the package boundary rather than module by module.

``run`` is the synchronous one-liner most callers want; the fluent :class:`nautilus.dsl.Stream` is the
richer builder whose ``.run`` terminal lands here. ``run_plan`` / ``run_compiled`` are the lower-level
runners the DSL terminal and the cluster coordinator both call.
"""

from nautilus.driver.local import run, run_local_chain
from nautilus.driver.parallel import (
    Stage,
    graph_from_pipeline,
    graph_from_stages,
    run_parallel_chain,
)
from nautilus.driver.result import RunResult
from nautilus.driver.run import plan_to_topology, run_compiled, run_plan

__all__ = [
    "run",
    "run_local_chain",
    "run_plan",
    "run_compiled",
    "plan_to_topology",
    "RunResult",
    "graph_from_pipeline",
    "Stage",
    "graph_from_stages",
    "run_parallel_chain",
]
