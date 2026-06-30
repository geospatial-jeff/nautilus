"""The physical plan — the serializable artifact a worker runs.

A :class:`PhysicalPlan` is what crosses to a worker: the operators expanded with their parallelism, the
edges between them, and a *spec* per edge that says how to route. It is plain data plus the operator
``factory`` callables, so :func:`cloudpickle.dumps` can ship it to a process that never imported the
original graph. Two properties keep the plan neutral, so a worker can run a plan it never compiled:

* **It carries no live partitioner.** A spec (:class:`ForwardSpec` / :class:`RoundRobinSpec` /
  :class:`KeyGroupSpec`) is a stateless selection the runtime turns into a fresh
  :class:`~nautilus.runtime.partition.Partitioner` at wiring time. A :class:`RoundRobin`'s rotation
  cursor is runtime state that must never be serialized or shared between workers, so it cannot live
  in the plan.
* **It carries no telemetry topology.** The structural facts here are neutral
  (:class:`PhysicalOperator` / :class:`PhysicalEdge`); the boundary that builds the report translates
  them into a :class:`~nautilus.telemetry.report.Topology`. So the compiler depends only on the IR,
  never on the report layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

# --- Partitioner specs: a stateless selection the runtime instantiates per output ---------------


@dataclass(frozen=True, slots=True)
class ForwardSpec:
    """1:1 forwarding to a single downstream instance (a non-fan-out edge)."""

    #: The edge's partitioner label in the report topology. Selection of the runtime partitioner is by
    #: spec *type* (``partitioner_from_spec``), not by this string; it happens to equal the runtime
    #: class name, which a unit test pins so the two cannot drift.
    partitioner_name: ClassVar[str] = "Forward"


@dataclass(frozen=True, slots=True)
class RoundRobinSpec:
    """Keyless N-way rebalancing across the downstream instances."""

    partitioner_name: ClassVar[str] = "RoundRobin"


@dataclass(frozen=True, slots=True)
class KeyGroupSpec:
    """A keyed shuffle through key-group indirection: hash each key to one of ``len(group_table)``
    groups, then route by ``group_table[group]`` to an instance. The table is computed once at compile
    from the chosen group count ``G`` and the operator's parallelism ``Q`` (``G == Q`` gives the identity
    table); it carries no live state, so it serializes safely and is fixed for the run."""

    key_columns: tuple[str, ...]
    group_table: tuple[int, ...]
    partitioner_name: ClassVar[str] = "KeyGroupPartitioner"


#: A routing selection on an edge. The keyed shuffle routes through key groups (:class:`KeyGroupSpec`).
PartitionerSpec = ForwardSpec | RoundRobinSpec | KeyGroupSpec


# --- The plan: operators expanded by parallelism, plus the edges between them --------------------


@dataclass(frozen=True, slots=True)
class PhysicalOperator:
    """One operator expanded with its parallelism. ``factory`` builds a fresh operator instance for
    each of the ``parallelism`` subtasks; it is ``None`` for the synthesized sink, which the executor
    runs as a collecting loop rather than a user operator. ``op_class`` is the operator's class name,
    recorded once at compile time (the factory is not re-consulted to name it later), so it can label
    telemetry even before the factory has been called in the worker.
    """

    operator_id: str
    op_class: str
    kind: str  # "source" | "one_input" | "two_input" | "async_sink" | "sink"
    parallelism: int
    factory: Callable[[], object] | None


@dataclass(frozen=True, slots=True)
class PhysicalEdge:
    """A connection from one operator to the next, with the routing spec for the whole connection. The
    physical channel count is the fan-out — the destination operator's parallelism — resolved at
    wiring time, so the edge stays a single neutral fact independent of worker placement.

    ``dst_input_port`` is which input of the destination this edge feeds: 0 for a one-input operator (and
    the left side of a join), 1 for a join's right side. It is a plain int so the plan still cloudpickles
    neutrally; it stays off the report topology and the structural digest (a distinct source already
    makes the two join edges distinct), so adding it leaves every existing linear-graph digest unchanged.
    """

    src_operator_id: str
    dst_operator_id: str
    spec: PartitionerSpec
    dst_input_port: int = 0


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    """The runnable, cloudpickle-able lowering of a :class:`~nautilus.api.LogicalGraph`: operators in
    topological order (source first, synthesized sink last) and the edges between adjacent operators.
    """

    operators: tuple[PhysicalOperator, ...]
    edges: tuple[PhysicalEdge, ...]
