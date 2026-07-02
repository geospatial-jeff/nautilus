"""Placement: which worker hosts each operator instance.

A pure function the coordinator runs once, before any worker starts, and ships as plain data. Placement
is per-operator round-robin over the sorted worker ids: subtask ``i`` of every operator goes to worker
``sorted_ids[i % W]``. Two consequences matter:

* **Same-index subtasks co-locate.** ``op0[0]`` and ``op1[0]`` land on the same worker, so a forward or
  same-width keyed edge between them stays an in-process channel — only edges that genuinely cross
  workers become sockets. A true keyed shuffle (e.g. one instance fanning out to several) is
  many-to-many and still crosses.
* **It is deterministic and W-aware.** The coordinator computes the map once and ships it whole to every
  worker as plain data — each worker reads its own slice and recomputes nothing; determinism is what
  lets any reader reproduce the same map. Capping ``W`` at the plan's maximum parallelism
  (:func:`max_parallelism`) keeps round-robin from ever assigning an empty worker.
"""

from __future__ import annotations

from collections.abc import Iterable

from nautilus.compile import PhysicalPlan


def place(plan: PhysicalPlan, worker_ids: Iterable[int]) -> dict[tuple[str, int], int]:
    """Map every ``(operator_id, subtask_index)`` to the worker id that hosts it."""
    ids = sorted(worker_ids)
    if not ids:
        raise ValueError("placement needs at least one worker")
    return {
        (op.operator_id, subtask): ids[subtask % len(ids)]
        for op in plan.operators
        for subtask in range(op.parallelism)
    }


def max_parallelism(plan: PhysicalPlan) -> int:
    """The widest operator. Effective worker count is capped here: under per-operator round-robin only
    workers ``0..max_parallelism-1`` ever host an instance, so a larger ``W`` would spawn empty workers.
    """
    return max(op.parallelism for op in plan.operators)


def effective_worker_count(plan: PhysicalPlan, requested: int) -> int:
    """How many workers a ``requested``-worker deploy actually runs on: capped at :func:`max_parallelism`,
    since round-robin placement would leave any worker past that width with no instance to host."""
    return min(requested, max_parallelism(plan))
