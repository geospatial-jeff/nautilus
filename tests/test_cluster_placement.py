"""Stage 2d: placement is pure per-operator round-robin over sorted worker ids.

Same-index subtasks co-locate (so a forward/diagonal edge stays in-process); a keyed fan-out spreads
across workers. The map must be deterministic so a worker can recompute nothing from just its id.
"""

from __future__ import annotations

from nautilus.api import LogicalVertex, linear_graph
from nautilus.cluster.placement import max_parallelism, place
from nautilus.compile import compile_graph
from nautilus.operators import InMemorySource, KeyedCount, Tokenize


def _plan(parallelism: int):
    return compile_graph(
        linear_graph(
            lambda: InMemorySource([]),
            [
                LogicalVertex("op0", lambda: Tokenize("line", "word"), "one_input"),
                LogicalVertex(
                    "op1", lambda: KeyedCount("word"), "one_input", parallelism, ("word",)
                ),
            ],
        )
    )


def test_per_operator_round_robin_over_sorted_ids() -> None:
    placement = place(_plan(3), [1, 0])  # unsorted input; placement sorts to [0, 1]
    assert placement[("op1", 0)] == 0
    assert placement[("op1", 1)] == 1
    assert placement[("op1", 2)] == 0  # wraps round-robin


def test_same_index_subtasks_co_locate() -> None:
    placement = place(_plan(2), [0, 1])
    # the single-instance operators and op1[0] all share worker 0, so their edges stay in-process
    assert placement[("source", 0)] == 0
    assert placement[("op0", 0)] == 0
    assert placement[("op1", 0)] == 0
    assert placement[("sink", 0)] == 0
    assert placement[("op1", 1)] == 1  # only the second keyed instance crosses


def test_placement_is_deterministic() -> None:
    plan = _plan(3)
    assert place(plan, [0, 1, 2]) == place(plan, [2, 1, 0])


def test_max_parallelism_is_the_widest_operator() -> None:
    assert max_parallelism(_plan(5)) == 5
    assert max_parallelism(_plan(1)) == 1
