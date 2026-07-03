"""The two-input actor path: a `two_input` vertex driven through the real compiler + executor.

These run a hand-built join graph (two sources into one `two_input` vertex on ports 0 and 1) through
`run_plan`, so they exercise `run_two_input`, the port-ordered mailbox, and the executor's list-valued
edge wiring together — left/right dispatch and EOS only after both sides close. The concrete join
operator lands in a later sub-stage; here a stub two-input operator isolates the actor/executor
behavior.
"""

from __future__ import annotations

from collections import Counter

import pyarrow as pa

from nautilus.api import LogicalEdge, LogicalGraph, one_input, source, two_input
from nautilus.core.operator import Collector, OperatorContext, TwoInputOperator
from nautilus.driver.run import run_plan
from nautilus.operators import InMemorySource, MapBatch
from nautilus.testing import EOS_FRAME, data


def _tag(batch: pa.RecordBatch, side: str) -> pa.RecordBatch:
    cols = [*batch.columns, pa.array([side] * batch.num_rows, pa.string())]
    return pa.RecordBatch.from_arrays(cols, names=[*batch.schema.names, "side"])


class _SideTagger(TwoInputOperator):
    """Passes every batch through, tagging the side it arrived on — so a test can read back which input
    port each row was dispatched to."""

    def process_left(self, batch: pa.RecordBatch, out: Collector) -> None:
        out.emit(_tag(batch, "L"))

    def process_right(self, batch: pa.RecordBatch, out: Collector) -> None:
        out.emit(_tag(batch, "R"))


class _EosLog(TwoInputOperator):
    """Records each on_eos fire and emits the running sequence — the observable proof that the terminal
    flush fires exactly once, and only after both inputs have closed (never once per input's EOS).
    """

    def open(self, ctx: OperatorContext) -> None:
        self._fires: list[int] = []

    def process_left(self, batch: pa.RecordBatch, out: Collector) -> None: ...

    def process_right(self, batch: pa.RecordBatch, out: Collector) -> None: ...

    def on_eos(self, out: Collector) -> None:
        self._fires.append(len(self._fires) + 1)
        out.emit(pa.record_batch({"fires": pa.array(self._fires, pa.int64())}))


class _CountBoth(TwoInputOperator):
    """Counts rows seen on each side, emitting the totals only at end of stream (on_eos) — so a non-zero,
    correct count proves both inputs were fully drained before the all-inputs-EOS flush."""

    def open(self, ctx: OperatorContext) -> None:
        self._left = 0
        self._right = 0

    def process_left(self, batch: pa.RecordBatch, out: Collector) -> None:
        self._left += batch.num_rows

    def process_right(self, batch: pa.RecordBatch, out: Collector) -> None:
        self._right += batch.num_rows

    def on_eos(self, out: Collector) -> None:
        out.emit(pa.record_batch({"left": [self._left], "right": [self._right]}))


async def test_two_input_dispatches_by_port() -> None:
    # Left batch -> process_left (tagged L), right batch -> process_right (tagged R).
    g = LogicalGraph(
        vertices=(
            source("L", lambda: InMemorySource([data(v=[1, 2]), EOS_FRAME])),
            source("R", lambda: InMemorySource([data(v=[3]), EOS_FRAME])),
            two_input("j", lambda: _SideTagger()),
        ),
        edges=(LogicalEdge("L", "j", 0), LogicalEdge("R", "j", 1)),
    )
    rows = (await run_plan(g)).to_pylist()
    assert Counter((r["v"], r["side"]) for r in rows) == Counter(
        {(1, "L"): 1, (2, "L"): 1, (3, "R"): 1}
    )


async def test_two_input_eos_after_both_inputs_close() -> None:
    # on_eos is the single terminal flush: the actor calls it exactly once, and only after BOTH inputs
    # have sent their data and EOS — one side closing must not fire it early. So a batch on each side
    # yields exactly one fire, regardless of how the two inputs interleave.
    g = LogicalGraph(
        vertices=(
            source("L", lambda: InMemorySource([data(v=[1, 2]), EOS_FRAME])),
            source("R", lambda: InMemorySource([data(v=[3]), EOS_FRAME])),
            two_input("j", lambda: _EosLog()),
        ),
        edges=(LogicalEdge("L", "j", 0), LogicalEdge("R", "j", 1)),
    )
    rows = (await run_plan(g)).to_pylist()
    # exactly one fire, emitted after both inputs closed — never once per input's EOS
    assert [r["fires"] for r in rows] == [1]


async def test_two_input_drains_a_parallel_left_before_flush() -> None:
    # The left port is fed by a parallelism-2 map (two channels into the join), the right by one
    # source. The join at parallelism 1 therefore has left_input_count == 2: indices 0,1 are left,
    # index 2 is right. A correct total proves the port-ordered mailbox routed both left channels to
    # process_left and the run drained every input before the terminal flush.
    g = LogicalGraph(
        vertices=(
            source("L", lambda: InMemorySource([data(v=[1, 2]), data(v=[3, 4]), EOS_FRAME])),
            one_input("mapL", lambda: MapBatch(lambda b: b), parallelism=2),
            source("R", lambda: InMemorySource([data(v=[5, 6, 7]), EOS_FRAME])),
            two_input("j", lambda: _CountBoth()),
        ),
        edges=(
            LogicalEdge("L", "mapL", 0),
            LogicalEdge("mapL", "j", 0),
            LogicalEdge("R", "j", 1),
        ),
    )
    rows = (await run_plan(g)).to_pylist()
    assert rows == [{"left": 4, "right": 3}]
