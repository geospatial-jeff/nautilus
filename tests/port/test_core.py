"""Characterization tests pinning the core execution contracts for a Python->Rust port.

These are Tier 1 tests: they fix the *observable* behavior of the three files a rewrite must reproduce
byte-for-byte — the collector that feeds ``run_local_chain``, the injectable clocks, and the operator
ABCs a user subclasses. Each test drives the real object directly (no network, no wall-clock sleeps) so
a faithful reimplementation is provable by making this suite pass.

Where the brief's assumption disagreed with the code, the test pins what the code actually does and the
divergence is called out inline (see ``test_two_input_operator_has_no_key_columns_method``).
"""

from __future__ import annotations

import asyncio

import pytest

import nautilus.core.operator as opmod
from nautilus.core.operator import (
    AsyncOneInputOperator,
    AsyncSink,
    ListCollector,
    OneInputOperator,
    OperatorContext,
    TwoInputOperator,
)
from nautilus.core.time import SystemClock, TestClock
from nautilus.testing import batch

# --- (a) ListCollector drops zero-row batches --------------------------------------------------
#
# This underpins digest/row-count stability: an operator that emits an empty batch (a filter that
# dropped every row, a flush with nothing buffered) must not add a phantom batch to the output, or the
# structural digest and per-stage row counts would depend on incidental emptiness.


def test_list_collector_drops_zero_row_batch_keeps_nonempty() -> None:
    coll = ListCollector()
    coll.emit(batch(x=[]))  # 0-row: dropped
    coll.emit(batch(x=[1, 2]))  # 2-row: kept
    drained = coll.drain()
    assert [b.num_rows for b in drained] == [2]


def test_list_collector_drain_empties_the_buffer() -> None:
    coll = ListCollector()
    coll.emit(batch(x=[1, 2]))
    assert len(coll.drain()) == 1
    assert coll.drain() == []  # a second drain yields nothing — drain() resets the buffer


# --- (b) SystemClock.now_micros ----------------------------------------------------------------


def test_system_clock_now_micros_is_int_nondecreasing_and_near_wall_clock() -> None:
    import time

    clock = SystemClock()
    first = clock.now_micros()
    second = clock.now_micros()
    reference = time.time_ns() // 1000
    assert isinstance(first, int)
    assert second >= first  # non-decreasing across two reads
    # Both reads and the reference are taken back-to-back; a second of slack is a wide, non-flaky bound.
    assert abs(first - reference) < 1_000_000


# --- (c) OperatorContext.io_wait ---------------------------------------------------------------
#
# io_wait records ``io.wait_micros`` = elapsed_ns // 1000 (FLOOR, not round) tagged with operator_id,
# in a finally block so the metric is written even if the wrapped body raises. Time is mocked by
# patching the module-level perf_counter_ns so the elapsed value is exact and the FLOOR-vs-round
# distinction is deterministic (1900 ns floors to 1, but rounds to 2).


class _RecordingRecorder:
    """Captures ``incr`` calls — io_wait's only recorder verb — so the recorded value can be asserted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, object]]] = []

    def incr(self, name: str, n: int = 1, **labels: object) -> None:
        self.calls.append((name, n, labels))


def _mock_perf_counter(monkeypatch: pytest.MonkeyPatch, values: list[int]) -> None:
    seq = iter(values)
    monkeypatch.setattr(opmod, "perf_counter_ns", lambda: next(seq))


def test_io_wait_records_floor_micros_tagged_operator_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_perf_counter(monkeypatch, [0, 1_900])  # elapsed 1900 ns
    rec = _RecordingRecorder()
    ctx = OperatorContext("op-a", metrics=rec)

    async def body() -> None:
        async with ctx.io_wait():
            pass

    asyncio.run(body())
    # 1900 // 1000 == 1 (FLOOR); round(1.9) would be 2, so this pins floor.
    assert rec.calls == [("io.wait_micros", 1, {"operator_id": "op-a"})]


def test_io_wait_records_in_finally_even_when_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_perf_counter(monkeypatch, [0, 3_000])
    rec = _RecordingRecorder()
    ctx = OperatorContext("op-b", metrics=rec)

    async def body() -> None:
        async with ctx.io_wait():
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(body())
    # The metric is still recorded — the finally contract holds even on an exception path.
    assert rec.calls == [("io.wait_micros", 3, {"operator_id": "op-b"})]


def test_io_wait_under_null_recorder_records_nothing_and_does_not_raise() -> None:
    # Default metrics is the null recorder: io_wait must be a silent no-op, not an error.
    ctx = OperatorContext("op-c")

    async def body() -> None:
        async with ctx.io_wait():
            pass

    asyncio.run(body())  # completes without raising; nothing observable to assert beyond that


# --- (d) operator ABC defaults + abstractmethod enforcement ------------------------------------


class _MinOneInput(OneInputOperator):
    def process(self, batch, out) -> None: ...


class _MinAsyncOneInput(AsyncOneInputOperator):
    async def fetch(self, batch) -> object:
        return None

    def integrate(self, batch, result, ctx, out) -> None: ...


class _MinAsyncSink(AsyncSink):
    async def write(self, batch) -> None: ...


class _MinTwoInput(TwoInputOperator):
    def process_left(self, batch, out) -> None: ...

    def process_right(self, batch, out) -> None: ...


def test_one_input_operator_defaults() -> None:
    op = _MinOneInput()
    assert op.key_columns() is None
    coll = ListCollector()
    assert op.on_eos(coll) is None  # default on_eos is a no-op
    assert coll.drain() == []


def test_async_one_input_operator_defaults() -> None:
    op = _MinAsyncOneInput()
    assert op.key_columns() is None
    assert op.max_in_flight() == 8
    assert op.ordered() is True
    assert op.timeout_micros() is None


def test_async_sink_defaults() -> None:
    op = _MinAsyncSink()
    assert op.key_columns() is None
    assert op.max_in_flight() == 8
    assert op.timeout_micros() is None


def test_two_input_operator_on_eos_is_no_op() -> None:
    op = _MinTwoInput()
    coll = ListCollector()
    assert op.on_eos(coll) is None  # default on_eos is a no-op
    assert coll.drain() == []


def test_two_input_operator_has_no_key_columns_method() -> None:
    # DIVERGENCE FROM BRIEF: the brief expected TwoInputOperator to expose key_columns() == None like
    # the one-input operators, but TwoInputOperator defines no such method — its inputs are already
    # co-partitioned on the join key by the shuffle, so it declares no key here. Pinning the actual API.
    assert not hasattr(TwoInputOperator, "key_columns")
    op = _MinTwoInput()
    assert not hasattr(op, "key_columns")


@pytest.mark.parametrize(
    ("base", "missing_method"),
    [
        (OneInputOperator, "process"),
        (AsyncSink, "write"),
    ],
)
def test_omitting_the_single_abstractmethod_raises_typeerror(base, missing_method) -> None:
    subclass = type("Incomplete", (base,), {})
    with pytest.raises(TypeError, match=missing_method):
        subclass()


def test_async_one_input_missing_integrate_raises_typeerror() -> None:
    class OnlyFetch(AsyncOneInputOperator):
        async def fetch(self, batch) -> object:
            return None

    with pytest.raises(TypeError, match="integrate"):
        OnlyFetch()


def test_two_input_missing_process_right_raises_typeerror() -> None:
    class OnlyLeft(TwoInputOperator):
        def process_left(self, batch, out) -> None: ...

    with pytest.raises(TypeError, match="process_right"):
        OnlyLeft()


# --- (e) TestClock -----------------------------------------------------------------------------


def test_test_clock_starts_at_zero_and_at_given_start() -> None:
    assert TestClock().now_micros() == 0
    assert TestClock(100).now_micros() == 100


def test_test_clock_advance_moves_now_and_returns_it() -> None:
    clock = TestClock()
    assert clock.advance(5) == 5
    assert clock.now_micros() == 5
    assert clock.advance(3) == 8
    assert clock.now_micros() == 8


def test_test_clock_set_moves_now_forward_and_returns_it() -> None:
    clock = TestClock()
    assert clock.set(10) == 10
    assert clock.now_micros() == 10


def test_test_clock_advance_negative_raises() -> None:
    clock = TestClock()
    with pytest.raises(ValueError, match="cannot go backwards"):
        clock.advance(-1)


def test_test_clock_set_into_past_raises() -> None:
    clock = TestClock(10)
    with pytest.raises(ValueError, match="cannot go backwards"):
        clock.set(9)
