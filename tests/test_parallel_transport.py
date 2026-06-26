"""Stage 1.5d: the identical parallel mesh runs over socket pairs, proving transport-agnosticism.

The capstone runs one ``source(P=1) → keyed(P=N) → sink`` graph twice — once with the in-process
factory, once with the socket-pair factory — and requires an equal collected multiset AND an equal
``structural_digest``. (Hash stability across processes is 1.5a's job; this proves the *wired graph* is
unchanged by the transport.) The remaining tests pin liveness, credit behavior at the tightest window,
and clean teardown of the 2·P·Q socket read-loops.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from time import perf_counter

from nautilus.core.records import EOS_FRAME
from nautilus.operators import InMemorySource, KeyedCount, KeyedTumblingSum, MapBatch
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.parallel import InProcFactory, Stage, run_parallel_chain
from nautilus.telemetry import TelemetryConfig
from nautilus.testing import TestClock, data, wm
from nautilus.transport.mesh import SocketPairFactory
from nautilus.windows import TumblingEventTimeWindows


def _ts_counts(result) -> Counter:
    return Counter(
        (r["key"], r["window_start"], r["window_end"], r["sum"]) for r in result.to_pylist()
    )


def _wc_counts(result) -> Counter:
    return Counter((r["word"], r["count"]) for r in result.to_pylist())


def _windowed_source() -> InMemorySource:
    frames = [
        data(key=["a", "b", "a", "c"], val=[1, 2, 3, 4], ts=[1, 2, 3, 4]),
        wm(10),
        data(key=["b", "a", "d", "c"], val=[5, 6, 7, 8], ts=[12, 13, 14, 15]),
        wm(20),
        data(key=["a", "d"], val=[9, 10], ts=[22, 23]),
        wm(30),
        EOS_FRAME,
    ]
    return InMemorySource(frames)


def _windowed_stage(q: int) -> list[Stage]:
    return [
        Stage(
            lambda: KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10)), q, ["key"]
        )
    ]


# --- capstone: transport-agnosticism -----------------------------------------------------------


async def test_inproc_and_socket_meshes_agree() -> None:
    inproc = await run_parallel_chain(
        _windowed_source(), _windowed_stage(3), clock=TestClock(), factory=InProcFactory()
    )
    sock = await run_parallel_chain(
        _windowed_source(), _windowed_stage(3), clock=TestClock(), factory=SocketPairFactory()
    )
    assert _ts_counts(inproc) == _ts_counts(sock)  # same rows out of the wired graph
    assert inproc.telemetry.structural_digest() == sock.telemetry.structural_digest()


# --- skewed key over sockets: empty instances still get EOS ------------------------------------


async def test_skew_over_sockets_terminates() -> None:
    frames = [
        data(key=["a"] * 8, val=list(range(8)), ts=list(range(8))),
        wm(10),
        EOS_FRAME,
    ]
    serial = await run_local_chain(
        InMemorySource(list(frames)),
        [KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))],
        clock=TestClock(),
    )
    par = await asyncio.wait_for(
        run_parallel_chain(
            InMemorySource(list(frames)),
            [
                Stage(
                    lambda: KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10)),
                    4,
                    ["key"],
                )
            ],
            clock=TestClock(),
            factory=SocketPairFactory(),
        ),
        timeout=20,
    )
    assert _ts_counts(par) == _ts_counts(serial)


# --- credit no-deadlock at the tightest window, with a P>1 × Q>1 connection ---------------------


async def test_no_deadlock_capacity_one_pxq_mesh() -> None:
    # source(1) → map(P=2) → KeyedCount(Q=3) → sink: the map→count edge is a 2×3 mesh at window 1.
    def stages() -> list[Stage]:
        return [
            Stage(lambda: MapBatch(lambda b: b), 2),  # RoundRobin from the source
            Stage(lambda: KeyedCount("word"), 3, ["word"]),  # keyed shuffle 2×3
        ]

    words = [["the", "cat", "the"], ["dog", "cat", "the"], ["fox", "fox", "cat"]]
    serial = await run_local_chain(
        InMemorySource([data(word=w) for w in words] + [EOS_FRAME]),
        [MapBatch(lambda b: b), KeyedCount("word")],
        clock=TestClock(),
    )
    par = await asyncio.wait_for(
        run_parallel_chain(
            InMemorySource([data(word=w) for w in words] + [EOS_FRAME]),
            stages(),
            capacity=1,
            clock=TestClock(),
            factory=SocketPairFactory(),
        ),
        timeout=20,
    )
    assert _wc_counts(par) == _wc_counts(serial)


# --- teardown cleanliness ----------------------------------------------------------------------


async def test_socket_mesh_leaks_no_tasks_and_tears_down_fast() -> None:
    baseline = len(asyncio.all_tasks())
    t0 = perf_counter()
    await run_parallel_chain(
        _windowed_source(),
        _windowed_stage(3),
        clock=TestClock(),
        telemetry=TelemetryConfig(clock=TestClock(), sample_system=False),
        factory=SocketPairFactory(),
    )
    elapsed = perf_counter() - t0
    await asyncio.sleep(0)  # let cancelled read-loops settle
    # every one of the 2·P·Q socket read-loops was closed → tasks back to baseline
    assert len(asyncio.all_tasks()) <= baseline
    # close() never calls finish(), so teardown is nowhere near the 5 s drain timeout
    assert elapsed < 5.0


# --- the in-process factory is a true no-op on close -------------------------------------------


async def test_inproc_factory_close_all_is_a_true_noop() -> None:
    f = InProcFactory()
    send_end, recv_end = await f.pair(4)
    assert send_end is recv_end  # one object is both ends in-process
    assert not hasattr(send_end, "close")  # InProcChannel has no close() to call
    await f.close_all()  # genuine no-op; must not raise
