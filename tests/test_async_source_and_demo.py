"""The unified async source path: a source can ``await`` between batches (the loop stays responsive so
the hardware sampler keeps ticking), and its ``frames()`` generator is finalized promptly on cancel so
user resource cleanup is not deferred to GC."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from nautilus.core.operator import SourceOperator
from nautilus.core.records import Frame
from nautilus.demos import DemoStreamSource
from nautilus.operators import KeyedTumblingSum
from nautilus.runtime.local import run_local_chain
from nautilus.telemetry.recorder import TelemetryConfig
from nautilus.testing import data
from nautilus.windows import TumblingEventTimeWindows


def _window_op():
    return KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(1_000_000))


async def test_demo_stream_stays_loop_responsive():
    src = DemoStreamSource(interval_s=0.01, max_batches=8)
    result = await run_local_chain(
        src,
        [_window_op()],
        telemetry=TelemetryConfig(sample_interval_micros=5000),  # 5ms sampler ticks
    )
    rep = result.telemetry
    # the source emitted every batch...
    src_rows = sum(
        p.value for p in rep.operator("source").counters if p.name == "operator.rows_out"
    )
    assert src_rows == 8 * 3
    # ...the window operator fired live...
    fires = sum(p.value for p in rep.operator("op0").counters if p.name == "window.fires")
    assert fires > 0
    # ...and the sampler kept ticking WHILE the source slept (loop never froze)
    proc = rep.operator("process")
    lag = next((h for h in proc.histograms if h.name == "runtime.loop_lag_micros"), None)
    assert lag is not None and lag.count >= 1
    assert src.closed


async def test_unbounded_demo_cancels_cleanly():
    src = DemoStreamSource(interval_s=0.01, max_batches=None)  # never ends on its own
    task = asyncio.create_task(
        run_local_chain(src, [_window_op()], telemetry=TelemetryConfig(sample_interval_micros=5000))
    )
    await asyncio.sleep(0.05)  # let a few batches flow
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert src.closed  # close() still ran in finally; no phantom operator.error, no hang


async def test_source_generator_finalized_on_cancel():
    """A try/finally inside frames() runs promptly on cancel (via aclose), not deferred to GC — the
    deterministic-cleanup guarantee distinct from the source's own close()."""
    cleaned = asyncio.Event()

    class FinallySource(SourceOperator):
        async def frames(self) -> AsyncIterator[Frame]:
            n = 0
            try:
                while True:  # unbounded; only cancellation ends it
                    yield data(key=["a"], val=[1], ts=[n])
                    await asyncio.sleep(0.01)
                    n += 1
            finally:
                cleaned.set()

    task = asyncio.create_task(
        run_local_chain(
            FinallySource(), [_window_op()], telemetry=TelemetryConfig(sample_system=False)
        )
    )
    await asyncio.sleep(0.03)  # let it suspend at the yield
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()  # generator body's finally ran during the unwind, not at GC
