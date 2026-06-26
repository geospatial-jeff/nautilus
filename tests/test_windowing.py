"""Stage 0 demo: keyed tumbling windows fire on watermark advance; an idle input does not stall
event-time progress."""

import asyncio

from nautilus.core.operator import OperatorContext
from nautilus.core.records import EOS, EOS_FRAME, Batch
from nautilus.operators import InMemorySource, KeyedTumblingSum
from nautilus.runtime.actor import Output, run_transform
from nautilus.runtime.channel import InProcChannel
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.mailbox import Mailbox
from nautilus.runtime.partition import Forward
from nautilus.testing import IDLE_FRAME, data, wm
from nautilus.windows import TumblingEventTimeWindows


def _rows(batches):
    rows = []
    for rb in batches:
        for k, s, e, sm in zip(
            rb.column("key").to_pylist(),
            rb.column("window_start").to_pylist(),
            rb.column("window_end").to_pylist(),
            rb.column("sum").to_pylist(),
            strict=True,
        ):
            rows.append((k, s, e, sm))
    return sorted(rows)


async def test_tumbling_sum_fires_on_watermark_and_flushes_at_eos():
    frames = [
        data(key=["a", "a"], val=[1, 2], ts=[1, 5]),  # window [0,10) -> 3
        wm(9),  # nothing fires yet (window end 10 > 9)
        data(key=["a"], val=[10], ts=[12]),  # out-of-order arrival for window [10,20)
        wm(10),  # [0,10) fires
        wm(20),  # [10,20) fires
        EOS_FRAME,
    ]
    op = KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))
    results = await run_local_chain(InMemorySource(frames), [op])
    assert _rows(results) == [("a", 0, 10, 3), ("a", 10, 20, 10)]


async def _drive_two_input(op, frames0, frames1):
    chans = [InProcChannel(128), InProcChannel(128)]
    out_chan = InProcChannel(128)
    outputs = [Output([out_chan], Forward())]
    results: list = []

    async def feed(ch, frames):
        for f in frames:
            await ch.send(f)

    async def collect():
        while True:
            fr = await out_chan.recv()
            if isinstance(fr, Batch):
                results.append(fr.data)
            elif isinstance(fr, EOS):
                return

    async with asyncio.TaskGroup() as tg:
        tg.create_task(run_transform(op, OperatorContext("op"), Mailbox(chans), outputs))
        tg.create_task(feed(chans[0], frames0))
        tg.create_task(feed(chans[1], frames1))
        tg.create_task(collect())
    return results


async def test_idle_input_does_not_freeze_windows():
    op = KeyedTumblingSum("key", "val", "ts", TumblingEventTimeWindows(10))
    frames0 = [data(key=["a"], val=[5], ts=[3]), wm(10), wm(20), EOS_FRAME]
    frames1 = [IDLE_FRAME, EOS_FRAME]  # input 1 never produces a watermark
    results = await _drive_two_input(op, frames0, frames1)
    # Without idle exclusion the combined watermark would be pinned at MIN and nothing would fire.
    assert _rows(results) == [("a", 0, 10, 5)]
