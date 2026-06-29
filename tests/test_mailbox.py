import asyncio

import pytest

from nautilus.core.records import Watermark
from nautilus.runtime.channel import InProcChannel
from nautilus.runtime.mailbox import Mailbox


async def test_preserves_per_channel_fifo():
    a, b = InProcChannel(64), InProcChannel(64)
    mb = Mailbox([a, b])
    for i in range(5):
        await a.send(Watermark(i))
    for j in range(100, 103):
        await b.send(Watermark(j))

    got: dict[int, list[int]] = {0: [], 1: []}
    for _ in range(8):
        idx, frame = await mb.get()
        assert isinstance(frame, Watermark)
        got[idx].append(frame.t)

    # Within each channel, order is exactly the send order (no reordering).
    assert got[0] == [0, 1, 2, 3, 4]
    assert got[1] == [100, 101, 102]


async def test_close_input_marks_exhausted():
    a, b = InProcChannel(8), InProcChannel(8)
    mb = Mailbox([a, b])
    assert not mb.exhausted
    mb.close_input(0)
    assert not mb.exhausted
    mb.close_input(1)
    assert mb.exhausted


async def test_fan_in_is_fair_across_ready_inputs():
    # Both inputs always have data; the fan-in must not lock onto input 0 (which would stall input 1's
    # watermark). Over many gets the two are drawn roughly evenly (rotating tie-break).
    a, b = InProcChannel(64), InProcChannel(64)
    mb = Mailbox([a, b])
    for _ in range(20):
        await a.send(Watermark(1))
        await b.send(Watermark(2))
    counts = {0: 0, 1: 0}
    for _ in range(20):
        idx, _frame = await mb.get()
        counts[idx] += 1
    # Rotating tie-break draws the two ready inputs evenly, not just non-zero (no starvation).
    assert counts[0] == counts[1] == 10


async def test_close_input_cancels_an_armed_recv():
    # C17: with one input idle, get() leaves its recv armed; close_input must cancel it (the dead branch).
    a, b = InProcChannel(8), InProcChannel(8)
    mb = Mailbox([a, b])
    await b.send(Watermark(1))
    idx, _ = await mb.get()  # yields from b; a's recv is left armed in _pending
    assert idx == 1
    assert mb._pending[0] is not None and not mb._pending[0].done()
    mb.close_input(0)
    assert mb._pending[0] is None


async def test_close_cancels_outstanding_recvs():
    # C8: Mailbox.close() cancels any recv still armed (fan-in teardown on fail-fast/cancel).
    a, b = InProcChannel(8), InProcChannel(8)
    mb = Mailbox([a, b])
    await a.send(Watermark(1))
    await mb.get()  # arms b's recv (a is re-armed too); both pending now
    mb.close()
    assert all(p is None for p in mb._pending)
    assert mb.exhausted


async def test_in_process_backpressure_blocks_a_full_channel():
    # C111: the default in-process channel's bound IS the backpressure — a full channel blocks the sender.
    ch = InProcChannel(capacity=2)
    await ch.send(Watermark(1))
    await ch.send(Watermark(2))  # now full
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(ch.send(Watermark(3)), timeout=0.2)  # 3rd send blocks
    await ch.recv()  # drain one
    await asyncio.wait_for(ch.send(Watermark(3)), timeout=0.5)  # now it completes


def test_in_process_channel_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        InProcChannel(0)
