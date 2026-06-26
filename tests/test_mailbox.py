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
