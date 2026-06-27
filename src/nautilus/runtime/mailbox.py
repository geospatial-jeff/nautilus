"""The non-reordering fan-in that merges an operator instance's several input channels.

Correctness of event-time handling depends on **per-channel FIFO**: a record must never be
observed after a watermark that bounds it. The mailbox guarantees this by keeping *at most one*
outstanding ``recv`` per input channel and a ``FIRST_COMPLETED`` merge that re-arms only the channel
it just yielded from. It never reorders within a channel and never drops a ready frame. A single-input
mailbox (a linear pipeline stage) has nothing to merge, so it awaits its one channel directly and skips
the per-``get`` Task and ``asyncio.wait`` the fan-in path needs.

Each ``get`` returns ``(input_index, frame)`` so the actor can attribute watermarks/EOS to the
right input. Inputs close individually (on EOS); the mailbox is :attr:`exhausted` once all close.
"""

from __future__ import annotations

import asyncio

from nautilus.core.records import Frame
from nautilus.runtime.channel import Channel


class Mailbox:
    def __init__(self, channels: list[Channel]) -> None:
        if not channels:
            raise ValueError("a mailbox needs at least one input channel")
        self._channels = channels
        self._pending: list[asyncio.Future[Frame] | None] = [None] * len(channels)
        self._closed = [False] * len(channels)

    @property
    def num_inputs(self) -> int:
        return len(self._channels)

    @property
    def exhausted(self) -> bool:
        return all(self._closed)

    async def get(self) -> tuple[int, Frame]:
        """Return the next ``(input_index, frame)`` preserving per-channel FIFO order."""
        if self.exhausted:
            raise RuntimeError("get() on an exhausted mailbox")

        if len(self._channels) == 1:
            # One input — the common linear case. Per-channel FIFO is just the channel's own order, so
            # await it directly and skip the Task allocation + asyncio.wait merge the fan-in path needs.
            # Nothing is left pending, so close_input(0) has no future to cancel.
            return 0, await self._channels[0].recv()

        # Arm exactly one recv per open, unarmed channel.
        for i, ch in enumerate(self._channels):
            if not self._closed[i] and self._pending[i] is None:
                self._pending[i] = asyncio.ensure_future(ch.recv())

        armed = {fut: i for i, fut in enumerate(self._pending) if fut is not None}
        done, _ = await asyncio.wait(armed.keys(), return_when=asyncio.FIRST_COMPLETED)

        # Deterministically yield from the lowest-indexed ready channel; leave the others' results
        # in place (still done) so the next call returns them without re-issuing a recv.
        chosen = min(armed[fut] for fut in done)
        fut = self._pending[chosen]
        assert fut is not None
        self._pending[chosen] = None
        return chosen, fut.result()

    def close_input(self, input_index: int) -> None:
        """Stop receiving on an input (called after its EOS has been consumed)."""
        self._closed[input_index] = True
        fut = self._pending[input_index]
        if fut is not None and not fut.done():
            fut.cancel()
        self._pending[input_index] = None
