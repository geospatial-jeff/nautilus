"""The non-reordering fan-in that merges an operator instance's several input channels.

Correctness of event-time handling depends on **per-channel FIFO**: a record must never be
observed after a watermark that bounds it. The mailbox guarantees this by keeping *at most one*
outstanding ``recv`` per input channel and re-arming only the channel it just yielded from. It never
reorders within a channel and never drops a ready frame. A single-input mailbox (a linear pipeline
stage) has nothing to merge, so it awaits its one channel directly and skips the per-``get`` Task and
``asyncio.wait`` the fan-in path needs.

When several inputs are ready at once, the choice between them is a fairness tie-break, *not* a
correctness requirement (any per-channel-FIFO-preserving pick is correct). The fan-in rotates the
starting input each ``get`` so a continuously-ready input cannot starve the others — which would
otherwise let one input's watermark grow stale (stalling the combined min watermark) and defer a
sibling's fail-fast exception.

Each ``get`` returns ``(input_index, frame)`` so the actor can attribute watermarks/EOS to the
right input. Inputs close individually (on EOS); the mailbox is :attr:`exhausted` once all close.
:meth:`close` cancels any outstanding recvs on teardown.
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
        self._next_start = 0  # rotating tie-break start, for fairness across ready inputs

    @property
    def num_inputs(self) -> int:
        return len(self._channels)

    @property
    def exhausted(self) -> bool:
        return all(self._closed)

    def decode_micros(self) -> int:
        """Total microseconds the inbound channels spent deserializing the wire (0 for in-process
        inputs). The actor totals this once at close as ``transport.decode_micros``."""
        return sum(d for ch in self._channels if (d := ch.decode_micros()) is not None)

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

        # Steady-state short-circuit: a recv future armed by an earlier get() may already be done (its
        # frame buffered), so skip asyncio.wait and its callback churn entirely. Only block when nothing
        # is ready yet. Either way, leave the unchosen futures in place (still done) so the next call
        # returns them without re-issuing a recv.
        ready = self._ready()
        if not ready:
            armed = [fut for fut in self._pending if fut is not None]
            await asyncio.wait(armed, return_when=asyncio.FIRST_COMPLETED)
            ready = self._ready()

        chosen = self._pick(ready)
        fut = self._pending[chosen]
        assert fut is not None
        self._pending[chosen] = None
        return chosen, fut.result()

    def _ready(self) -> list[int]:
        return [i for i, fut in enumerate(self._pending) if fut is not None and fut.done()]

    def _pick(self, ready: list[int]) -> int:
        """Pick the first ready input at or after the rotating start, then advance the start past it —
        fair round-robin among ready inputs while preserving each channel's own FIFO order."""
        n = len(self._channels)
        ready_set = set(ready)
        for off in range(n):
            i = (self._next_start + off) % n
            if i in ready_set:
                self._next_start = (i + 1) % n
                return i
        raise AssertionError("_pick called with no ready channel")

    def close(self) -> None:
        """Cancel any outstanding recvs and mark every input closed (fan-in teardown). Idempotent, and a
        no-op for the single-input and clean-EOS paths, where nothing is ever left pending — it only
        matters when an actor unwinds (fail-fast/cancellation) with recvs still armed."""
        for i, fut in enumerate(self._pending):
            if fut is not None:
                if not fut.done():
                    fut.cancel()
                elif not fut.cancelled():
                    fut.exception()  # retrieve a set exception so asyncio doesn't warn it went unread
            self._pending[i] = None
            self._closed[i] = True

    def close_input(self, input_index: int) -> None:
        """Stop receiving on an input (called after its EOS has been consumed). In production the input's
        recv was already consumed (so nothing is pending), but this also handles a still-armed future
        defensively, matching :meth:`close`."""
        self._closed[input_index] = True
        fut = self._pending[input_index]
        if fut is not None:
            if not fut.done():
                fut.cancel()
            elif not fut.cancelled():
                fut.exception()  # retrieve a set exception so asyncio doesn't warn it went unread
        self._pending[input_index] = None
