"""``SocketChannel``: a credit-limited :class:`~nautilus.runtime.channel.Channel` across a socket.

The two ends share one full-duplex connection. Data frames are limited by a credit window — the
producer sends only while it holds a credit, the consumer returns one per data frame in :meth:`recv` —
so a fast producer cannot outrun a slow consumer. Control frames are sent without a credit, so a full
data window never delays a watermark or end-of-stream. A background task reads the socket: credit
returns on the producer end, data and control frames (queued for :meth:`recv`) on the consumer end.

Termination is explicit. When the reader stops — clean end-of-stream, an early disconnect, or a
malformed message — the channel goes terminal and wakes any blocked `send`/`recv`, which then raise
:class:`TransportClosed` instead of hanging. A disconnect before end-of-stream is an error; after it,
clean.

After its last frame the producer calls :meth:`finish`, which half-closes the write side and drains the
returning credits until the consumer has read everything and closed. Closing while the consumer still
has unread bytes would reset the connection and could drop them; draining first avoids that.

One writer per end means writes need no lock; the credit decrement and the `writer.write` happen under
one lock with no await between, so a cancelled `send` cannot lose a credit or split a frame.
"""

from __future__ import annotations

import asyncio
import contextlib
from time import perf_counter_ns

from nautilus.core.records import EOS, Frame
from nautilus.runtime.channel import DEFAULT_CAPACITY, Channel
from nautilus.transport.framing import Kind, decode, encode_credit, encode_frame, read_message


class TransportClosed(RuntimeError):
    """Raised by :meth:`SocketChannel.send` / :meth:`SocketChannel.recv` once the connection has
    closed or failed."""


class _Terminal:
    """Queue sentinel that marks end-of-input for :meth:`SocketChannel.recv`."""


_TERMINAL = _Terminal()

_DRAIN_TIMEOUT = 5.0  # seconds finish() waits for the consumer to drain and close before close()


class SocketChannel(Channel):
    """One end of a cross-process edge. Drop-in for :class:`InProcChannel` (`send`/`recv`/`depth`)."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._reader = reader
        self._writer = writer
        self._capacity = capacity
        self._credits = capacity  # producer end: remaining data-send permits
        self._cond = asyncio.Condition()  # guards _credits and the terminal state
        self._incoming: asyncio.Queue[Frame | _Terminal] = asyncio.Queue()  # consumer end
        self._terminated = False
        self._error: BaseException | None = None
        self._eos_seen = False  # consumer end: an EOS frame was received before the peer closed
        self._closing = False  # producer end: finish() in progress, so peer EOF is expected
        self._bytes_written = 0  # cumulative wire bytes this end has written (forward data + control)
        self._credit_wait_ns = 0  # cumulative time send() blocked awaiting a data credit
        self._read_task: asyncio.Task[None] = asyncio.create_task(self._read_loop())

    async def send(self, frame: Frame) -> None:
        message = encode_frame(frame)
        if frame.is_control:
            self._raise_if_terminated()
            await self._send_bytes(message)
            return
        async with self._cond:
            if self._credits == 0:  # time only the genuine flow-control stall (not the happy path)
                w0 = perf_counter_ns()
                while self._credits == 0:
                    self._raise_if_terminated()
                    await self._cond.wait()
                self._credit_wait_ns += perf_counter_ns() - w0
            self._raise_if_terminated()
            self._credits -= 1
            # Buffer synchronously, still under the lock and with no await between spending the
            # credit and handing the bytes to the transport, so a cancellation here cannot lose a
            # credit or split a frame. drain() (backpressure) happens after, outside the lock.
            self._writer.write(message)
            self._bytes_written += len(message)
        with contextlib.suppress(ConnectionError, OSError):
            await self._writer.drain()

    async def recv(self) -> Frame:
        item = await self._incoming.get()
        if isinstance(item, _Terminal):
            raise self._error or TransportClosed("recv() on a closed channel")
        if not item.is_control:  # a data slot was freed → return one credit to the producer
            # Best-effort: if the producer has already finished and gone, the credit is moot.
            with contextlib.suppress(ConnectionError, OSError):
                await self._send_bytes(encode_credit(1))
        return item

    def depth(self) -> int | None:
        return None

    def bytes_written(self) -> int | None:
        return self._bytes_written

    def credit_wait_micros(self) -> int | None:
        return self._credit_wait_ns // 1000

    async def finish(self) -> None:
        """Producer-side graceful end of stream: call after the last frame (the EOS), before
        :meth:`close`.

        Half-closes the write side, then waits for the consumer to drain and close (returning credits
        meanwhile) so :meth:`close` does not reset the connection on unread data. A consumer that does
        not close within ``_DRAIN_TIMEOUT`` is left to the abortive :meth:`close`."""
        self._closing = True
        with contextlib.suppress(ConnectionError, OSError, RuntimeError):
            if self._writer.can_write_eof():
                self._writer.write_eof()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(self._read_task), _DRAIN_TIMEOUT)

    async def close(self) -> None:
        self._read_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._read_task
        await self._terminate(None)  # idempotent; wakes any blocked send/recv
        self._writer.close()
        with contextlib.suppress(OSError, asyncio.CancelledError):
            await self._writer.wait_closed()

    def _raise_if_terminated(self) -> None:
        if self._terminated:
            raise self._error or TransportClosed("channel closed")

    async def _send_bytes(self, message: bytes) -> None:
        self._writer.write(message)
        self._bytes_written += len(message)
        with contextlib.suppress(ConnectionError, OSError):
            await self._writer.drain()

    async def _read_loop(self) -> None:
        try:
            while True:
                kind, payload = await read_message(self._reader)
                obj = decode(kind, payload)
                if kind == Kind.CREDIT:
                    assert isinstance(obj, int)
                    await self._grant_credits(obj)
                else:
                    assert isinstance(obj, Frame)
                    if isinstance(obj, EOS):
                        self._eos_seen = True
                    await self._incoming.put(obj)
        except asyncio.CancelledError:
            raise  # close() cancelled us — normal teardown
        except (asyncio.IncompleteReadError, ConnectionError):
            clean = self._eos_seen or self._closing  # EOF after EOS, or during our own finish()
            disconnect = None if clean else TransportClosed("peer disconnected before EOS")
            await self._terminate(disconnect)
        except Exception as exc:  # malformed frame, credit overflow, etc. — propagate, don't hang
            await self._terminate(exc)

    async def _terminate(self, error: BaseException | None) -> None:
        async with self._cond:
            if self._terminated:
                return
            self._terminated = True
            self._error = error
            self._cond.notify_all()  # wake any send() blocked on credit
        await self._incoming.put(_TERMINAL)  # wake any recv() blocked on the queue

    async def _grant_credits(self, n: int) -> None:
        async with self._cond:
            if self._credits + n > self._capacity:
                raise RuntimeError(
                    f"credit overflow: {self._credits} + {n} exceeds window {self._capacity}"
                )
            self._credits += n
            self._cond.notify_all()
