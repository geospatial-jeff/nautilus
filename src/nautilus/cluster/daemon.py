"""The worker daemon: a long-lived process a coordinator dials to run one job at a time.

The remote replacement for a spawned worker. Across machines the coordinator cannot spawn a process or
``mp.Queue`` to it, so a daemon is started independently (one per container) and the coordinator *connects*
to its control port. It is the same data-plane slice a spawned worker runs
(:func:`~nautilus.cluster.worker_main.run_worker_slice`), with the control messages crossing the
:mod:`~nautilus.cluster.control_link` socket instead of queues.

**Job-scoped state machine** (the control connection is per job):

1. *Idle* — accept one coordinator connection. A second concurrent connection is refused, so only one job
   runs at a time; a fresh connection after a job is the next job (the daemon stays up).
2. *Running* — read a ``Launch``, run the slice on this event loop binding a fresh ephemeral data port,
   reporting ``Register``/``Done``/``Failed`` back over the socket.
3. A control drop or ``Abort`` *before* the job's ``Done`` means the coordinator is gone or aborting →
   cancel the job, tear it down, return to *Idle*. A control close *after* ``Done`` is the normal job
   boundary → return to *Idle*. Either way the daemon does not exit.

**No-orphan guarantee.** Locally a coordinator SIGKILLs a wedged child; it cannot signal a non-child PID
across a machine, so the daemon enforces it itself. asyncio cancellation is cooperative — it lands only at
an ``await``, so an operator wedged in a non-yielding loop or a blocking C call never unwinds. When an
abort cannot unwind within a grace window, an out-of-band watchdog hard-exits the daemon's own process;
compose's restart policy brings a fresh one back. A normal job never trips it.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
from contextlib import suppress
from typing import Any

from nautilus.cluster.control_link import Abort, ControlLinkError, Launch, encode, read_message
from nautilus.cluster.worker_main import run_worker_slice

_ABORT_GRACE = (
    10.0  # seconds an aborted job may take to unwind before the daemon hard-exits its process
)

# A coordinator that vanishes mid-protocol shows up as one of these on the control socket.
_GONE = (asyncio.IncompleteReadError, ConnectionError, ControlLinkError, OSError)


async def _run_job(
    launch: Launch,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    bind_host: str,
    advertise_host: str,
) -> None:
    """Run one job's slice while watching the control connection for an abort. The slice writes its
    terminal ``Done``/``Failed`` to the socket; a control drop or ``Abort`` before that cancels it.
    """
    loop = asyncio.get_running_loop()
    abort = asyncio.Event()
    address_book: asyncio.Future[Any] = loop.create_future()

    async def control_reader() -> None:
        # After Launch the coordinator sends exactly the address book, then stays silent until it aborts
        # or closes. So: deliver the address book, then treat the next message (Abort) or EOF as an abort.
        try:
            book = await read_message(reader)
            if not address_book.done():
                address_book.set_result(book)
            while True:
                if isinstance(await read_message(reader), Abort):
                    return
        except _GONE:
            return
        finally:
            abort.set()
            if not address_book.done():
                address_book.set_exception(
                    ConnectionError("coordinator closed before sending the address book")
                )

    def send_event(message: Any) -> None:
        writer.write(encode(message))

    reader_task = loop.create_task(control_reader())
    job_task = loop.create_task(
        run_worker_slice(
            launch.worker_id,
            launch.plan_bytes,
            launch.placement,
            bind_host,
            advertise_host,
            launch.capacity,
            launch.config,
            send_event,
            lambda: address_book,
        )
    )
    abort_task = loop.create_task(abort.wait())
    try:
        await asyncio.wait({job_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if job_task.done():
            # The slice finished and wrote its terminal Done/Failed to the buffer; flush before closing.
            with suppress(Exception):
                await writer.drain()
            return
        # Abort fired first. Cancel the job; arm an out-of-band watchdog so a wedged execute() that the
        # cancel cannot unwind still releases this process (the network replacement for SIGKILL).
        watchdog = threading.Timer(_ABORT_GRACE, os._exit, args=(1,))
        watchdog.daemon = True
        watchdog.start()
        job_task.cancel()
        with suppress(BaseException):
            await job_task
        watchdog.cancel()
    finally:
        reader_task.cancel()
        abort_task.cancel()
        for task in (reader_task, abort_task):
            with suppress(BaseException):
                await task


async def _serve(listen_host: str, listen_port: int, bind_host: str, advertise_host: str) -> None:
    job_lock = asyncio.Lock()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if job_lock.locked():
            writer.close()  # one job at a time: refuse a second coordinator rather than queue it
            with suppress(Exception):
                await writer.wait_closed()
            return
        async with job_lock:
            try:
                launch = await read_message(reader)
                if isinstance(launch, Launch):
                    await _run_job(launch, reader, writer, bind_host, advertise_host)
            except _GONE:
                pass  # coordinator vanished before/at launch — nothing to report; stay up
            finally:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()

    server = await asyncio.start_server(handle, listen_host, listen_port)
    async with server:
        await server.serve_forever()


def run_daemon(listen_host: str, listen_port: int, bind_host: str, advertise_host: str) -> None:
    """Run the daemon until killed, serving one job per coordinator connection and returning to idle
    between jobs. ``listen_host``:``listen_port`` is the control port the coordinator dials; ``bind_host``
    is the interface its data listener binds (``0.0.0.0`` in a container); ``advertise_host`` is the
    routable host peers dial for its data edges (its service/DNS name)."""
    asyncio.run(_serve(listen_host, listen_port, bind_host, advertise_host))


def healthcheck(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if a daemon's control port accepts a TCP connection — the compose healthcheck probe.
    Proves the accept loop is listening (and DNS resolves), not merely that the process exists."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


__all__ = ["run_daemon", "healthcheck"]
