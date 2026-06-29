"""``WorkerCohort`` — the seam between the coordinator and however its workers are reached.

The coordinator's bootstrap and completion loop need only three things from its workers: hand one a
control message, take the next event from any of them (and learn if one died), and stop them all at the
end. Those three are the *only* operations that differ between a single-machine run and a multi-node one.
Locally they are a ``multiprocessing`` spawn, two ``mp.Queue``s, and a child's exit code; across machines
(Stage 4.2) they become a dialed TCP control connection per worker. Reading a queue *and* a process's
exit code in one call — the way :func:`~nautilus.cluster.rendezvous.recv_event` fuses transport with
liveness — is exactly what tied the loop to one machine, so this ABC pulls that fusion behind ``send`` /
``next_event`` / ``reap`` and lets :func:`~nautilus.cluster.coordinator.deploy` run unchanged over either
backend.

:class:`LocalCohort` is the single-machine implementation: it wraps the ``(procs, events, commands)`` a
:func:`~nautilus.cluster.launcher.spawn_workers` produces and delegates to the existing ``recv_event`` and
:func:`~nautilus.cluster.launcher.reap`, so the local path stays byte-for-byte what it was.
:class:`RemoteCohort` is the multi-node implementation: it dials a roster of long-lived
:mod:`~nautilus.cluster.daemon` processes, carries the same messages over one
:mod:`~nautilus.cluster.control_link` TCP connection per worker, and detects a crash from that connection
closing before a worker's ``Done`` — there is no exit code to read across a machine.
"""

from __future__ import annotations

import selectors
import socket
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any

from nautilus.cluster.control_link import Abort, Launch, encode, take_message
from nautilus.cluster.launcher import reap
from nautilus.cluster.rendezvous import WorkerCrashed, recv_event
from nautilus.telemetry import TelemetryConfig

# Bound the control dial. A daemon can be milliseconds slow to bind even behind a healthcheck, and a
# host-run coordinator has no depends_on, so the dial retries a refused/transient connect with backoff
# until connect_timeout before giving up.
_CONTROL_DIAL_TIMEOUT = 5.0  # per-attempt connect timeout
_CONTROL_DIAL_BACKOFF = 0.2  # seconds between dial attempts
# Same keepalive triple as the data plane (transport.connector): a silent partition on the control
# connection becomes a bounded ConnectionError, since the completion wait is otherwise unbounded.
_KEEPALIVE = (("TCP_KEEPIDLE", 10), ("TCP_KEEPINTVL", 5), ("TCP_KEEPCNT", 3))


class WorkerCohort(ABC):
    """How the coordinator reaches its workers. Each method replaces one machine-specific primitive in the
    bootstrap/completion loop, so the loop itself carries no spawn, queue, or exit-code assumption.
    """

    @abstractmethod
    def send(self, worker_id: int, message: Any) -> None:
        """Deliver one control message to a single worker (the address book, at the end of the bind)."""

    @abstractmethod
    def next_event(self, timeout: float | None, watch: set[int] | None = None) -> Any:
        """Return the next event from any *watched* worker, or fail fast if a watched worker died.

        ``watch`` is the set of worker ids whose liveness still matters; ``None`` watches every worker.
        The completion loop narrows ``watch`` as each worker's ``Done`` arrives, because a worker that has
        already reported may exit or close its end during teardown and must not fail a run that is already
        complete. ``timeout`` bounds *silence* and, when given, raises ``TimeoutError`` — the bootstrap,
        where a silent worker is a hang; ``None`` waits indefinitely for a busy worker, crash detection
        still active. Raises :class:`~nautilus.cluster.rendezvous.WorkerCrashed` if a watched worker died
        without reporting."""

    @abstractmethod
    def reap(self) -> None:
        """Stop every worker and release its resources. Unconditional — called in ``deploy``'s ``finally``
        so a worker never lingers, whether the run finished or failed."""


class LocalCohort(WorkerCohort):
    """Workers as local child processes reached over ``multiprocessing`` queues — the single-machine path.
    Wraps the ``(procs, events, commands)`` from :func:`~nautilus.cluster.launcher.spawn_workers`; crash
    detection is a child's exit code, read by :func:`~nautilus.cluster.rendezvous.recv_event`."""

    def __init__(self, procs: list[Any], events: Any, commands: dict[int, Any]) -> None:
        self._procs = procs
        self._events = events
        self._commands = commands

    def send(self, worker_id: int, message: Any) -> None:
        self._commands[worker_id].put(message)

    def next_event(self, timeout: float | None, watch: set[int] | None = None) -> Any:
        procs = self._procs if watch is None else [self._procs[w] for w in watch]
        return recv_event(self._events, procs, timeout)

    def reap(self) -> None:
        reap(self._procs)


def _enable_keepalive(sock: socket.socket) -> None:
    """TCP keepalive for a control connection, so a silent partition becomes a bounded error. The
    ``TCP_*`` tunables are Linux-specific and skipped where absent; best-effort."""
    with suppress(OSError):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in _KEEPALIVE:
            if hasattr(socket, name):
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)


def _dial_control(host: str, port: int, connect_timeout: float) -> socket.socket:
    """Dial a daemon's control port, retrying a refused/transient connect with backoff until
    ``connect_timeout`` — the daemon may be slightly slow to bind. Raises ``ConnectionError`` if it never
    accepts in time. Returns a blocking socket with keepalive on."""
    deadline = time.monotonic() + connect_timeout
    last: Exception | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ConnectionError(
                f"daemon at {host}:{port} never accepted within {connect_timeout:.0f}s"
            ) from last
        try:
            sock = socket.create_connection(
                (host, port), timeout=min(remaining, _CONTROL_DIAL_TIMEOUT)
            )
        except OSError as exc:
            last = exc
            time.sleep(min(_CONTROL_DIAL_BACKOFF, max(0.0, deadline - time.monotonic())))
            continue
        sock.settimeout(
            None
        )  # blocking sends; reads are gated by the selector, so they never block
        _enable_keepalive(sock)
        return sock


class _DaemonConn:
    """One coordinator→daemon control socket and its read-reassembly buffer. Sends block (control
    messages are small and infrequent); reads are driven by :meth:`RemoteCohort.next_event`'s selector,
    since one readable event may carry a partial frame or several."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = bytearray()

    @property
    def sock(self) -> socket.socket:
        return self._sock

    def send(self, message: Any) -> None:
        self._sock.sendall(encode(message))

    def read_available(self) -> bool:
        """Read whatever is ready into the buffer; return True if the connection dropped. A clean close
        gives an empty read (EOF); a daemon that refuses by closing with unread bytes (a busy daemon
        rejecting a new job) sends a reset, so ``recv`` raises ``ConnectionResetError`` — both mean the
        same thing here, the worker is gone."""
        try:
            chunk = self._sock.recv(65536)
        except OSError:
            return True
        if not chunk:
            return True
        self._buffer += chunk
        return False

    def take_message(self) -> Any | None:
        return take_message(self._buffer)

    def close(self) -> None:
        with suppress(OSError):
            self._sock.close()


class RemoteCohort(WorkerCohort):
    """Workers as long-lived daemons, one TCP control connection each — the multi-node path. Built by
    :meth:`launch`, which dials the roster and sends each daemon its ``Launch``. Crash detection is the
    control connection closing before a worker's ``Done``: there is no exit code across a machine.
    """

    def __init__(self, conns: dict[int, _DaemonConn]) -> None:
        self._conns = conns

    @classmethod
    def launch(
        cls,
        daemons: list[tuple[str, int]],
        plan_bytes: bytes,
        placement: dict[tuple[str, int], int],
        capacity: int,
        config: TelemetryConfig,
        effective: int,
        connect_timeout: float,
    ) -> RemoteCohort:
        """Dial the first ``effective`` daemons on the roster, assign ``worker_id = roster index``, and
        send each its ``Launch`` (the plan, placement, capacity, config). Surplus roster entries are left
        untouched. On any failure, closes every connection already opened."""
        conns: dict[int, _DaemonConn] = {}
        try:
            for worker_id in range(effective):
                host, port = daemons[worker_id]
                conns[worker_id] = _DaemonConn(_dial_control(host, port, connect_timeout))
            for worker_id, conn in conns.items():
                try:
                    conn.send(Launch(worker_id, plan_bytes, placement, capacity, config))
                except OSError as exc:
                    # A daemon already serving a job refuses by closing, so the launch send hits a reset.
                    raise WorkerCrashed(
                        f"worker {worker_id} closed before accepting Launch"
                    ) from exc
        except BaseException:
            for conn in conns.values():
                conn.close()
            raise
        return cls(conns)

    def send(self, worker_id: int, message: Any) -> None:
        try:
            self._conns[worker_id].send(message)
        except OSError as exc:
            raise WorkerCrashed(
                f"worker {worker_id} control connection closed before it could be sent to"
            ) from exc

    def next_event(self, timeout: float | None, watch: set[int] | None = None) -> Any:
        watched = set(self._conns) if watch is None else set(watch)
        # A prior recv may have buffered more than one message; return a complete buffered one first.
        for worker_id in watched:
            message = self._conns[worker_id].take_message()
            if message is not None:
                return message
        selector = selectors.DefaultSelector()
        for worker_id in watched:
            selector.register(self._conns[worker_id].sock, selectors.EVENT_READ, worker_id)
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                wait = None if deadline is None else max(0.0, deadline - time.monotonic())
                ready = selector.select(wait)
                if not ready:
                    raise TimeoutError(f"no control message from any worker within {timeout:.0f}s")
                for key, _ in ready:
                    worker_id = key.data
                    conn = self._conns[worker_id]
                    if conn.read_available():
                        raise WorkerCrashed(
                            f"worker {worker_id} control connection closed before reporting Done"
                        )
                    message = conn.take_message()
                    if message is not None:
                        return message
        finally:
            selector.close()

    def reap(self) -> None:
        for conn in self._conns.values():
            with suppress(OSError):
                conn.send(Abort())  # graceful stop; a closed peer just raises and is ignored
            conn.close()


__all__ = ["WorkerCohort", "LocalCohort", "RemoteCohort"]
