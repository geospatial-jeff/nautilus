"""The two-phase startup bootstrap and the control-message receiver.

**Phase 1 (bind).** Every worker binds its :class:`EdgeListener` and registers its address. The
coordinator waits for *all* registrations — the bind-all barrier — before phase 2. **Phase 2 (connect).**
The coordinator broadcasts the address book; each worker then dials its outbound edges and accepts its
inbound ones. The barrier is what makes the connect deadlock-free: by the time anyone dials, every
destination listener exists, and because each worker dials every outbound before it accepts any inbound,
no central ordering of individual connections is needed.

This runs once, at startup, moving only these control messages. ``recv_event`` is the shared, fail-fast
receiver — it surfaces a worker that crashed (a bad exit code) instead of letting the coordinator block
forever on a peer that will never report. :func:`bind_barrier` drives the two phases over a
:class:`~nautilus.cluster.cohort.WorkerCohort`, so the bootstrap is independent of how the workers are
reached (local processes today, dialed daemons in Stage 4.2).
"""

from __future__ import annotations

import queue
import time
from typing import TYPE_CHECKING, Any

from nautilus.cluster.membership import AddressBook
from nautilus.cluster.protocol import Failed, Register

if TYPE_CHECKING:
    from nautilus.cluster.cohort import WorkerCohort


class WorkerError(RuntimeError):
    """A worker reported a :class:`Failed` — carries the child's traceback so ``deploy`` can re-raise it."""

    def __init__(self, worker_id: int, child_traceback: str) -> None:
        super().__init__(f"worker {worker_id} failed:\n{child_traceback}")
        self.worker_id = worker_id
        self.child_traceback = child_traceback


class WorkerCrashed(RuntimeError):
    """A worker process exited with a bad code without reporting (a hard crash, not a caught error)."""


def recv_event(events: Any, procs: list[Any], timeout: float | None, poll: float = 0.2) -> Any:
    """Return the next control message, or fail fast. Raises :class:`WorkerCrashed` if a worker exited
    with a non-zero code and sent nothing.

    ``timeout`` bounds *silence* and applies only where silence means a hang: the bootstrap, where every
    worker must register promptly. ``timeout=None`` waits indefinitely (only crash detection fires) — the
    right behavior awaiting completion, since a healthy job runs as long as its data does, so a silent but
    still-alive worker is busy, not stuck. (A genuinely hung-but-alive worker needs a heartbeat, out of
    scope here; a wall-clock cap on completion would falsely reap a healthy long-running job.)
    """
    start = time.monotonic()
    while True:
        try:
            return events.get(timeout=poll)
        except queue.Empty:
            if any(proc.exitcode not in (None, 0) for proc in procs):
                try:
                    return events.get(
                        timeout=1.0
                    )  # grace for a message still in flight from the feeder
                except queue.Empty:
                    proc = next(p for p in procs if p.exitcode not in (None, 0))
                    raise WorkerCrashed(
                        f"a worker exited with code {proc.exitcode} without reporting"
                    ) from None
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"no control message from any worker within {timeout:.0f}s"
                ) from None


def bind_barrier(cohort: WorkerCohort, num_workers: int, timeout: float) -> None:
    """Phase 1 → phase 2: collect every worker's :class:`Register`, then broadcast the address book over
    the ``cohort``. Every worker is watched throughout, since none has reported its ``Done`` yet; a worker
    that fails before binding surfaces as :class:`WorkerError`."""
    addresses: dict[int, tuple[str, int]] = {}
    while len(addresses) < num_workers:
        message = cohort.next_event(timeout, None)
        if isinstance(message, Register):
            addresses[message.worker_id] = (message.host, message.port)
        elif isinstance(message, Failed):
            raise WorkerError(message.worker_id, message.traceback)
        else:
            raise RuntimeError(f"unexpected control message during bootstrap: {message!r}")
    book = AddressBook(addresses)
    for worker_id in range(num_workers):
        cohort.send(worker_id, book)
