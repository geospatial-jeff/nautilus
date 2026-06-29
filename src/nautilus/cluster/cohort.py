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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nautilus.cluster.launcher import reap
from nautilus.cluster.rendezvous import recv_event


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


__all__ = ["WorkerCohort", "LocalCohort"]
