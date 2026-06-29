"""Stage 2d: recv_event's fail-fast policy distinguishes a hang from a busy worker from a crash.

During bootstrap, silence means a hang, so a finite timeout fires. Awaiting completion, silence means a
healthy worker is busy (it reports only when its slice finishes), so ``timeout=None`` waits — but a hard
crash (bad exit code) is still caught, so completion never hangs on a dead worker.
"""

from __future__ import annotations

import queue

import pytest

from nautilus.cluster.rendezvous import WorkerCrashed, recv_event


class _FakeProc:
    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode


def test_finite_timeout_fires_on_silence_during_bootstrap() -> None:
    events: queue.Queue = queue.Queue()  # empty; the worker never registers
    with pytest.raises(TimeoutError):
        recv_event(events, [_FakeProc(None)], timeout=0.3)


def test_none_timeout_waits_for_a_busy_worker_instead_of_aborting() -> None:
    events: queue.Queue = queue.Queue()
    events.put("done")  # a busy worker that has now reported
    assert recv_event(events, [_FakeProc(None)], None) == "done"


def test_none_timeout_still_detects_a_crash() -> None:
    # Blocking for completion must not hang on a dead worker: a bad exit code is caught even with no
    # wall-clock deadline.
    events: queue.Queue = queue.Queue()  # empty; the worker exited without reporting
    with pytest.raises(WorkerCrashed):
        recv_event(events, [_FakeProc(1)], None)
