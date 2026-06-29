"""Stage 4.0: the WorkerCohort seam over the local backend.

LocalCohort must reproduce recv_event's fail-fast policy *through the seam*: narrow liveness to the
watched workers, return a busy worker's message, fire a finite timeout on silence, and surface a crash
(bad exit code) without hanging — and deliver a sent message to exactly one worker. These drive the
cohort with fakes, no real processes, so they pin the contract deploy relies on.
"""

from __future__ import annotations

import queue

import pytest

from nautilus.cluster.cohort import LocalCohort
from nautilus.cluster.rendezvous import WorkerCrashed


class _FakeProc:
    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode


def _cohort(
    events: queue.Queue,
    procs: list[_FakeProc],
    commands: dict[int, queue.Queue] | None = None,
) -> LocalCohort:
    return LocalCohort(procs, events, commands or {})


def test_next_event_returns_a_busy_workers_message() -> None:
    events: queue.Queue = queue.Queue()
    events.put("done")
    assert _cohort(events, [_FakeProc(None)]).next_event(None, {0}) == "done"


def test_next_event_detects_a_crash_of_a_watched_worker() -> None:
    events: queue.Queue = queue.Queue()  # empty; worker 1 exited without reporting
    cohort = _cohort(events, [_FakeProc(None), _FakeProc(1)])
    with pytest.raises(WorkerCrashed):
        cohort.next_event(None, {1})


def test_next_event_ignores_a_crash_outside_the_watch_set() -> None:
    # A worker whose Done was already received leaves the watch set; its later non-zero exit must not fail
    # an otherwise-complete run. With only worker 0 watched, worker 1's bad exit is invisible.
    events: queue.Queue = queue.Queue()
    events.put("done")
    cohort = _cohort(events, [_FakeProc(None), _FakeProc(1)])
    assert cohort.next_event(None, {0}) == "done"


def test_next_event_finite_timeout_fires_on_silence() -> None:
    events: queue.Queue = queue.Queue()
    with pytest.raises(TimeoutError):
        _cohort(events, [_FakeProc(None)]).next_event(0.3, {0})


def test_send_delivers_to_exactly_one_worker() -> None:
    q0: queue.Queue = queue.Queue()
    q1: queue.Queue = queue.Queue()
    cohort = _cohort(queue.Queue(), [_FakeProc(None), _FakeProc(None)], {0: q0, 1: q1})
    cohort.send(1, "book")
    assert q1.get_nowait() == "book"
    assert q0.empty()
