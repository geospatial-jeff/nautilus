"""The WorkerCohort seam, both backends, driven without real processes or daemons.

LocalCohort must reproduce recv_event's fail-fast policy *through the seam*: narrow liveness to the
watched workers, return a busy worker's message, fire a finite timeout on silence, and surface a crash
(bad exit code) without hanging — and deliver a sent message to exactly one worker.

RemoteCohort must do the same over TCP, where there is no exit code: a control connection closing before a
worker's Done is a crash. Critically, the completion wait uses ``timeout=None``, so a dead peer must be
detected by the connection dropping — not block forever — which the socketpair tests below pin.
"""

from __future__ import annotations

import queue
import socket

import pytest

from nautilus.cluster.cohort import LocalCohort, RemoteCohort, _DaemonConn
from nautilus.cluster.control_link import encode
from nautilus.cluster.protocol import Register
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


# --- RemoteCohort: crash detection over a real socket pair, no daemon needed --------------------


def test_remote_next_event_returns_a_framed_message() -> None:
    ours, theirs = socket.socketpair()
    cohort = RemoteCohort({0: _DaemonConn(ours)})
    theirs.sendall(encode(Register(0, "worker-0", 9000)))
    try:
        assert cohort.next_event(None, {0}) == Register(0, "worker-0", 9000)
    finally:
        ours.close()
        theirs.close()


def test_remote_next_event_detects_eof_before_done_as_crash() -> None:
    # The daemon's control connection closing before a Done is a crash. With timeout=None the wait must
    # still return — crash detection fires on the EOF rather than blocking the coordinator forever.
    ours, theirs = socket.socketpair()
    cohort = RemoteCohort({0: _DaemonConn(ours)})
    theirs.close()  # the daemon vanished without reporting
    try:
        with pytest.raises(WorkerCrashed):
            cohort.next_event(None, {0})
    finally:
        ours.close()


def test_remote_next_event_ignores_eof_outside_the_watch_set() -> None:
    # Worker 1 has reported and left the watch set; its later close must not fail an otherwise-complete
    # run. Watching only worker 0, worker 1's EOF is invisible and worker 0's message comes through.
    o0, t0 = socket.socketpair()
    o1, t1 = socket.socketpair()
    cohort = RemoteCohort({0: _DaemonConn(o0), 1: _DaemonConn(o1)})
    t0.sendall(encode(Register(0, "worker-0", 9000)))
    t1.close()  # worker 1 closed, but it is not watched
    try:
        assert cohort.next_event(None, {0}) == Register(0, "worker-0", 9000)
    finally:
        for s in (o0, t0, o1):
            s.close()
