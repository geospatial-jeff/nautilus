"""The recorder registry — created at the job boundary, holds references to every actor's recorder.

It lives in the data-path layer but never writes on the hot path: it only *collects* recorders so the
boundary can read them. ``snapshot_all()`` is a point-in-time read that MUST run on the owning event
loop thread — ``InstanceRecorder.snapshot`` iterates the live instrument dicts, which the writer loop
mutates mid-run (new label keys, the events ring buffer), so a *cross-thread* read would race. The live
server therefore schedules onto the loop thread (``run_coroutine_threadsafe``) to call this between actor steps,
rather than reading it from its HTTP thread — so adding the live endpoint needs no instrumentation change.
"""

from __future__ import annotations

from nautilus.telemetry.model import InstanceSnapshot
from nautilus.telemetry.recorder import NULL_RECORDER, Recorder


class RecorderRegistry:
    def __init__(self) -> None:
        self._recorders: list[Recorder] = []

    def register(self, recorder: Recorder) -> Recorder:
        """Register a recorder and return it (so call sites read fluently). NULL recorders are skipped."""
        if recorder is not NULL_RECORDER:
            self._recorders.append(recorder)
        return recorder

    def snapshot_all(self) -> list[InstanceSnapshot]:
        """Point-in-time snapshot of every registered recorder. Safe to call mid-run."""
        return [r.snapshot() for r in self._recorders]
