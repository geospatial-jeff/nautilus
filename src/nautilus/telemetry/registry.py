"""The recorder registry — created at the job boundary; holds a reference to every actor's recorder.

It only collects recorders so the boundary can read them; it never writes on the hot path.
``snapshot_all()`` must run on the owning event-loop thread, because ``InstanceRecorder.snapshot`` reads
instrument dicts the writer mutates mid-run — a cross-thread read would race. The live server hops onto
the loop thread (``run_coroutine_threadsafe``) to call it between actor steps.
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
