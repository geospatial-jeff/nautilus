"""A Sink consumes an assembled :class:`RunReport` at the job boundary.

Instrumentation never touches a Sink — it only writes to a recorder — so adding a reader of the
telemetry is purely additive. The live HTTP dashboard is one such reader, though it pulls
``RecorderRegistry.snapshot_all()`` mid-run rather than receiving a finished report through a Sink.
Stage 0 ships the no-op default and a buffering sink for tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nautilus.telemetry.report.report import RunReport


class Sink(ABC):
    @abstractmethod
    def emit_report(self, report: RunReport) -> None: ...


class NullSink(Sink):
    """The default: do nothing. The report is still returned in-process via ``RunResult.telemetry``."""

    def emit_report(self, report: RunReport) -> None:
        pass


class BufferSink(Sink):
    """Captures emitted reports in memory (tests / programmatic use)."""

    def __init__(self) -> None:
        self.reports: list[RunReport] = []

    def emit_report(self, report: RunReport) -> None:
        self.reports.append(report)
