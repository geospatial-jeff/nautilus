"""Process/host resource sampling: each process samples its own resource usage.

Each process samples *itself*: a :class:`SystemSampler` periodically writes process CPU/memory/fd/thread
gauges and an event-loop-lag histogram into one dedicated process-level recorder (``operator_id="process"``,
``kind="process"``), registered in the same :class:`~nautilus.telemetry.registry.RecorderRegistry` as
every other recorder. So the readings flow through ``snapshot_all`` → ``build_report`` → JSON with no
special-casing, and per-record instrumentation is untouched.

``psutil`` is imported lazily and every call is individually guarded: if psutil is absent or a reading is
denied, that gauge is omitted (not zeroed) and sampling continues — event-loop lag needs no psutil and is
always recorded. This module is on the data-path side and must not import the report (boundary) layer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import perf_counter_ns
from typing import Any

from nautilus.telemetry.recorder import Recorder, TelemetryConfig, make_recorder


def make_system_recorder(config: TelemetryConfig, *, node: str = "local") -> Recorder:
    """A recorder dedicated to one process's resource samples (the sole writer is the SystemSampler)."""
    return make_recorder(
        operator_id="process", op_class="SystemSampler", kind="process", node=node, config=config
    )


def _guard(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception:
        pass  # a denied/absent reading is omitted, never fatal


class SystemSampler:
    """Periodically samples this process's resources into a recorder. The sole writer of that recorder."""

    def __init__(
        self,
        recorder: Recorder,
        *,
        node: str = "local",
        interval_micros: int = 500_000,
        host: bool = False,
        proc: Any = None,
        psutil_mod: Any = None,
    ) -> None:
        self._rec = recorder
        self._node = node
        self._interval_micros = interval_micros
        self._host = host
        self._proc = proc
        self._psutil = psutil_mod
        self._loaded = proc is not None or psutil_mod is not None

    def _load(self) -> None:
        """Lazily import psutil and prime CPU sampling. Never imported at OFF (the sampler isn't made)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            import psutil
        except Exception:
            return
        self._psutil = psutil
        try:
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)  # prime: the first call always returns 0.0
            if self._host:
                psutil.cpu_percent(None)
        except Exception:
            self._proc = None

    def sample_once(self, *, loop_lag_micros: int | None = None, sample_cpu: bool = True) -> None:
        """One fully-guarded sample. Records loop lag if given, plus whatever psutil readings succeed."""
        rec = self._rec
        if loop_lag_micros is not None:
            rec.observe("runtime.loop_lag_micros", int(loop_lag_micros))
        self._load()
        proc = self._proc
        if proc is not None:
            _guard(lambda: rec.set_gauge("process.rss_bytes", float(proc.memory_info().rss)))
            _guard(lambda: rec.set_gauge("process.num_threads", float(proc.num_threads())))
            _guard(lambda: rec.set_gauge("process.num_fds", float(proc.num_fds())))
            if sample_cpu:
                _guard(lambda: rec.set_gauge("process.cpu_percent", float(proc.cpu_percent(None))))
        if self._host and self._psutil is not None and sample_cpu:
            ps = self._psutil
            _guard(lambda: rec.set_gauge("host.cpu_percent", float(ps.cpu_percent(None))))
            _guard(lambda: rec.set_gauge("host.mem_percent", float(ps.virtual_memory().percent)))

    async def run(self) -> None:
        """Sample on a fixed cadence until cancelled. Created outside the data TaskGroup so it can never
        delay job completion or cancel sibling data tasks."""
        self._load()
        interval_s = self._interval_micros / 1_000_000
        while True:
            t0 = perf_counter_ns()
            await asyncio.sleep(interval_s)
            elapsed_micros = (perf_counter_ns() - t0) // 1000
            lag = max(0, elapsed_micros - self._interval_micros)
            self.sample_once(loop_lag_micros=lag, sample_cpu=True)
