"""Process and host resource sampling: each process samples itself.

A :class:`SystemSampler` periodically writes process CPU/memory/fd/thread gauges, host CPU/memory and
network gauges, an event-loop-lag histogram, and — at the FULL tier — a GIL-contention gauge into one
dedicated recorder (``operator_id="process"``), which the registry snapshots like any other, so the
readings reach the report with no special-casing.

``psutil`` (host/process resources) and ``gilknocker`` (GIL contention) are imported lazily and every
call is guarded: if a library is absent or a reading is denied, that gauge is omitted (not zeroed) and
sampling continues. Event-loop lag needs neither library and is always recorded. GIL contention is the
one continuous cost — gilknocker runs a monitor thread — so it is started only when the run is at the
FULL tier. This module is on the data path and must not import the report layer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import perf_counter_ns
from typing import Any

from nautilus.telemetry.recorder import Recorder, TelemetryConfig, make_recorder


def make_system_recorder(config: TelemetryConfig, *, node: str = "local") -> Recorder:
    """A recorder for one process's resource samples. The SystemSampler is its only writer once running;
    the executor records the one-time ``placement.instances_per_worker`` gauge on it before sampling
    starts, so there is never a concurrent writer."""
    return make_recorder(
        operator_id="process", op_class="SystemSampler", kind="process", node=node, config=config
    )


def _guard(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception:
        pass  # a denied/absent reading is omitted, never fatal


#: gilknocker's monitor-thread polling cadence — fine enough to catch contention, cheap enough to leave
#: on, and only ever started at the FULL tier.
_GIL_POLL_MICROS = 1000


class SystemSampler:
    """Periodically samples this process's resources into a recorder. The only writer once its task runs
    (see :func:`make_system_recorder` for the one-time placement gauge written before it starts)."""

    def __init__(
        self,
        recorder: Recorder,
        *,
        interval_micros: int = 500_000,
        host: bool = False,
        enable_gil: bool = False,
        proc: Any = None,
        psutil_mod: Any = None,
        gil_monitor: Any = None,
    ) -> None:
        # node is not stored here: the recorder already carries it (make_system_recorder), and every
        # sample is written through that recorder, so the sampler never needs node of its own.
        self._rec = recorder
        self._interval_micros = interval_micros
        self._host = host
        self._enable_gil = enable_gil
        self._proc = proc
        self._psutil = psutil_mod
        self._gil = gil_monitor
        # Previous cumulative (bytes_sent, bytes_recv), so a periodic sample can emit the delta since it.
        self._prev_net: tuple[int, int] | None = None
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
        except Exception:
            self._proc = None
        if self._host:
            # Prime host CPU independently of the process probe, so a host-reading failure never disables
            # process sampling (host=True is now on for every worker).
            _guard(lambda: psutil.cpu_percent(None))
        if self._enable_gil and self._gil is None:
            _guard(
                self._start_gil
            )  # a monitor thread; only started at FULL (see the module docstring)

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
        if self._host and self._psutil is not None:
            ps = self._psutil
            # mem_percent is a point reading — valid on the final teardown sample (like rss). cpu_percent
            # and the network deltas are interval readings, so they are taken only on periodic ticks
            # (sample_cpu); a bare teardown call would return a meaningless instant.
            _guard(lambda: rec.set_gauge("host.mem_percent", float(ps.virtual_memory().percent)))
            if sample_cpu:
                _guard(lambda: rec.set_gauge("host.cpu_percent", float(ps.cpu_percent(None))))
                _guard(lambda: self._sample_net(ps))
        if self._gil is not None and sample_cpu:
            _guard(
                self._sample_gil
            )  # GIL contention over the interval, then reset for the next one

    async def run(self) -> None:
        """Sample on a fixed cadence until cancelled. Created outside the data TaskGroup so it can never
        delay job completion or cancel sibling data tasks."""
        self._load()
        interval_s = self._interval_micros / 1_000_000
        try:
            while True:
                t0 = perf_counter_ns()
                await asyncio.sleep(interval_s)
                elapsed_micros = (perf_counter_ns() - t0) // 1000
                lag = max(0, elapsed_micros - self._interval_micros)
                self.sample_once(loop_lag_micros=lag, sample_cpu=True)
        finally:
            self.close()  # stop the GIL monitor thread on cancellation (the normal teardown path)

    def _start_gil(self) -> None:
        """Start a gilknocker monitor thread (FULL tier only). Guarded: absent gilknocker leaves it off."""
        from gilknocker import KnockKnock

        monitor = KnockKnock(polling_interval_micros=_GIL_POLL_MICROS)
        monitor.start()
        self._gil = monitor

    def _sample_net(self, ps: Any) -> None:
        """Emit bytes sent/received across all host NICs since the previous sample. The first sample only
        primes the baseline (no prior reading to diff), so nothing is emitted until the second."""
        io = ps.net_io_counters()
        cur = (int(io.bytes_sent), int(io.bytes_recv))
        prev = self._prev_net
        self._prev_net = cur
        if prev is not None:  # max(0, ...) guards a counter reset/rollover between samples
            self._rec.set_gauge("host.net_bytes_sent", float(max(0, cur[0] - prev[0])))
            self._rec.set_gauge("host.net_bytes_recv", float(max(0, cur[1] - prev[1])))

    def _sample_gil(self) -> None:
        """Record GIL contention over the interval (0..1 → percent), then reset the monitor's counter."""
        monitor = self._gil
        self._rec.set_gauge("runtime.gil_percent", float(monitor.contention_metric) * 100.0)
        monitor.reset_contention_metric()

    def close(self) -> None:
        """Stop the GIL monitor thread if one is running. Idempotent and fully guarded, so the sampler
        task's teardown can call it and a second call (or one with no monitor) is a no-op."""
        monitor = self._gil
        if monitor is not None:
            self._gil = None
            _guard(monitor.stop)
