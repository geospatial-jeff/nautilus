"""Stage 1: hardware sampling flows through the existing seam, degrades gracefully, and never
changes a run's structural identity."""

import sys

import pytest

from nautilus.core.records import EOS_FRAME
from nautilus.core.time import TestClock
from nautilus.driver.local import run_local_chain
from nautilus.operators import InMemorySource, KeyedCount, MapBatch, Tokenize
from nautilus.telemetry.catalog import STRUCTURAL_METRICS, Tier
from nautilus.telemetry.recorder import InstanceRecorder, TelemetryConfig
from nautilus.telemetry.system import SystemSampler
from nautilus.testing import data

WORDS = [data(line=["the cat sat", "the dog ran"]), EOS_FRAME]


class _FakeMem:
    rss = 123_456


class _FakeProc:
    def memory_info(self):
        return _FakeMem()

    def num_threads(self):
        return 7

    def num_fds(self):
        return 11

    def cpu_percent(self, interval):
        return 42.0


class _FakeVM:
    percent = 55.5


class _FakeNetIO:
    def __init__(self, sent, recv):
        self.bytes_sent = sent
        self.bytes_recv = recv


class _FakePsutil:
    """Stands in for the psutil module for host sampling: cpu_percent, virtual_memory, net counters."""

    def __init__(self):
        self._net_calls = 0

    def cpu_percent(self, interval):
        return 88.0

    def virtual_memory(self):
        return _FakeVM()

    def net_io_counters(self):
        self._net_calls += (
            1  # rising cumulative counters, so the second sample yields a positive delta
        )
        return _FakeNetIO(1000 * self._net_calls, 400 * self._net_calls)


class _FakeGil:
    """Stands in for a gilknocker KnockKnock monitor: a contention fraction that can be read and reset."""

    def __init__(self, contention=0.42):
        self.contention_metric = contention
        self.reset_calls = 0
        self.stopped = False

    def reset_contention_metric(self):
        self.reset_calls += 1

    def stop(self):
        self.stopped = True


def _gauges(snap):
    return {name: last for (name, _labels), (last, _mn, _mx) in snap.gauges.items()}


def _proc_recorder(tier=Tier.COUNTERS):
    return InstanceRecorder(
        operator_id="process",
        op_class="SystemSampler",
        kind="process",
        config=TelemetryConfig(tier=tier, clock=TestClock(0)),
    )


def test_sample_once_writes_gauges_and_loop_lag():
    rec = _proc_recorder()
    SystemSampler(rec, proc=_FakeProc()).sample_once(loop_lag_micros=250)
    snap = rec.snapshot()
    gauges = {name: last for (name, _l), (last, _mn, _mx) in snap.gauges.items()}
    assert gauges["process.rss_bytes"] == 123_456.0
    assert gauges["process.num_threads"] == 7.0
    assert gauges["process.cpu_percent"] == 42.0
    ((_, hist),) = [(k, v) for k, v in snap.histograms.items() if k[0] == "runtime.loop_lag_micros"]
    assert hist.count == 1 and hist.sum == 250


def test_loop_lag_recorded_even_when_psutil_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` raise
    rec = _proc_recorder()
    SystemSampler(rec).sample_once(loop_lag_micros=99)
    snap = rec.snapshot()
    assert not snap.gauges  # no psutil gauges
    assert any(k[0] == "runtime.loop_lag_micros" for k in snap.histograms)  # but lag still recorded


def test_denied_readings_are_omitted_not_fatal():
    class _Denied:
        def memory_info(self):
            raise PermissionError("denied")

        def num_threads(self):
            raise PermissionError("denied")

        def num_fds(self):
            raise PermissionError("denied")

        def cpu_percent(self, interval):
            raise PermissionError("denied")

    rec = _proc_recorder()
    SystemSampler(rec, proc=_Denied()).sample_once(loop_lag_micros=5)  # must not raise
    snap = rec.snapshot()
    assert not snap.gauges
    assert any(k[0] == "runtime.loop_lag_micros" for k in snap.histograms)


def test_system_metrics_are_excluded_from_structural_digest():
    system_names = {
        "process.cpu_percent",
        "process.rss_bytes",
        "process.num_fds",
        "process.num_threads",
        "host.cpu_percent",
        "host.mem_percent",
        "host.net_bytes_sent",
        "host.net_bytes_recv",
        "runtime.loop_lag_micros",
        "runtime.gil_percent",
    }
    assert system_names.isdisjoint(STRUCTURAL_METRICS)


async def test_digest_identical_with_sampling_on_and_off():
    src = InMemorySource(list(WORDS))
    on = await run_local_chain(
        src,
        [Tokenize("line", "word"), KeyedCount("word")],
        clock=TestClock(),
        telemetry=TelemetryConfig(sample_system=True, clock=TestClock()),
    )
    src = InMemorySource(list(WORDS))
    off = await run_local_chain(
        src,
        [Tokenize("line", "word"), KeyedCount("word")],
        clock=TestClock(),
        telemetry=TelemetryConfig(sample_system=False, clock=TestClock()),
    )
    assert on.telemetry.structural_digest() == off.telemetry.structural_digest()


async def test_hardware_row_present_with_sampling_absent_when_off():
    on = await run_local_chain(
        InMemorySource(list(WORDS)),
        [Tokenize("line", "word")],
        telemetry=TelemetryConfig(sample_system=True, clock=TestClock()),
    )
    proc = on.telemetry.operator("process")
    assert proc is not None
    assert any(g.name == "process.rss_bytes" for g in proc.gauges)
    # ...and it stays OUT of the dataflow operator ranking
    assert "process" not in {s.operator_id for s in on.telemetry.by_self_time()}

    off = await run_local_chain(
        InMemorySource(list(WORDS)),
        [Tokenize("line", "word")],
        telemetry=TelemetryConfig(tier=Tier.OFF),
    )
    assert off.telemetry.operator("process") is None


def test_sample_once_records_host_metrics_when_enabled():
    rec = _proc_recorder()
    SystemSampler(rec, proc=_FakeProc(), psutil_mod=_FakePsutil(), host=True).sample_once(
        loop_lag_micros=1
    )
    gauges = _gauges(rec.snapshot())
    assert gauges["host.cpu_percent"] == 88.0
    assert gauges["host.mem_percent"] == 55.5
    assert gauges["process.num_fds"] == 11.0  # process resources recorded alongside the host ones


def test_host_mem_taken_on_teardown_sample_but_host_cpu_is_not():
    # The guaranteed final reading passes sample_cpu=False. host.mem_percent is a point value and is
    # still recorded (like process.rss_bytes); the interval-based cpu gauges are skipped.
    rec = _proc_recorder()
    SystemSampler(rec, proc=_FakeProc(), psutil_mod=_FakePsutil(), host=True).sample_once(
        sample_cpu=False
    )
    gauges = _gauges(rec.snapshot())
    assert gauges["host.mem_percent"] == 55.5
    assert "host.cpu_percent" not in gauges
    assert "process.cpu_percent" not in gauges


def test_no_host_metrics_when_host_disabled():
    rec = _proc_recorder()
    SystemSampler(rec, proc=_FakeProc(), psutil_mod=_FakePsutil()).sample_once(loop_lag_micros=1)
    gauges = _gauges(rec.snapshot())
    assert "host.cpu_percent" not in gauges and "host.mem_percent" not in gauges


async def test_host_metric_present_on_process_row():
    # An end-to-end run enables host=True; the guaranteed teardown reading records host.mem_percent on
    # the process row even on a run too short for a periodic (interval) sample to fire.
    on = await run_local_chain(
        InMemorySource(list(WORDS)),
        [Tokenize("line", "word")],
        telemetry=TelemetryConfig(sample_system=True, clock=TestClock()),
    )
    proc = on.telemetry.operator("process")
    assert proc is not None
    assert any(g.name == "host.mem_percent" for g in proc.gauges)


def test_net_delta_recorded_across_two_samples():
    rec = _proc_recorder()
    sampler = SystemSampler(rec, proc=_FakeProc(), psutil_mod=_FakePsutil(), host=True)
    sampler.sample_once(loop_lag_micros=1)  # primes the baseline (net_io_counters call #1)
    sampler.sample_once(loop_lag_micros=1)  # call #2 → the delta since the baseline is emitted
    gauges = _gauges(rec.snapshot())
    assert gauges["host.net_bytes_sent"] == 1000.0
    assert gauges["host.net_bytes_recv"] == 400.0


def test_net_not_recorded_on_teardown_sample():
    # Network deltas are interval readings, so the final sample_cpu=False teardown reading skips them.
    rec = _proc_recorder()
    sampler = SystemSampler(rec, proc=_FakeProc(), psutil_mod=_FakePsutil(), host=True)
    sampler.sample_once(loop_lag_micros=1)  # prime
    sampler.sample_once(sample_cpu=False)
    assert "host.net_bytes_sent" not in _gauges(rec.snapshot())


def test_gil_percent_recorded_from_monitor_at_full_tier():
    rec = _proc_recorder(Tier.FULL)  # runtime.gil_percent is a FULL-tier metric
    gil = _FakeGil(0.42)
    SystemSampler(rec, proc=_FakeProc(), psutil_mod=_FakePsutil(), gil_monitor=gil).sample_once(
        loop_lag_micros=1
    )
    assert _gauges(rec.snapshot())["runtime.gil_percent"] == 42.0  # 0.42 fraction → percent
    assert gil.reset_calls == 1  # read-then-reset, so each sample measures exactly one interval


def test_gil_percent_dropped_below_full_tier():
    rec = _proc_recorder(Tier.COUNTERS)  # below FULL the gauge resolves to a no-op
    SystemSampler(
        rec, proc=_FakeProc(), psutil_mod=_FakePsutil(), gil_monitor=_FakeGil()
    ).sample_once(loop_lag_micros=1)
    assert "runtime.gil_percent" not in _gauges(rec.snapshot())


def test_gil_not_read_on_teardown_sample():
    rec = _proc_recorder(Tier.FULL)
    gil = _FakeGil(0.9)
    SystemSampler(rec, proc=_FakeProc(), psutil_mod=_FakePsutil(), gil_monitor=gil).sample_once(
        sample_cpu=False
    )
    assert "runtime.gil_percent" not in _gauges(rec.snapshot())
    assert gil.reset_calls == 0


def test_close_stops_the_gil_monitor():
    gil = _FakeGil()
    sampler = SystemSampler(_proc_recorder(), gil_monitor=gil)
    sampler.close()
    assert gil.stopped
    sampler.close()  # idempotent: a second call is a no-op


def test_real_gilknocker_contention_recorded_if_installed():
    pytest.importorskip("gilknocker")
    rec = _proc_recorder(Tier.FULL)
    sampler = SystemSampler(
        rec, host=True, enable_gil=True
    )  # real load starts a real monitor thread
    try:
        total = 0
        for i in range(3_000_000):  # hold the GIL, so the monitor observes contention
            total += i
        sampler.sample_once(sample_cpu=True)
    finally:
        sampler.close()
    assert "runtime.gil_percent" in _gauges(
        rec.snapshot()
    )  # the real gilknocker API produced a reading


async def test_sampler_does_not_block_fail_fast():
    def _boom(_batch):
        raise ValueError("boom")

    with pytest.raises(BaseException):  # noqa: B017 - ExceptionGroup or ValueError, both fail-fast
        await run_local_chain(
            InMemorySource(list(WORDS)),
            [MapBatch(_boom)],
            telemetry=TelemetryConfig(sample_system=True, clock=TestClock()),
        )
