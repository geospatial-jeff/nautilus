"""The benchmark harness: robust statistics, noise-aware comparison, measurement, and baseline I/O."""

import pytest

from nautilus import bench
from nautilus.bench import BenchResult, Environment, Stats
from nautilus.telemetry.catalog import Tier


def test_summarize_uses_median_and_iqr():
    s = bench.summarize([300, 100, 500, 200, 400])  # unsorted on purpose
    assert s.median == 300
    assert s.iqr == 200  # inclusive quartiles of 1..5 hundreds: q1=200, q3=400
    assert s.rel_spread == pytest.approx(200 / 300)
    assert (s.min, s.max) == (100, 500)


def test_summarize_single_sample_has_zero_spread():
    s = bench.summarize([1234.0])
    assert s.median == 1234 and s.iqr == 0 and s.rel_spread == 0


def _result(median, *, spread=0.02, digest="A", platform="linux", processor="cpu", pipeline="p"):
    stats = Stats((median,), median, median * spread, spread, median, median)
    env = Environment("0.0.1", "3.12", platform, processor, "abc1234")
    scale = {
        "rows": 1,
        "batch": 1,
        "keys": 1,
        "parallelism": 1,
        "workers": 1,
        "tier": 1,
    }
    return BenchResult(pipeline, scale, 1, stats, digest, True, env, "")


def test_compare_flags_a_regression_beyond_the_noise():
    c = bench.compare(_result(1000), _result(800))  # -20%, noise 2% -> threshold 7%
    assert c.status == "REGRESSED" and c.delta == pytest.approx(-0.2)
    assert bench.is_failure(c.status)


def test_compare_flags_an_improvement():
    c = bench.compare(_result(1000), _result(1200))
    assert c.status == "IMPROVED" and not bench.is_failure(c.status)


def test_compare_calls_a_small_change_unchanged():
    c = bench.compare(_result(1000), _result(1030))  # +3% < 7% floor
    assert c.status == "unchanged"


def test_compare_refuses_a_win_below_twice_the_noise():
    # +15% looks like a win, but with 10% run-to-run noise the bar is 2x10% = 20%, so it does not count.
    c = bench.compare(_result(1000, spread=0.10), _result(1150, spread=0.10))
    assert c.threshold == pytest.approx(0.20)
    assert c.status == "unchanged"


def test_compare_output_change_always_fails_even_on_a_speedup():
    c = bench.compare(_result(1000, digest="A"), _result(2000, digest="B"))
    assert c.status == "OUTPUT-CHANGED" and bench.is_failure(c.status)


def test_compare_does_not_judge_perf_across_machines_but_still_checks_output():
    same_output = bench.compare(_result(1000, platform="linux"), _result(500, platform="darwin"))
    assert same_output.status == "machine-differs" and not bench.is_failure(same_output.status)


def test_confirm_regression_clears_a_transient_slow_run():
    # First run dips 15% (a slow machine state); a retry hits baseline speed, so the fastest kept reads as
    # unchanged — the noisy-neighbour dip does not become a false failure.
    best, cmp, used = bench.confirm_regression(
        _result(1000), _result(850), lambda: _result(1000), min_threshold=0.10
    )
    assert used == 1 and cmp.status == "unchanged"
    assert best.throughput_rows_per_sec.median == 1000


def test_confirm_regression_still_fails_a_real_regression():
    # Every re-measure stays slow (the code really is slower), so it exhausts the retries and still fails —
    # keeping the fastest cannot lift throughput past what the code allows.
    best, cmp, used = bench.confirm_regression(
        _result(1000), _result(850), lambda: _result(860), min_threshold=0.10, retries=3
    )
    assert used == 3 and cmp.status == "REGRESSED" and bench.is_failure(cmp.status)
    assert best.throughput_rows_per_sec.median == 860  # the fastest of the slow runs


def test_confirm_regression_never_re_measures_a_pass():
    calls = 0

    def remeasure():
        nonlocal calls
        calls += 1
        return _result(9999)

    _best, cmp, used = bench.confirm_regression(
        _result(1000), _result(980), remeasure, min_threshold=0.10
    )
    assert used == 0 and calls == 0 and cmp.status == "unchanged"
    changed_output = bench.compare(
        _result(1000, platform="linux", digest="A"), _result(1000, platform="darwin", digest="B")
    )
    assert changed_output.status == "OUTPUT-CHANGED"  # correctness is machine-independent


def test_compare_treats_a_different_cpu_as_machine_differs():
    # The GitHub-hosted-runner case: same OS image, different physical CPU. A big throughput drop is not
    # comparable across CPUs, so it is machine-differs (never a false regression) — the whole point of the
    # CPU-aware gate. The digest is still checked regardless of CPU.
    drop = bench.compare(_result(1000, processor="Xeon-8370C"), _result(600, processor="EPYC-7763"))
    assert drop.status == "machine-differs" and not bench.is_failure(drop.status)
    changed = bench.compare(
        _result(1000, processor="Xeon-8370C", digest="A"),
        _result(1000, processor="EPYC-7763", digest="B"),
    )
    assert changed.status == "OUTPUT-CHANGED"  # correctness is CPU-independent


def test_compare_gates_throughput_on_the_same_cpu():
    # Same platform and CPU -> a real regression is judged, not excused.
    c = bench.compare(_result(1000, processor="Xeon-8370C"), _result(700, processor="Xeon-8370C"))
    assert c.status == "REGRESSED" and bench.is_failure(c.status)


def test_measure_reduces_trials_to_a_stable_deterministic_result():
    env = Environment(
        "0.0.1", "3.12", "test", "cpu", None
    )  # fixed env: no git subprocess in the test
    r = bench.measure("wordcount", trials=3, warmup=1, tier=Tier.COUNTERS, environment=env)
    assert (
        r.deterministic and r.structural_digest
    )  # bounded pipeline -> identical results each trial
    assert r.throughput_rows_per_sec.median > 0
    assert len(r.throughput_rows_per_sec.samples) == 3


def test_measure_requires_telemetry_for_the_digest_anchor():
    with pytest.raises(ValueError, match="tier"):
        bench.measure("wordcount", trials=2, tier=Tier.OFF)


def test_bench_join_is_a_graph_pipeline_run_through_the_harness(monkeypatch):
    # bench-join has two sources, so the harness must run it via run_plan (the graph path), not the linear
    # (source, transforms) one — and still produce the 1:1 stream-table result. At parallelism 2 the keyed
    # shuffle co-partitions both sides, so the output row count is identical to the serial run.
    from nautilus.pipelines import is_graph_pipeline

    assert is_graph_pipeline("bench-join")
    for k, v in {
        "NAUTILUS_BENCH_ROWS": "8192",
        "NAUTILUS_BENCH_BATCH": "4096",
        "NAUTILUS_BENCH_KEYS": "100",
    }.items():
        monkeypatch.setenv(k, v)
    serial = bench.run_once("bench-join", parallelism=1, workers=1, capacity=16, tier=Tier.COUNTERS)
    parallel = bench.run_once(
        "bench-join", parallelism=2, workers=1, capacity=16, tier=Tier.COUNTERS
    )
    serial_rows = sum(b.num_rows for b in serial)
    assert serial_rows > 0
    assert sum(b.num_rows for b in parallel) == serial_rows


def test_measure_handles_a_graph_pipeline():
    # the full median-of-trials path (measure -> run_once -> run_plan) works on a graph pipeline and the
    # join's result is deterministic across trials.
    env = Environment("0.0.1", "3.12", "test", "cpu", None)
    r = bench.measure(
        "bench-join",
        rows=8192,
        batch=4096,
        keys=100,
        trials=2,
        warmup=0,
        tier=Tier.COUNTERS,
        environment=env,
    )
    assert r.deterministic and r.throughput_rows_per_sec.median > 0


def test_baseline_round_trips_through_json(tmp_path):
    path = tmp_path / "baseline.json"
    original = {"p": _result(12345.0, digest="deadbeef")}
    bench.save_baseline(path, original)
    loaded = bench.load_baseline(path)
    assert (
        loaded == original
    )  # frozen dataclasses compare by value; samples survive the JSON tuple/list


def test_compare_is_symmetric_about_the_threshold_definition():
    # the bar is max(floor, 2*noise) using the larger of the two runs' noise
    c = bench.compare(_result(1000, spread=0.01), _result(900, spread=0.30))
    assert c.threshold == pytest.approx(0.60) and c.status == "unchanged"  # -10% < 60% bar
