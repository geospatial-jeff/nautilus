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


def _result(median, *, spread=0.02, digest="A", platform="linux", pipeline="p"):
    stats = Stats((median,), median, median * spread, spread, median, median)
    env = Environment("0.0.1", "3.12", platform, "cpu", "abc1234")
    scale = {
        "rows": 1,
        "batch": 1,
        "keys": 1,
        "wm_every": 1,
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
    changed_output = bench.compare(
        _result(1000, platform="linux", digest="A"), _result(1000, platform="darwin", digest="B")
    )
    assert changed_output.status == "OUTPUT-CHANGED"  # correctness is machine-independent


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
