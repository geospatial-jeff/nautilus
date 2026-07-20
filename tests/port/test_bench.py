"""Port-fidelity anchors for the benchmark sources and harness (Tier 3).

These pin the *inputs* a cross-language structural digest depends on — the exact draws of the
fixed-seed synthetic sources, the region-tagger's map contract, and the harness's byte layout and
comparison rules — so a future Python->Rust rewrite that reproduces them is provably feeding the same
data and classifying results the same way. Every golden here (key lists, null-index sets, value lists,
grid values, baseline bytes, the one structural-digest hex) was derived by running the current code and
hardcoding the actual output; a divergence in the rewrite is a real behavior change, not a stale test.

The digest itself is exercised once, on a tiny ``bench-keyed`` run, because reproducing a full digest
cross-language is the ultimate fidelity check; the rest anchor the deterministic inputs that feed it.
"""

import asyncio
import json

import numpy as np
import pyarrow as pa
import pytest

from nautilus import bench
from nautilus.bench import BenchResult, Environment, Stats, result_from_dict
from nautilus.benchmarks import (
    SyntheticGridSource,
    SyntheticKeyedSource,
    make_region_tagger,
)
from nautilus.core.records import Batch
from nautilus.telemetry.catalog import Tier

# --- helpers (match the style of tests/test_benchmarks.py) --------------------------------------


async def _drain(source):
    return [f async for f in source.frames()]


def _batches(source):
    return [f.data for f in asyncio.run(_drain(source)) if isinstance(f, Batch)]


def _result(median, *, spread=0.02, digest="A", platform="linux", processor="cpu", pipeline="p"):
    """A minimal BenchResult, mirroring tests/test_bench.py's fixture so comparisons are apples-to-apples."""
    stats = Stats((median,), median, median * spread, spread, median, median)
    env = Environment("0.0.1", "3.12", platform, processor, "abc1234")
    scale = {"rows": 1, "batch": 1, "keys": 1, "parallelism": 1, "workers": 1, "tier": 1}
    return BenchResult(pipeline, scale, 1, stats, digest, True, env, "")


# --- (a) SyntheticKeyedSource skewed key draw ---------------------------------------------------


def test_skew_draws_a_fixed_golden_key_list():
    # skew>0 draws keys from a zipfian PMF via default_rng(_SEED).choice — a fixed seed, so the draw is
    # a hard-coded golden. A faithful port must reproduce this exact list from the same seed and PMF.
    b = _batches(SyntheticKeyedSource(num_batches=1, batch_rows=8, key_cardinality=5, skew=1.2))[0]
    assert b.column("key").to_pylist() == [0, 4, 0, 1, 0, 0, 2, 0]


# --- (b) null_fraction + value_spread: exact null-index set, value list, shared-rng across batches --


def test_null_and_value_draws_are_golden_and_rng_carries_across_batches():
    # Each batch draws the null mask BEFORE the value from ONE generator seeded once, so batch 2's draws
    # continue batch 1's stream — they are not independent re-seeds. Pin both batches: identical null-index
    # sets or value lists across batches would betray a per-batch reseed; distinct ones prove the carry.
    bs = _batches(
        SyntheticKeyedSource(
            num_batches=2, batch_rows=8, key_cardinality=4, null_fraction=0.5, value_spread=10
        )
    )
    assert len(bs) == 2

    keys0 = bs[0].column("key").to_pylist()
    null_idx0 = [i for i, k in enumerate(keys0) if k is None]
    assert null_idx0 == [0, 2, 4, 5, 7]
    assert bs[0].column("value").to_pylist() == [4, 7, 3, 2, 6, 4, 2, 5]
    assert bs[0].column("value").type == pa.int64()

    keys1 = bs[1].column("key").to_pylist()
    null_idx1 = [i for i, k in enumerate(keys1) if k is None]
    assert null_idx1 == [
        0,
        4,
        5,
        6,
    ]  # differs from batch 0 -> generator state carried across batches
    assert bs[1].column("value").to_pylist() == [9, 10, 1, 6, 8, 3, 5, 6]

    # ts is the global row index (batch 1 continues where batch 0 stopped), independent of the rng.
    assert bs[0].column("ts").to_pylist() == list(range(8))
    assert bs[1].column("ts").to_pylist() == list(range(8, 16))


# --- (c) SyntheticGridSource: gid formula and reseeded-per-timestep value ------------------------


def test_grid_gid_formula_and_reseeded_value_are_golden():
    # A tiny 2x3 grid, one day (24 hourly timesteps), one batch per timestep (6 cells < rows_per_batch).
    src = SyntheticGridSource(n_days=1, nlat=2, nlon=3, rows_per_batch=100)
    bs = _batches(src)
    assert len(bs) == 24  # one batch per hourly timestep

    nlat, nlon = 2, 3
    lat_idx = np.repeat(np.arange(nlat), nlon)
    lon_idx = np.tile(np.arange(nlon), nlat)
    # gid == (lat_idx*nlon + lon_idx)*24 + t%24, verified against every timestep.
    for t, b in enumerate(bs):
        expected_gid = ((lat_idx * nlon + lon_idx) * 24 + (t % 24)).tolist()
        assert b.column("gid").to_pylist() == expected_gid

    # Value == 273.15 + 30*cos(deg2rad(lat)) + default_rng(_SEED+t).normal(0,3,n), reseeded per t.
    # Pin the exact t=0 draw so a port reproduces the pole-to-equator gradient AND the per-t noise seed.
    assert bs[0].column("value").to_pylist() == [
        274.15056067048204,
        275.9016645781096,
        277.8690155682963,
        273.1011153482602,
        268.0667849911731,
        276.2860697736128,
    ]
    assert bs[0].column("value").type == pa.float64()


def test_grid_value_reseeds_per_timestep_not_across():
    # The per-timestep reseed means the noise is a function of t alone (order-independent): rebuild t=0's
    # value directly from _SEED and confirm it equals the source's t=0 batch.
    from nautilus.benchmarks import _SEED

    src = SyntheticGridSource(n_days=1, nlat=2, nlon=3, rows_per_batch=100)
    bs = _batches(src)
    nlat, nlon, n = 2, 3, 6
    lat = np.linspace(90.0, -90.0, nlat)
    latitude = np.repeat(lat, nlon)
    base = 273.15 + 30.0 * np.cos(np.deg2rad(latitude))
    expected = (base + np.random.default_rng(_SEED + 0).normal(0.0, 3.0, n)).tolist()
    assert bs[0].column("value").to_pylist() == expected


# --- (d) make_region_tagger: inclusive bounds, schema, empty-batch pass-through ------------------


def test_region_tagger_bounds_are_inclusive_on_all_four_edges():
    # An on-edge pixel on each of lat_min / lat_max / lon_min / lon_max must be tagged (>= and <=), while
    # a just-outside pixel is dropped. Region 0 spans [0,10]x[0,10].
    tag = make_region_tagger([("A", 0.0, 10.0, 0.0, 10.0)], "value")
    # lat_min edge, lat_max edge, lon_min edge, lon_max edge, inside, outside(dropped)
    batch = pa.record_batch(
        {
            "latitude": pa.array([0.0, 10.0, 5.0, 5.0, 5.0, -1.0]),
            "longitude": pa.array([5.0, 5.0, 0.0, 10.0, 5.0, 5.0]),
            "value": pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        }
    )
    out = tag(batch)
    assert out.column("region_id").to_pylist() == [
        0,
        0,
        0,
        0,
        0,
    ]  # five kept, the -1 latitude dropped
    assert out.column("value").to_pylist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_region_tagger_output_schema_is_exactly_region_id_int32_value_float64():
    tag = make_region_tagger([("Box", 10.0, 20.0, 100.0, 110.0)], "value")
    batch = pa.record_batch(
        {
            "latitude": pa.array([15.0]),
            "longitude": pa.array([105.0]),
            "value": pa.array([1.0]),
        }
    )
    out = tag(batch)
    assert out.schema.names == ["region_id", "value"]
    assert out.schema.field("region_id").type == pa.int32()
    assert out.schema.field("value").type == pa.float64()


def test_region_tagger_all_outside_yields_empty_batch_not_a_raise():
    # A batch with no pixel in any box returns an EMPTY batch carrying the same schema — not an error and
    # not a zero-column batch — so a downstream operator sees a consistent schema even on empty input.
    tag = make_region_tagger([("Box", 10.0, 20.0, 100.0, 110.0)], "value")
    batch = pa.record_batch(
        {
            "latitude": pa.array([0.0, 50.0]),
            "longitude": pa.array([0.0, 200.0]),
            "value": pa.array([9.0, 8.0]),
        }
    )
    out = tag(batch)
    assert out.num_rows == 0
    assert out.schema.names == ["region_id", "value"]
    assert out.schema.field("region_id").type == pa.int32()
    assert out.schema.field("value").type == pa.float64()


# --- (e) save_baseline byte layout --------------------------------------------------------------


def test_save_baseline_byte_layout(tmp_path):
    path = tmp_path / "baseline.json"
    # Insert out of alphabetical order to prove results are sorted by name on write.
    results = {
        "zeta": _result(200.0, digest="z", pipeline="zeta"),
        "alpha": _result(100.0, digest="a", pipeline="alpha"),
    }
    bench.save_baseline(path, results)
    raw = path.read_text()

    assert raw.endswith("\n")  # trailing newline (POSIX text file)
    assert "\n  " in raw  # indent=2 (a nested key is indented two spaces)

    obj = json.loads(raw)
    # sort_keys=True sorts every object's keys, so the TOP-LEVEL order is alphabetical
    # (note, results, version), not the declaration order (version, note, results).
    assert list(obj.keys()) == ["note", "results", "version"]
    assert obj["version"] == 1
    assert "note" in obj and isinstance(obj["note"], str)
    # results sorted by pipeline name.
    assert list(obj["results"].keys()) == ["alpha", "zeta"]
    # each result dict's keys are sorted too.
    assert list(obj["results"]["alpha"].keys()) == [
        "deterministic",
        "environment",
        "pipeline",
        "recorded_at",
        "scale",
        "structural_digest",
        "throughput_rows_per_sec",
        "trials",
    ]


# --- (f) compare(): boundary at +/-threshold, and base median 0 ---------------------------------


def test_compare_delta_exactly_at_threshold_is_unchanged():
    # threshold with zero noise is the 0.07 floor; a delta of EXACTLY +/-threshold is unchanged (the
    # classifier uses strict < and >), so only a change that clears the bar is called real.
    up = bench.compare(_result(1000.0, spread=0.0), _result(1070.0, spread=0.0))
    assert up.threshold == pytest.approx(0.07)
    assert up.delta == pytest.approx(0.07)
    assert up.status == "unchanged"

    down = bench.compare(_result(1000.0, spread=0.0), _result(930.0, spread=0.0))
    assert down.delta == pytest.approx(-0.07)
    assert down.status == "unchanged"


def test_compare_base_median_zero_is_delta_zero_not_div_by_zero():
    # A zero base median (a degenerate/empty run) yields delta 0.0 rather than raising ZeroDivisionError,
    # so the comparison degrades to "unchanged" instead of crashing the check.
    c = bench.compare(_result(0.0, spread=0.0), _result(500.0, spread=0.0))
    assert c.delta == 0.0
    assert c.status == "unchanged"


# --- (g) result_from_dict coercion; load_baseline without 'results' -----------------------------


def test_result_from_dict_coerces_string_numbers_and_null_commit():
    # A baseline loaded from JSON may carry numbers as strings (e.g. hand-edited); result_from_dict
    # coerces scale to int, throughput fields to float, and preserves a null commit as None.
    d = {
        "pipeline": "p",
        "scale": {
            "rows": "10",
            "batch": "4096",
            "keys": "1000",
            "parallelism": "2",
            "workers": "1",
            "tier": "1",
        },
        "trials": "5",
        "throughput_rows_per_sec": {
            "samples": ["1.5", "2.5"],
            "median": "2.0",
            "iqr": "1.0",
            "rel_spread": "0.5",
            "min": "1.5",
            "max": "2.5",
        },
        "structural_digest": "abcd",
        "deterministic": True,
        "environment": {
            "nautilus_version": "0.0.1",
            "python_version": "3.12",
            "platform": "linux",
            "processor": "cpu",
            "commit": None,
        },
        "recorded_at": "",
    }
    r = result_from_dict(d)
    assert r.scale["rows"] == 10 and isinstance(r.scale["rows"], int)
    assert r.scale["tier"] == 1 and isinstance(r.scale["tier"], int)
    assert r.trials == 5 and isinstance(r.trials, int)
    assert r.throughput_rows_per_sec.median == 2.0
    assert isinstance(r.throughput_rows_per_sec.median, float)
    assert r.throughput_rows_per_sec.samples == (1.5, 2.5)
    assert r.environment.commit is None


def test_load_baseline_without_results_key_is_empty_dict(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"version": 1, "note": "x"}))
    assert bench.load_baseline(path) == {}


# --- (h) one golden structural_digest hex -------------------------------------------------------


def test_bench_keyed_structural_digest_is_a_pinned_hex(monkeypatch):
    # The ultimate port anchor: a real bench-keyed run (keyed count over the clean SyntheticKeyedSource)
    # at a tiny fixed scale produces this exact digest. The digest is machine-independent (it hashes
    # output structure, not timing), so a Rust port that computes the same aggregation over the same
    # deterministic input must reproduce this hex. Kept small (800 rows) to stay fast and hermetic.
    monkeypatch.setenv("NAUTILUS_BENCH_ROWS", "800")
    monkeypatch.setenv("NAUTILUS_BENCH_BATCH", "100")
    monkeypatch.setenv("NAUTILUS_BENCH_KEYS", "10")
    # Stressor knobs bench-keyed ignores; set them to prove they do not perturb the digest.
    monkeypatch.setenv("NAUTILUS_BENCH_SKEW", "1.5")
    monkeypatch.setenv("NAUTILUS_BENCH_DELAY_US", "500")

    result = bench.run_once(
        "bench-keyed", parallelism=1, workers=1, capacity=16, tier=Tier.COUNTERS
    )
    assert (
        result.telemetry.structural_digest()
        == "906dfc54141104ed02a2412424f521b32286eb1820981d041179b00e37a8e822"
    )
