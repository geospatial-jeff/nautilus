"""The ``.agg_by`` grouped-aggregation verb and its :class:`~nautilus.operators.KeyedAgg` operator:
sum/count/mean/min/max over one or several keys, the integer fast path and the general path, matching
SQL null semantics and agreeing with the specialized :class:`~nautilus.operators.KeyedMean`."""

import asyncio

import numpy as np
import pyarrow as pa
import pytest

from nautilus import from_batches, source
from nautilus.driver.local import run_local_chain
from nautilus.driver.run import run_plan
from nautilus.operators import KeyedAgg, KeyedMean


def _by_key(batches, key: str) -> dict:
    return {row[key]: row for b in batches for row in b.to_pylist()}


def test_agg_by_integer_fast_path_all_funcs():
    # Single non-negative-integer key → the bincount/scatter fast path (no per-batch group-by).
    b = pa.record_batch(
        {"k": pa.array([0, 1, 0, 2, 1], pa.int64()), "v": pa.array([1.0, 2.0, 3.0, 4.0, 5.0])}
    )
    out = (
        source(from_batches(b))
        .agg_by(
            "k", s=("v", "sum"), c=("v", "count"), m=("v", "mean"), lo=("v", "min"), hi=("v", "max")
        )
        .run()
        .batches
    )
    r = _by_key(out, "k")
    assert r[0] == {"k": 0, "s": 4.0, "c": 2, "m": 2.0, "lo": 1.0, "hi": 3.0}
    assert r[1] == {"k": 1, "s": 7.0, "c": 2, "m": 3.5, "lo": 2.0, "hi": 5.0}
    assert r[2] == {"k": 2, "s": 4.0, "c": 1, "m": 4.0, "lo": 4.0, "hi": 4.0}


def test_agg_by_string_key_general_path():
    b = pa.record_batch({"k": pa.array(["a", "b", "a"]), "v": pa.array([1.0, 5.0, 3.0])})
    r = _by_key(
        source(from_batches(b)).agg_by("k", m=("v", "mean"), c=("v", "count")).run().batches, "k"
    )
    assert r["a"] == {"k": "a", "m": 2.0, "c": 2}
    assert r["b"] == {"k": "b", "m": 5.0, "c": 1}


def test_agg_by_composite_key():
    b = pa.record_batch(
        {
            "k": pa.array([0, 0, 1], pa.int64()),
            "g": pa.array(["x", "y", "x"]),
            "v": pa.array([2.0, 4.0, 6.0]),
        }
    )
    out = source(from_batches(b)).agg_by(["k", "g"], m=("v", "mean")).run().batches
    got = {(row["k"], row["g"]): row["m"] for bb in out for row in bb.to_pylist()}
    assert got == {(0, "x"): 2.0, (0, "y"): 4.0, (1, "x"): 6.0}


def test_agg_by_skips_null_values_like_sql():
    # A null value joins no aggregate — COUNT(col)/SUM/MEAN skip it (both key paths).
    b = pa.record_batch({"k": pa.array([0, 0, 1], pa.int64()), "v": pa.array([2.0, None, 5.0])})
    r = _by_key(
        source(from_batches(b))
        .agg_by("k", s=("v", "sum"), c=("v", "count"), m=("v", "mean"))
        .run()
        .batches,
        "k",
    )
    assert r[0] == {"k": 0, "s": 2.0, "c": 1, "m": 2.0}
    assert r[1] == {"k": 1, "s": 5.0, "c": 1, "m": 5.0}


def test_agg_by_accumulator_grows_across_batches():
    # Later batches introduce higher keys, so the fast-path numpy accumulators must grow and keep prior state.
    b1 = pa.record_batch({"k": pa.array([0, 1], pa.int64()), "v": pa.array([1.0, 2.0])})
    b2 = pa.record_batch({"k": pa.array([1, 5], pa.int64()), "v": pa.array([10.0, 20.0])})
    r = _by_key(
        source(from_batches(b1, b2)).agg_by("k", s=("v", "sum"), hi=("v", "max")).run().batches, "k"
    )
    assert r[0] == {"k": 0, "s": 1.0, "hi": 1.0}
    assert r[1] == {"k": 1, "s": 12.0, "hi": 10.0}
    assert r[5] == {"k": 5, "s": 20.0, "hi": 20.0}


def test_agg_by_mean_matches_keyed_mean():
    rng = np.random.default_rng(2)
    data = [
        pa.record_batch(
            {"k": pa.array(rng.integers(0, 30, 3000)), "v": pa.array(rng.normal(0, 1, 3000))}
        )
        for _ in range(3)
    ]
    km = {
        row["k"]: round(row["m"], 9)
        for b in source(from_batches(*data))
        .apply(KeyedMean("k", "v", "m"), key_columns="k")
        .run()
        .batches
        for row in b.to_pylist()
    }
    ab = {
        row["k"]: round(row["m"], 9)
        for b in source(from_batches(*data)).agg_by("k", m=("v", "mean")).run().batches
        for row in b.to_pylist()
    }
    assert km == ab and len(ab) == 30


def test_agg_by_parallel_matches_serial():
    # The keyed shuffle co-partitions by key, so parallelism > 1 must not change the grouped result.
    rng = np.random.default_rng(1)
    data = [
        pa.record_batch(
            {"k": pa.array(rng.integers(0, 50, 2000)), "v": pa.array(rng.normal(0, 1, 2000))}
        )
        for _ in range(4)
    ]

    def means(par: int) -> dict:
        graph = source(from_batches(*data)).agg_by("k", m=("v", "mean")).to_graph(parallelism=par)
        res = asyncio.run(run_plan(graph))
        return {row["k"]: round(row["m"], 9) for b in res.batches for row in b.to_pylist()}

    assert means(1) == means(3)


def test_agg_by_is_deterministic():
    rng = np.random.default_rng(3)
    data = [
        pa.record_batch(
            {"k": pa.array(rng.integers(0, 40, 2000)), "v": pa.array(rng.normal(0, 1, 2000))}
        )
        for _ in range(3)
    ]
    digests = {
        asyncio.run(
            run_local_chain(from_batches(*data), [KeyedAgg(("k",), {"m": ("v", "mean")})])
        ).telemetry.structural_digest()
        for _ in range(3)
    }
    assert len(digests) == 1


def test_agg_by_rejects_no_aggs_and_unknown_func():
    with pytest.raises(ValueError, match="at least one aggregation"):
        source(from_batches()).agg_by("k")
    with pytest.raises(ValueError, match="unknown func"):
        KeyedAgg(("k",), {"x": ("v", "median")})
