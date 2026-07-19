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


def test_agg_by_rejects_bad_specs_and_reserved_names():
    # All validation is eager, at the .agg_by() call site — not deep in a run.
    with pytest.raises(ValueError, match="at least one aggregation"):
        source(from_batches()).agg_by("k")
    with pytest.raises(ValueError, match="unknown func"):
        source(from_batches()).agg_by("k", x=("v", "median"))
    with pytest.raises(ValueError, match="must be"):
        source(from_batches()).agg_by("k", x=("v", "sum", "extra"))
    with pytest.raises(
        ValueError, match="collide"
    ):  # output name == a key column silently drops the key
        source(from_batches()).agg_by("k", k=("v", "sum"))
    with pytest.raises(TypeError, match="parallelism"):  # reserved keyword, not an aggregation
        source(from_batches()).agg_by("k", parallelism=("v", "sum"))


def test_agg_by_all_null_group_is_null_on_both_paths():
    # A group with rows but all-null values → NULL for sum/mean/min/max and 0 for count (SQL semantics),
    # identically on the integer fast path and the string general path.
    vals = pa.array([5.0, None, None])
    aggs = dict(s=("v", "sum"), c=("v", "count"), m=("v", "mean"), lo=("v", "min"), hi=("v", "max"))
    fast = _by_key(
        source(from_batches(pa.record_batch({"k": pa.array([0, 1, 2], pa.int64()), "v": vals})))
        .agg_by("k", **aggs)
        .run()
        .batches,
        "k",
    )
    gen = _by_key(
        source(from_batches(pa.record_batch({"k": pa.array(["a", "b", "c"]), "v": vals})))
        .agg_by("k", **aggs)
        .run()
        .batches,
        "k",
    )
    assert fast[1] == {"k": 1, "s": None, "c": 0, "m": None, "lo": None, "hi": None}
    assert gen["b"] == {"k": "b", "s": None, "c": 0, "m": None, "lo": None, "hi": None}


def test_agg_by_general_path_all_null_across_batches():
    # Regression: an all-null group split across batches used to crash the run (None + None / None / 0).
    b1 = pa.record_batch({"k": pa.array(["b", "b"]), "v": pa.array([None, None], pa.float64())})
    b2 = pa.record_batch({"k": pa.array(["b", "a"]), "v": pa.array([None, 3.0])})
    r = _by_key(
        source(from_batches(b1, b2)).agg_by("k", s=("v", "sum"), m=("v", "mean")).run().batches, "k"
    )
    assert r["b"] == {"k": "b", "s": None, "m": None}
    assert r["a"] == {"k": "a", "s": 3.0, "m": 3.0}


def test_agg_by_fast_and_general_paths_agree_under_nulls():
    # The two paths are selected invisibly (by key dtype), so they must agree exactly — fuzz with heavy
    # nulls (many all-null groups) over all five funcs, int key (fast) vs string key (general).
    rng = np.random.default_rng(7)
    aggs = dict(s=("v", "sum"), c=("v", "count"), m=("v", "mean"), lo=("v", "min"), hi=("v", "max"))
    for _ in range(20):
        fb, gb = [], []
        for _ in range(int(rng.integers(1, 4))):
            n = int(rng.integers(1, 40))
            k = rng.integers(0, 8, n)
            v = pa.array(rng.normal(0, 5, n), mask=rng.random(n) < 0.4)
            fb.append(pa.record_batch({"k": pa.array(k), "v": v}))
            gb.append(pa.record_batch({"k": pa.array([f"g{x}" for x in k]), "v": v}))
        f = _by_key(source(from_batches(*fb)).agg_by("k", **aggs).run().batches, "k")
        g = _by_key(source(from_batches(*gb)).agg_by("k", **aggs).run().batches, "k")
        for ik, fr in f.items():
            gr = g[f"g{ik}"]
            for a in aggs:
                fa, ga = fr[a], gr[a]
                assert (fa is None and ga is None) or abs(fa - ga) < 1e-9, (ik, a, fa, ga)


def test_agg_by_integer_column_keeps_integer_dtype_both_paths():
    aggs = dict(s=("v", "sum"), lo=("v", "min"), hi=("v", "max"))
    fast = (
        source(
            from_batches(
                pa.record_batch(
                    {"k": pa.array([0, 0, 1], pa.int64()), "v": pa.array([2, 3, 4], pa.int64())}
                )
            )
        )
        .agg_by("k", **aggs)
        .run()
        .batches[0]
    )
    gen = (
        source(
            from_batches(
                pa.record_batch(
                    {"k": pa.array(["x", "x", "y"]), "v": pa.array([2, 3, 4], pa.int64())}
                )
            )
        )
        .agg_by("k", **aggs)
        .run()
        .batches[0]
    )
    for rb in (fast, gen):
        assert rb.schema.field("s").type == pa.int64()
        assert rb.schema.field("lo").type == pa.int64()
        assert rb.schema.field("hi").type == pa.int64()


def test_agg_by_narrow_integer_key_emits_int64():
    # An int8 key is emitted as int64 so a later batch's larger key value cannot overflow the output type.
    out = (
        source(
            from_batches(
                pa.record_batch(
                    {"k": pa.array([1, 2, 1], pa.int8()), "v": pa.array([1.0, 2.0, 3.0])}
                )
            )
        )
        .agg_by("k", s=("v", "sum"))
        .run()
        .batches[0]
    )
    assert out.schema.field("k").type == pa.int64()


def test_agg_by_negative_or_null_key_in_later_batch_demotes_to_general_path():
    # The fast path latches on the first (non-negative int) batch; a later negative/null key must not crash
    # — it drains the accumulators into the group-by path, which handles any key.
    b1 = pa.record_batch({"k": pa.array([0, 1], pa.int64()), "v": pa.array([1.0, 2.0])})
    neg = pa.record_batch({"k": pa.array([-1, 0], pa.int64()), "v": pa.array([9.0, 3.0])})
    r = {
        row["k"]: row["s"]
        for b in source(from_batches(b1, neg)).agg_by("k", s=("v", "sum")).run().batches
        for row in b.to_pylist()
    }
    assert r == {0: 4.0, 1: 2.0, -1: 9.0}
    nul = pa.record_batch({"k": pa.array([None, 0], pa.int64()), "v": pa.array([9.0, 3.0])})
    r = {
        row["k"]: row["s"]
        for b in source(from_batches(b1, nul)).agg_by("k", s=("v", "sum")).run().batches
        for row in b.to_pylist()
    }
    assert r == {0: 4.0, 1: 2.0, None: 9.0}  # null key is its own group on the general path


def test_agg_by_sparse_key_demotes_without_oom():
    # A sparse 1e9 key the value-indexed fast path must refuse — a value-indexed array would be gigabytes.
    # It demotes to the general path and aggregates exactly.
    sparse = pa.record_batch(
        {"k": pa.array([10**9, 10**9, 3], pa.int64()), "v": pa.array([1.0, 2.0, 5.0])}
    )
    r = {
        row["k"]: row["s"]
        for b in source(from_batches(sparse)).agg_by("k", s=("v", "sum")).run().batches
        for row in b.to_pylist()
    }
    assert r == {10**9: 3.0, 3: 5.0}


def test_agg_by_empty_stream_emits_nothing():
    out = source(from_batches()).agg_by("k", m=("v", "mean")).run().batches
    assert sum(b.num_rows for b in out) == 0


def test_agg_by_distributed_run_matches_single_process():
    # workers=2 spawns processes and shuffles across a socket — exercises cloudpickle of KeyedAgg and the
    # cross-process keyed shuffle, not just in-process parallelism.
    rng = np.random.default_rng(5)
    data = [
        pa.record_batch(
            {"k": pa.array(rng.integers(0, 20, 1000)), "v": pa.array(rng.normal(0, 1, 1000))}
        )
        for _ in range(3)
    ]
    s = source(from_batches(*data)).agg_by("k", m=("v", "mean"), n=("v", "count"))
    dist = {
        r["k"]: (round(r["m"], 9), r["n"])
        for b in s.run(workers=2, parallelism=2).batches
        for r in b.to_pylist()
    }
    serial = {r["k"]: (round(r["m"], 9), r["n"]) for b in s.run().batches for r in b.to_pylist()}
    assert dist == serial and len(serial) == 20


def _keyed_mean(batch):
    return {
        row["k"]: row
        for b in source(from_batches(batch))
        .apply(KeyedMean("k", "v", "m"), key_columns="k")
        .run()
        .batches
        for row in b.to_pylist()
    }


def test_keyed_mean_normal_output_and_null_key_group():
    # The tuned single-key mean still emits (key, mean, n), and a null key forms its own group.
    b = pa.record_batch({"k": pa.array([0, 1, 0], pa.int64()), "v": pa.array([1.0, 10.0, 3.0])})
    assert _keyed_mean(b)[0] == {"k": 0, "m": 2.0, "n": 2}
    nk = pa.record_batch({"k": pa.array([0, None, 0], pa.int64()), "v": pa.array([1.0, 9.0, 3.0])})
    r = _keyed_mean(nk)
    assert r[0]["m"] == 2.0 and r[None]["m"] == 9.0


def test_keyed_mean_all_null_group_and_negative_key_fixed():
    # Regression: an all-null value group used to be silently dropped; now kept with NULL mean, 0 count.
    b = pa.record_batch({"k": pa.array([0, 1, 2], pa.int64()), "v": pa.array([5.0, None, None])})
    r = _keyed_mean(b)
    assert r[1] == {"k": 1, "m": None, "n": 0} and r[2] == {"k": 2, "m": None, "n": 0}
    # Regression: a negative integer key used to crash np.bincount; now the group-by path handles it.
    neg = pa.record_batch({"k": pa.array([-1, 0, -1], pa.int64()), "v": pa.array([1.0, 2.0, 3.0])})
    assert {k: v["m"] for k, v in _keyed_mean(neg).items()} == {-1: 2.0, 0: 2.0}


def test_keyed_mean_agrees_with_agg_by_under_nulls():
    # The specialized mean and the general verb must not diverge — including on all-null groups.
    rng = np.random.default_rng(11)
    data = [
        pa.record_batch(
            {
                "k": pa.array(rng.integers(0, 15, 2000)),
                "v": pa.array(rng.normal(0, 1, 2000), mask=rng.random(2000) < 0.25),
            }
        )
        for _ in range(3)
    ]

    def norm(batches):
        return {
            r["k"]: (None if r["m"] is None else round(r["m"], 9), r["n"])
            for b in batches
            for r in b.to_pylist()
        }

    km = norm(
        source(from_batches(*data)).apply(KeyedMean("k", "v", "m"), key_columns="k").run().batches
    )
    ab = norm(
        source(from_batches(*data)).agg_by("k", m=("v", "mean"), n=("v", "count")).run().batches
    )
    assert km == ab
