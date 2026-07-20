"""Tier 1 characterization tests pinning nautilus operator behavior for a Python->Rust rewrite.

These lock the *observable* semantics of the built-in operators in ``nautilus.operators`` — the null and
dtype rules, and the invariant that the vectorized fast paths (value-indexed accumulators, the one-to-one
join probe) agree exactly with the general per-distinct-key path they optimize. A rewrite that reproduces
these outputs is faithful by construction. Every golden here was derived by running the current code, not
assumed; where the brief's assumption differed from actual behavior, the actual behavior is pinned and
called out in a comment.

Operators are driven the two ways the existing suite drives them: two-input operators (HashJoin, Union)
directly through ``open`` / ``process_*`` / ``on_eos`` / ``close`` via :func:`_drive_join`, mirroring
``test_join.py`` / ``test_union.py``; keyed aggregations through the ``.agg_by`` / ``.apply`` DSL and
``.run()``, mirroring ``test_agg_by.py``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pyarrow as pa
import pytest

from nautilus import from_batches, source
from nautilus.core.operator import ListCollector, OperatorContext
from nautilus.operators import FilterRows, HashJoin, KeyedCount, Union
from nautilus.testing import batch


def _drive_join(join: HashJoin, steps: list[tuple[str, pa.RecordBatch]]) -> list[pa.RecordBatch]:
    """Open the join, feed ``(side, batch)`` steps, flush at EOS, close, return emitted batches — the
    same driver ``test_join.py`` uses."""
    coll = ListCollector()
    join.open(OperatorContext("j"))
    for side, b in steps:
        (join.process_left if side == "L" else join.process_right)(b, coll)
    join.on_eos(coll)
    join.close()
    return coll.drain()


def _triples(batches: list[pa.RecordBatch]) -> Counter[tuple[object, object, object]]:
    rows = [r for b in batches for r in b.to_pylist()]
    return Counter((r["id"], r["lval"], r["rval"]) for r in rows)


def _by_key(batches: list[pa.RecordBatch], key: str) -> dict:
    return {row[key]: row for b in batches for row in b.to_pylist()}


# --- (a) HashJoin one-to-one probe fast path emitting MULTIPLE rows == the general cross-product -------
# The fast branch fires when total == nq and qcount.max() == 1: every query row matches exactly one
# buffered row. Buffering R first, then probing with a 3-row L whose keys each hit one unique R row, drives
# that branch with a 3-row output (today's other fast-path tests only exercise a single output row).


def test_join_one_to_one_fast_path_multiple_rows_equals_cross_product() -> None:
    left = batch(id=[1, 2, 3], lval=["a", "b", "c"])
    right = batch(id=[1, 2, 3], rval=[10, 20, 30])  # each key unique -> qcount.max() == 1
    # R buffered first so the L probe is the one-to-one query; verified to hit the fast branch (3 rows).
    out = _drive_join(HashJoin("id"), [("R", right), ("L", left)])
    reference = Counter(  # plain-Python cross-product of equal keys
        (lk, lv, rv)
        for lk, lv in [(1, "a"), (2, "b"), (3, "c")]
        for rk, rv in [(1, 10), (2, 20), (3, 30)]
        if lk == rk
    )
    assert _triples(out) == reference
    assert reference == Counter({(1, "a", 10): 1, (2, "b", 20): 1, (3, "c", 30): 1})


def test_join_one_to_one_fast_path_equals_general_repeat_path() -> None:
    # Same query rows, but one buffered key now carries two rows, so the probe takes the general
    # repeat/cumsum branch instead of the one-to-one fast branch. The fast-path output above and this
    # general-path output must be the identical multiset once we account for the extra buffered row.
    left = batch(id=[1, 2], lval=["a", "b"])
    right = batch(id=[1, 2, 2], rval=[10, 20, 21])  # key 2 has two rows -> general branch
    out = _drive_join(HashJoin("id"), [("R", right), ("L", left)])
    assert _triples(out) == Counter({(1, "a", 10): 1, (2, "b", 20): 1, (2, "b", 21): 1})


# --- (b) inner join with a null key inside an int-fast batch: null matches null -------------------------


def test_join_inner_null_key_in_int_fast_batch_matches_null_and_value() -> None:
    # L id=[1, None, 2], R id=[None, 2] on how='inner': None==None and 2==2 match; id 1 has no right row.
    left = batch(id=[1, None, 2], lval=["a", "b", "c"])
    right = batch(id=[None, 2], rval=[10, 20])
    expected = Counter({(None, "b", 10): 1, (2, "c", 20): 1})
    assert _triples(_drive_join(HashJoin("id", "id", "inner"), [("L", left), ("R", right)])) == (
        expected
    )
    # Order-independent: buffering R first gives the same matches.
    assert _triples(_drive_join(HashJoin("id", "id", "inner"), [("R", right), ("L", left)])) == (
        expected
    )


# --- (c) KeyedCount preserves the key width and emits int64 counts --------------------------------------


def test_keyed_count_preserves_int8_key_and_emits_int64_count() -> None:
    # KeyedCount('k', count_col='n') over an int8 key: the count column 'n' is int64, the key keeps int8.
    b = pa.record_batch({"k": pa.array([5, 1, 5], pa.int8())})
    rb = source(from_batches(b)).apply(KeyedCount("k", "n"), key_columns="k").run().batches[0]
    assert (
        rb.schema.field("k").type == pa.int8()
    )  # key width preserved (unlike KeyedAgg's int64 widen)
    assert rb.schema.field("n").type == pa.int64()
    assert {r["k"]: r["n"] for r in rb.to_pylist()} == {1: 1, 5: 2}


# --- (d) KeyedCount stays on the fast path for a sparse non-negative key (no demotion) ------------------
# The deliberate opposite of KeyedAgg: KeyedCount does not guard its sparse case, so a high-valued key
# keeps the value-indexed bincount and simply sizes the array to max(key) + 1.


def test_keyed_count_sparse_nonneg_key_stays_on_fast_path() -> None:
    op = KeyedCount("k")
    op.open(OperatorContext("kc"))
    coll = ListCollector()
    op.process(pa.record_batch({"k": pa.array([5, 1000, 5], pa.int64())}), coll)
    # No demotion: the integer bincount accumulator is still live (sized to 1001), never drained to state.
    assert op._counts is not None
    assert op._counts.size == 1001
    op.on_eos(coll)
    out = coll.drain()
    assert {r["k"]: r["count"] for b in out for r in b.to_pylist()} == {5: 2, 1000: 1}


# --- (e) KeyedAgg min/max over negative int64 values: exact -5/3 as int64 on both paths -----------------


def test_agg_by_min_max_negative_int_values_exact_on_both_paths() -> None:
    # The fast-path min/max accumulators seed at +/-inf (float64); an integer input column still yields an
    # exact int64 result. A single non-negative int key stays on that inf-seed fast path; a string key
    # takes the general path. Both must return -5 / 3 as int64. (Values are negative here; keys are not.)
    vals = pa.array([-5, -1, 3], pa.int64())
    aggs = dict(lo=("v", "min"), hi=("v", "max"))
    fast = (
        source(from_batches(pa.record_batch({"k": pa.array([0, 0, 0], pa.int64()), "v": vals})))
        .agg_by("k", **aggs)
        .run()
        .batches[0]
    )
    gen = (
        source(from_batches(pa.record_batch({"k": pa.array(["a", "a", "a"]), "v": vals})))
        .agg_by("k", **aggs)
        .run()
        .batches[0]
    )
    assert _by_key([fast], "k")[0] == {"k": 0, "lo": -5, "hi": 3}
    assert _by_key([gen], "k")["a"] == {"k": "a", "lo": -5, "hi": 3}
    for rb in (fast, gen):
        assert rb.schema.field("lo").type == pa.int64()
        assert rb.schema.field("hi").type == pa.int64()


# --- (f) KeyedAgg mean: fast path and general path agree bit-for-bit (exact ==, not <1e-9) --------------
# The two paths accumulate the same values in the same batch/row order, so the float64 means are equal
# bit-for-bit, not merely close. Multi-batch, normal-distributed input makes the fold order-sensitive.


def test_agg_by_mean_fast_and_general_paths_agree_bit_for_bit() -> None:
    rng = np.random.default_rng(1234)
    fast_batches, gen_batches = [], []
    for _ in range(4):
        k = rng.integers(0, 12, 700)
        v = rng.normal(0, 1, 700)
        fast_batches.append(pa.record_batch({"k": pa.array(k), "v": pa.array(v)}))
        gen_batches.append(pa.record_batch({"k": pa.array([f"g{x}" for x in k]), "v": pa.array(v)}))
    fast = source(from_batches(*fast_batches)).agg_by("k", m=("v", "mean")).run().batches
    gen = source(from_batches(*gen_batches)).agg_by("k", m=("v", "mean")).run().batches
    fm = {r["k"]: r["m"] for b in fast for r in b.to_pylist()}
    gm = {int(r["k"][1:]): r["m"] for b in gen for r in b.to_pylist()}
    assert fm.keys() == gm.keys() and len(fm) == 12
    for k in fm:  # exact equality, not abs(fa - ga) < 1e-9
        assert fm[k] == gm[k]


# --- (g) KeyedAgg composite key: a null group component is distinct from a present value ----------------


def test_agg_by_composite_key_null_component_is_its_own_group() -> None:
    # Composite key ['k', 'g'] with g = ['x', None, 'x']: the (0, None) group is distinct from (0, 'x').
    b = pa.record_batch(
        {
            "k": pa.array([0, 0, 0], pa.int64()),
            "g": pa.array(["x", None, "x"]),
            "v": pa.array([2.0, 4.0, 6.0]),
        }
    )
    out = source(from_batches(b)).agg_by(["k", "g"], s=("v", "sum")).run().batches
    got = {(r["k"], r["g"]): r["s"] for bb in out for r in bb.to_pylist()}
    assert got == {(0, "x"): 8.0, (0, None): 4.0}


# --- (h) FilterRows: an all-false mask emits nothing; a null mask cell drops that row ------------------


def _filter_rows(mask: pa.Array, b: pa.RecordBatch) -> list[dict]:
    op = FilterRows(lambda _b: mask)
    op.open(OperatorContext("f"))
    coll = ListCollector()
    op.process(b, coll)
    return [r for out in coll.drain() for r in out.to_pylist()]


def test_filter_rows_all_false_emits_no_rows() -> None:
    # An all-false mask filters every row; ListCollector drops the resulting empty batch, so nothing is
    # emitted (0 rows and 0 batches).
    op = FilterRows(lambda _b: pa.array([False, False, False]))
    op.open(OperatorContext("f"))
    coll = ListCollector()
    op.process(batch(id=[1, 2, 3]), coll)
    out = coll.drain()
    assert out == []
    assert sum(b.num_rows for b in out) == 0


def test_filter_rows_null_mask_cell_drops_that_row() -> None:
    # A mask of [True, None, False]: only the True row survives — a null mask cell drops its row (Arrow's
    # filter treats null as "do not keep"), same as False.
    rows = _filter_rows(pa.array([True, None, False]), batch(id=[10, 20, 30]))
    assert rows == [{"id": 10}]


# --- (i) Union: rejects an int64/int32 name clash, accepts a metadata-only schema difference ------------


def test_union_rejects_same_name_int64_vs_int32() -> None:
    # Same column name, differing type (int64 then int32): the schemas are not equal, so Union raises.
    op = Union()
    op.open(OperatorContext("u"))
    coll = ListCollector()
    op.process_left(batch(id=pa.array([1], pa.int64())), coll)
    with pytest.raises(ValueError, match="share a schema"):
        op.process_right(batch(id=pa.array([2], pa.int32())), coll)


def test_union_accepts_metadata_only_schema_difference() -> None:
    # Schemas are compared by name and type, ignoring metadata, so a field-metadata-only difference is
    # accepted and both rows are forwarded.
    op = Union()
    op.open(OperatorContext("u"))
    coll = ListCollector()
    plain = pa.record_batch(
        [pa.array([2], pa.int64())], schema=pa.schema([pa.field("id", pa.int64())])
    )
    with_meta = pa.record_batch(
        [pa.array([1], pa.int64())],
        schema=pa.schema([pa.field("id", pa.int64(), metadata={b"note": b"x"})]),
    )
    op.process_left(plain, coll)
    op.process_right(with_meta, coll)  # metadata-only diff must not raise
    ids = [r["id"] for b in coll.drain() for r in b.to_pylist()]
    assert Counter(ids) == Counter([2, 1])
