"""Tier 1 port-conformance tests for the partitioners' observable routing behavior.

A future Python->Rust rewrite that passes this file routes the same rows to the same instances as the
current code does. Where ``tests/test_partition.py`` fuzzes the invariants (row conservation,
co-location, per-row equivalence), this file pins the exact, hand-derived buckets and the error-message
contract for a small set of adversarial keys — combo-fold multi-column routing, partially- and
fully-null keys, the deliberate ``True != 1`` split, and the ``_validate_key`` scalar-type message. The
group-table indirection is pinned by comparing a spec-built partitioner to a directly-constructed one.

Every golden here was produced by running the current code (see the module note per test); none is
invented. Buckets are for the CURRENT ``stable_bucket`` (msgpack + blake2b), so they are the port's
target, not an accident of this run.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from nautilus.compile.plan import ForwardSpec, KeyGroupSpec
from nautilus.runtime.execute import partitioner_from_spec
from nautilus.runtime.partition import (
    Forward,
    HashPartitioner,
    KeyGroupPartitioner,
    _validate_key,
    stable_bucket,
)
from nautilus.testing import batch


def _instance_of_key(
    partitioner: object, b: pa.RecordBatch, q: int, key_columns: tuple[str, ...]
) -> dict[tuple[object, ...], set[int]]:
    """Map each distinct routed key tuple to the set of instances it landed on. A correctly co-locating
    partitioner leaves each key on exactly one instance, so the value set has length 1."""
    where: dict[tuple[object, ...], set[int]] = {}
    for idx, sub in partitioner.route(b, q):  # type: ignore[attr-defined]
        cols = [sub.column(c).to_pylist() for c in key_columns]
        for r in range(sub.num_rows):
            where.setdefault(tuple(col[r] for col in cols), set()).add(idx)
    return where


# --- (a) multi-column combo-fold routing pins to per-tuple stable_bucket ------------------------
# Golden: for a=['x','x','y'], c=[1,2,1] at parallelism 4, stable_bucket of each raw tuple is
# ('x',1)->3, ('x',2)->3, ('y',1)->2 (run against the current code). The vectorized combo-fold in
# _bucket_per_row must reproduce exactly this, conserving every row.


def test_multi_column_combo_fold_routes_by_stable_bucket_of_each_tuple() -> None:
    b = batch(a=["x", "x", "y"], c=[1, 2, 1])
    routed = HashPartitioner(["a", "c"]).route(b, 4)

    # Rows are conserved exactly once across the emitted sub-batches.
    assert sum(sub.num_rows for _, sub in routed) == 3
    assert all(sub.num_rows > 0 for _, sub in routed)  # no zero-row frame emitted

    # Each row's instance is exactly stable_bucket of its raw tuple over the parallelism.
    where = _instance_of_key(HashPartitioner(["a", "c"]), b, 4, ("a", "c"))
    assert where == {
        ("x", 1): {3},
        ("x", 2): {3},
        ("y", 1): {2},
    }
    # Byte-for-byte: the hand-derived buckets above equal stable_bucket, so the fold matches the oracle.
    for key, instances in where.items():
        assert instances == {stable_bucket(key, 4)}


# --- (b) partially-null multi-column keys are distinct keys, never merged ------------------------
# Golden: at parallelism 4 both ('x',None) and (None,'x') hash to bucket 0, so they co-reside on
# instance 0 — but they are DISTINCT keys (distinct msgpack bytes), which the hash proves by splitting
# them at parallelism 3: ('x',None)->2, (None,'x')->0. "Never merged" means a partitioner must not
# collapse them into one key; each stays its own routable key.


def test_partial_null_keys_route_by_stable_bucket_and_are_distinct() -> None:
    b = batch(
        a=pa.array(["x", None], pa.string()),
        c=pa.array([None, "x"], pa.string()),
    )
    # At parallelism 4 both land on instance 0 (their stable_bucket), each on exactly one instance.
    where4 = _instance_of_key(HashPartitioner(["a", "c"]), b, 4, ("a", "c"))
    assert where4 == {("x", None): {0}, (None, "x"): {0}}
    assert stable_bucket(("x", None), 4) == 0
    assert stable_bucket((None, "x"), 4) == 0


def test_partial_null_keys_are_never_merged_they_split_at_parallelism_3() -> None:
    # The two keys are distinct: at parallelism 3 they route to DIFFERENT instances, which is only
    # possible if the shuffle never merged ('x',None) with (None,'x') into a single key.
    b = batch(
        a=pa.array(["x", None], pa.string()),
        c=pa.array([None, "x"], pa.string()),
    )
    where3 = _instance_of_key(HashPartitioner(["a", "c"]), b, 3, ("a", "c"))
    assert where3 == {("x", None): {2}, (None, "x"): {0}}
    assert stable_bucket(("x", None), 3) == 2
    assert stable_bucket((None, "x"), 3) == 0


# --- (c) all-null single-column key routes as (None,) and conserves rows -------------------------
# Golden: [None,None,None] at parallelism 3 all route to stable_bucket((None,),3) == 1.


def test_all_null_single_column_key_co_locates_and_conserves() -> None:
    b = batch(key=pa.array([None, None, None], pa.string()))
    routed = HashPartitioner(["key"]).route(b, 3)

    # All three null rows land on the single instance stable_bucket((None,),3); rows conserved.
    assert stable_bucket((None,), 3) == 1
    assert len(routed) == 1  # one instance owns every null row
    idx, sub = routed[0]
    assert idx == 1
    assert sub.num_rows == 3
    assert sub.column("key").to_pylist() == [None, None, None]


# --- (d) True and 1 route to DIFFERENT buckets despite True == 1 in Python -----------------------
# Golden: stable_bucket((True,),4) == 3 but stable_bucket((1,),4) == 0. This is a deliberate contract:
# msgpack tags bool and int distinctly, so a bool key and an int key are different keys even though
# True == 1 under Python equality.


def test_bool_true_and_int_one_route_to_different_buckets() -> None:
    assert (
        True == 1
    )  # noqa: E712 — the Python equality that the shuffle deliberately does NOT honor
    assert stable_bucket((True,), 4) == 3
    assert stable_bucket((1,), 4) == 0
    assert stable_bucket((True,), 4) != stable_bucket((1,), 4)

    # And through the real route: a bool key column and an int key column land on those instances.
    bool_batch = batch(key=pa.array([True], pa.bool_()))
    int_batch = batch(key=pa.array([1], pa.int64()))
    bool_idx, _ = HashPartitioner(["key"]).route(bool_batch, 4)[0]
    int_idx, _ = HashPartitioner(["key"]).route(int_batch, 4)[0]
    assert bool_idx == 3
    assert int_idx == 0


# --- (e) _validate_key error message contract ---------------------------------------------------
# Golden message (run against current code): "cannot route on key scalar 1.5 of type float; allowed
# key scalars are str/int/bool/bytes/null". The message names the allowed scalars and echoes the bad
# value's repr and type name.


def test_validate_key_rejects_float_with_allowed_scalars_message() -> None:
    with pytest.raises(
        TypeError,
        match=r"allowed key scalars are str/int/bool/bytes/null",
    ):
        _validate_key((1.5,))


def test_validate_key_message_echoes_bad_value_repr_and_type() -> None:
    # The message includes the offending value's repr (1.5) and its type name (float), so an operator
    # keying on a float sees exactly which cell is bad.
    with pytest.raises(TypeError, match=r"cannot route on key scalar 1\.5 of type float"):
        _validate_key((1.5,))


def test_validate_key_accepts_the_allowed_scalars() -> None:
    # The complement of the rejection: str/int/bool/bytes/None all validate (no raise), so the message
    # test above pins a real boundary, not a blanket reject.
    _validate_key(("s", 1, True, b"x", None))


# --- (f) partitioner_from_spec builds an equivalent partitioner ---------------------------------
# A KeyGroupSpec(('k',),(2,0,1)) must build a KeyGroupPartitioner routing byte-for-byte like a directly
# constructed KeyGroupPartitioner(('k',),(2,0,1)); a ForwardSpec must thread sender_index into Forward.


def _rid_to_instance(partitioner: object, b: pa.RecordBatch, q: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for idx, sub in partitioner.route(b, q):  # type: ignore[attr-defined]
        for rid in sub.column("rid").to_pylist():
            out[rid] = idx
    return out


def test_partitioner_from_keygroup_spec_matches_direct_construction() -> None:
    from_spec = partitioner_from_spec(KeyGroupSpec(("k",), (2, 0, 1)))
    direct = KeyGroupPartitioner(("k",), (2, 0, 1))
    assert isinstance(from_spec, KeyGroupPartitioner)

    # route() is identical row-for-row across a batch spanning several key groups.
    b = batch(rid=list(range(20)), k=[f"v{i % 6}" for i in range(20)])
    assert _rid_to_instance(from_spec, b, 3) == _rid_to_instance(direct, b, 3)


def test_partitioner_from_forward_spec_threads_sender_index() -> None:
    # ForwardSpec carries no index; partitioner_from_spec threads the emitting instance's sender_index
    # into the Forward, so sender 3 on an equal-width edge co-locates onto downstream instance 3.
    fwd = partitioner_from_spec(ForwardSpec(), 3)
    assert isinstance(fwd, Forward)
    b = batch(x=[1, 2, 3])
    assert fwd.route(b, 4) == [(3, b)]
    # A single downstream owner collapses any sender to instance 0.
    assert fwd.route(b, 1) == [(0, b)]
