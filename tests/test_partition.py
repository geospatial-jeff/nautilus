"""Stage 1.5a: the keyed-shuffle hash and the partitioners, isolated from any runtime wiring.

These tests pin the three traps the shuffle has to survive: a process-stable hash (never builtin
``hash()``), a type-distinguishing encoder, and row-conserving routing that keeps every key co-located
on one instance.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest

from nautilus.runtime.partition import (
    Forward,
    HashPartitioner,
    KeyGroupPartitioner,
    RoundRobin,
    stable_bucket,
)
from nautilus.tensors import is_tensor, to_numpy
from nautilus.testing import batch

# A prime fan-out so the few type-distinctness keys are extremely unlikely to collide modulo the parallelism. The hash
# is deterministic, so any collision would be a permanent (non-flaky) failure caught once here.
_BIG_Q = 65521


# --- the mandated process-stable hash ----------------------------------------------------------

_SEED_PROBE = """
import os, sys
sys.path.insert(0, os.environ["NAUT_SRC"])
from nautilus.runtime.partition import stable_bucket
keys = [("the",), ("cat",), (1,), (2,), (True,), (b"x",), ("a", "bc"), (42, "z")]
stable = [stable_bucket(k, 7) for k in keys]
builtin = [hash(k[0]) % 7 for k in keys if isinstance(k[0], str)]  # salted per-process
print(",".join(map(str, stable)))
print(",".join(map(str, builtin)))
"""


def _probe(seed: str) -> tuple[str, str]:
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    env = {**os.environ, "PYTHONHASHSEED": seed, "NAUT_SRC": os.path.abspath(src)}
    out = subprocess.run(
        [sys.executable, "-c", _SEED_PROBE], env=env, capture_output=True, text=True, check=True
    )
    stable, builtin = out.stdout.splitlines()
    return stable, builtin


def test_stable_bucket_is_seed_and_process_stable() -> None:
    # Two child processes with different PYTHONHASHSEED. stable_bucket must agree; builtin hash() must
    # not (which both proves the seeds really differ and that we never fell back to hash()).
    stable_a, builtin_a = _probe("0")
    stable_b, builtin_b = _probe("123456789")
    assert (
        stable_a == stable_b
    ), "stable_bucket changed across PYTHONHASHSEED — builtin hash() leaked in"
    assert builtin_a != builtin_b, "PYTHONHASHSEED did not vary builtin hash() — test is vacuous"


# --- the type-distinguishing encoder -----------------------------------------------------------


def test_encoder_distinguishes_scalar_types() -> None:
    # int 1 / str "1" / bool True / bytes b"1" are four different keys.
    buckets = {
        "int": stable_bucket((1,), _BIG_Q),
        "str": stable_bucket(("1",), _BIG_Q),
        "bool": stable_bucket((True,), _BIG_Q),
        "bytes": stable_bucket((b"1",), _BIG_Q),
    }
    assert len(set(buckets.values())) == 4, buckets


def test_encoder_is_length_prefixed() -> None:
    # length-prefixing keeps ("a","bc") and ("ab","c") apart even though they concatenate the same.
    assert stable_bucket(("a", "bc"), _BIG_Q) != stable_bucket(("ab", "c"), _BIG_Q)


# --- Forward / RoundRobin ----------------------------------------------------------------------


def test_forward_single_owner_routes_to_instance_zero() -> None:
    # One downstream instance (a fan-in, or the trivial 1:1 edge): every sender's batch goes to it,
    # whatever the sender's own index.
    b = batch(x=[1, 2, 3])
    assert Forward().route(b, 1) == [(0, b)]
    assert Forward(3).route(b, 1) == [(0, b)]


def test_forward_equal_width_co_locates_sender_to_same_index() -> None:
    # Equal-width keyless edge: sender i hands its whole batch straight to downstream instance i — the
    # data-locality forward the compiler picks when the two stages are the same width.
    b = batch(x=[1, 2, 3])
    assert Forward(0).route(b, 4) == [(0, b)]
    assert Forward(2).route(b, 4) == [(2, b)]


def test_roundrobin_rotates_whole_batches() -> None:
    rr = RoundRobin()
    b = batch(x=[1])
    assert [rr.route(b, 3)[0][0] for _ in range(5)] == [0, 1, 2, 0, 1]


def test_roundrobin_cursor_is_per_instance() -> None:
    a, b = RoundRobin(), RoundRobin()
    rb = batch(x=[1])
    a.route(rb, 3)
    a.route(rb, 3)
    assert b.route(rb, 3)[0][0] == 0  # a's cursor advancing did not move b's


# --- HashPartitioner: routing, conservation, co-location ---------------------------------------


def test_hash_q1_routes_whole_batch_but_validates_keys() -> None:
    # parallelism 1 sends the whole batch to instance 0 without bucketing, but still validates key
    # types — so a float key (rejected at parallelism above 1) is rejected at parallelism 1 too
    # (fail-fast), not silently working until scaled.
    valid = batch(key=["a", "b"])
    assert HashPartitioner(["key"]).route(valid, 1) == [(0, valid)]
    with pytest.raises(TypeError):
        HashPartitioner(["key"]).route(batch(key=[1.0, 2.0]), 1)


def test_hash_routes_disjoint_covering_subbatches() -> None:
    p = HashPartitioner(["key"])
    b = batch(rid=list(range(20)), key=[i % 4 for i in range(20)])
    routed = p.route(b, 3)
    assert all(0 <= idx < 3 for idx, _ in routed)
    assert all(sub.num_rows > 0 for _, sub in routed)  # no zero-row frame emitted
    rids = [r for _, sub in routed for r in sub.column("rid").to_pylist()]
    assert sorted(rids) == list(range(20))  # disjoint AND covers every row exactly once


def test_hash_co_locates_every_key() -> None:
    keys = ["a", "b", "c", "d", "alpha", "beta", "x", "y", "z"]
    for q in (2, 3, 5, 7):
        b = batch(key=[keys[i % len(keys)] for i in range(50)])
        where: dict[object, set[int]] = {}
        for idx, sub in HashPartitioner(["key"]).route(b, q):
            for k in sub.column("key").to_pylist():
                where.setdefault(k, set()).add(idx)
        for k, idxs in where.items():
            assert idxs == {stable_bucket((k,), q)}, (k, q, idxs)


def test_hash_multi_column_key_co_locates() -> None:
    b = batch(a=["x", "x", "y", "y", "x"], c=[1, 2, 1, 2, 1])
    where: dict[tuple, set[int]] = {}
    for idx, sub in HashPartitioner(["a", "c"]).route(b, 4):
        for a, c in zip(sub.column("a").to_pylist(), sub.column("c").to_pylist(), strict=True):
            where.setdefault((a, c), set()).add(idx)
    for key, idxs in where.items():
        assert idxs == {stable_bucket(key, 4)}, (key, idxs)


def test_hash_conserves_rows_and_tensors_under_fuzz() -> None:
    rng = random.Random(1234)
    for _ in range(200):
        n = rng.randint(1, 40)
        cardinality = rng.randint(1, 8)
        q = rng.choice([2, 3, 4, 5, 7])
        use_str = rng.random() < 0.5
        if use_str:
            keys = [f"k{rng.randrange(cardinality)}" for _ in range(n)]
        else:
            keys = [rng.randrange(cardinality) for _ in range(n)]
        imgs = [[[rng.randrange(256) for _ in range(2)] for _ in range(2)] for _ in range(n)]
        b = batch(rid=list(range(n)), key=keys, img=np.array(imgs, dtype=np.uint8))
        routed = HashPartitioner(["key"]).route(b, q)

        rids = [r for _, sub in routed for r in sub.column("rid").to_pylist()]
        assert sorted(rids) == list(range(n))  # exact partition: disjoint + covering
        assert sum(sub.num_rows for _, sub in routed) == n
        assert all(sub.num_rows > 0 for _, sub in routed)
        for _, sub in routed:
            assert is_tensor(sub.column("img").type)  # fixed_shape_tensor survives take()
            assert to_numpy(sub.column("img")).shape == (sub.num_rows, 2, 2)


def test_hash_routes_and_co_locates_null_keys() -> None:
    # A null key cell is a real key the keyed operators count (via value_counts / to_pylist), so the
    # shuffle must route it — not raise — and keep it co-located, like any other key.
    b = batch(key=["a", None, "b", None, "a", None])
    where: dict[object, set[int]] = {}
    for idx, sub in HashPartitioner(["key"]).route(b, 3):
        for k in sub.column("key").to_pylist():
            where.setdefault(k, set()).add(idx)
    assert None in where
    for k, idxs in where.items():
        assert idxs == {stable_bucket((k,), 3)}, (k, idxs)


# --- vectorized route == the per-row reference, byte-for-byte ----------------------------------
# The keyed route is vectorized (dictionary_encode -> bucket-per-distinct -> filter). These oracles
# pin it to the semantics of the original per-row loop EXACTLY: every row lands on the same instance,
# AND each sub-batch keeps its rows in input order (filter preserves order, as the per-row append did).


def _route_rid_map(partitioner: object, b: pa.RecordBatch, q: int) -> dict[int, list[int]]:
    """Each instance index -> the ``rid``s routed to it, in emitted order."""
    return {idx: sub.column("rid").to_pylist() for idx, sub in partitioner.route(b, q)}  # type: ignore[attr-defined]


def _reference_rid_map(
    b: pa.RecordBatch, q: int, key_columns: tuple[str, ...], instance_of: object
) -> dict[int, list[int]]:
    """The pre-vectorization semantics: a Python loop sending each row to ``instance_of(key)``, appended
    in row order. The independent oracle the vectorized route must reproduce byte-for-byte."""
    cols = [b.column(c).to_pylist() for c in key_columns]
    out: dict[int, list[int]] = {}
    for r in range(b.num_rows):
        inst = instance_of(tuple(col[r] for col in cols))  # type: ignore[operator]
        out.setdefault(inst, []).append(r)
    return out


def _random_key_column(rng: random.Random, n: int, kind: str) -> pa.Array:
    card = rng.randint(1, max(2, n))  # up to ~n distinct -> exercises the high-cardinality fold
    if kind == "str":
        vals: list[object] = [f"k{rng.randrange(card)}" for _ in range(n)]
        typ = pa.string()
    elif kind == "int":
        vals, typ = [rng.randrange(card) for _ in range(n)], pa.int64()
    elif kind == "bool":
        vals, typ = [rng.random() < 0.5 for _ in range(n)], pa.bool_()
    else:  # bytes
        vals, typ = [f"b{rng.randrange(card)}".encode() for _ in range(n)], pa.binary()
    if rng.random() < 0.4:  # sometimes punch nulls in — a null cell is a real, routable key
        vals = [None if rng.random() < 0.2 else v for v in vals]
    return pa.array(vals, typ)


def test_route_matches_per_row_reference_byte_identical_under_fuzz() -> None:
    rng = random.Random(20260628)
    kinds = ["str", "int", "bool", "bytes"]
    for _ in range(300):
        n = rng.randint(1, 60)
        kc = tuple(f"key{j}" for j in range(rng.randint(1, 3)))  # 1-3 key columns
        q = rng.choice([2, 3, 4, 5, 7])
        cols = {c: _random_key_column(rng, n, rng.choice(kinds)) for c in kc}
        b = batch(rid=list(range(n)), **cols)
        # Direct hash: the instance is stable_bucket of the key over the parallelism.
        assert _route_rid_map(HashPartitioner(kc), b, q) == _reference_rid_map(
            b, q, kc, lambda k, q=q: stable_bucket(k, q)
        )
        # Key-group indirection (at least as many groups as instances): the instance is the table
        # entry at stable_bucket of the key over the key-group count.
        table = tuple(i % q for i in range(q + rng.randrange(0, 6)))
        assert _route_rid_map(KeyGroupPartitioner(kc, table), b, q) == _reference_rid_map(
            b, q, kc, lambda k, table=table: table[stable_bucket(k, len(table))]
        )


def test_route_multi_column_high_cardinality_overflow_guard() -> None:
    # Three near-unique columns: the naive mixed-radix product (card0*card1*card2) is large, so this
    # exercises the per-fold re-encode that keeps the combo id within [0, num_rows). Must still match.
    n = 200
    b = batch(
        rid=list(range(n)),
        a=pa.array([f"a{i}" for i in range(n)], pa.string()),
        c=pa.array(list(range(n)), pa.int64()),
        d=pa.array([f"d{i}".encode() for i in range(n)], pa.binary()),
    )
    kc = ("a", "c", "d")
    for q in (2, 3, 5, 7):
        assert _route_rid_map(HashPartitioner(kc), b, q) == _reference_rid_map(
            b, q, kc, lambda k, q=q: stable_bucket(k, q)
        )


def test_route_empty_batch_returns_nothing() -> None:
    b = batch(rid=pa.array([], pa.int64()), key=pa.array([], pa.string()))
    assert HashPartitioner(["key"]).route(b, 3) == []
    assert KeyGroupPartitioner(["key"], (0, 1, 2)).route(b, 3) == []


# --- rejected key types ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "col",
    [
        pa.array([1.0, 2.0], pa.float64()),
        pa.array([1, 2], pa.timestamp("us")),
        pa.array([Decimal("1.5"), Decimal("2.5")], pa.decimal128(5, 2)),
    ],
)
def test_hash_rejects_unsupported_key_scalars(col: pa.Array) -> None:
    b = batch(key=col)
    with pytest.raises(TypeError):
        HashPartitioner(["key"]).route(b, 2)


def test_hash_requires_a_key_column() -> None:
    with pytest.raises(ValueError):
        HashPartitioner([])


# --- KeyGroupPartitioner: indirection through a group -> instance table -------------------------


def _route_map(partitioner: object, b: pa.RecordBatch, q: int) -> dict[int, int]:
    """Map each row's ``rid`` to the instance it was routed to."""
    out: dict[int, int] = {}
    for idx, sub in partitioner.route(b, q):  # type: ignore[attr-defined]
        for rid in sub.column("rid").to_pylist():
            out[rid] = idx
    return out


def test_keygroup_identity_table_equals_hash_partitioner() -> None:
    # With the key-group count equal to the parallelism and the identity table, every row routes to
    # the same instance as the direct hash. This is the equivalence that keeps 2a's
    # compiled-vs-legacy digest oracle green.
    rng = random.Random(99)
    for _ in range(200):
        n = rng.randint(1, 40)
        cardinality = rng.randint(1, 8)
        q = rng.choice([2, 3, 4, 5, 7])
        keys = [f"k{rng.randrange(cardinality)}" for _ in range(n)]
        b = batch(rid=list(range(n)), key=keys)
        identity = tuple(range(q))
        assert _route_map(KeyGroupPartitioner(["key"], identity), b, q) == _route_map(
            HashPartitioner(["key"]), b, q
        )


def test_keygroup_co_locates_and_conserves_for_g_ge_q() -> None:
    # For at least as many key groups as instances (multiples and non-multiples of the parallelism),
    # the table routes every key to exactly one instance and conserves every row — the
    # co-partitioning the keyed operators depend on.
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 40)
        cardinality = rng.randint(1, 10)
        q = rng.choice([2, 3, 4, 5])
        # at least as many key groups as instances, including non-multiples of the parallelism
        g = q + rng.randrange(0, 8)
        table = tuple(i % q for i in range(g))
        keys = [f"k{rng.randrange(cardinality)}" for _ in range(n)]
        b = batch(rid=list(range(n)), key=keys)
        routed = KeyGroupPartitioner(["key"], table).route(b, q)

        rids = [r for _, sub in routed for r in sub.column("rid").to_pylist()]
        assert sorted(rids) == list(range(n))  # disjoint AND covering
        assert all(0 <= idx < q for idx, _ in routed)
        where: dict[object, set[int]] = {}
        for idx, sub in routed:
            for k in sub.column("key").to_pylist():
                where.setdefault(k, set()).add(idx)
        assert all(len(instances) == 1 for instances in where.values()), where


def test_keygroup_q1_routes_whole_batch_but_validates_keys() -> None:
    valid = batch(key=["a", "b"])
    assert KeyGroupPartitioner(["key"], (0,)).route(valid, 1) == [(0, valid)]
    with pytest.raises(TypeError):
        KeyGroupPartitioner(["key"], (0,)).route(batch(key=[1.0, 2.0]), 1)


@pytest.mark.parametrize(
    "col",
    [
        pa.array([1.0, 2.0], pa.float64()),
        pa.array([1, 2], pa.timestamp("us")),
        pa.array([Decimal("1.5"), Decimal("2.5")], pa.decimal128(5, 2)),
    ],
)
def test_keygroup_rejects_unsupported_key_scalars(col: pa.Array) -> None:
    b = batch(key=col)
    with pytest.raises(TypeError):
        KeyGroupPartitioner(["key"], (0, 1)).route(b, 2)


def test_keygroup_requires_key_column_and_table() -> None:
    with pytest.raises(ValueError):
        KeyGroupPartitioner([], (0, 1))
    with pytest.raises(ValueError):
        KeyGroupPartitioner(["k"], ())
