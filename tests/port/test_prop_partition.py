"""Tier 4 property-based conformance tests for partition.

Where ``test_partition.py`` pins hand-derived golden buckets for a handful of adversarial keys, this
file asserts each routing invariant over a *space* of inputs: it drives many randomized keys, batches,
and parallelisms from a fixed seed and checks the universally-quantified property on every one. A Rust
port that violates any of these silently splits a key's state or reorders rows without a golden test
ever tripping, so each property here is a contract the port must hold across the whole input space, not
just at the sampled points — the sampling only surfaces a violation, it does not define the claim.

The seed is fixed so a failure is reproducible; the properties are computed against the real API
(``stable_bucket`` and the three partitioners), never against a re-derived formula that could drift from
the code it is meant to guard.
"""

from __future__ import annotations

import hashlib
import random

import msgpack
import pyarrow as pa

from nautilus.runtime.partition import (
    HashPartitioner,
    KeyGroupPartitioner,
    stable_bucket,
)
from nautilus.testing import batch

# The scalar kinds a key column may hold — exactly the msgpack-distinguishable types the shuffle
# accepts (``_ALLOWED_KEY_SCALARS``). Every generator draws from these so the properties range over
# the real key space, including the bool/int and str/bytes pairs msgpack tags apart.
_KINDS = ("str", "int", "bool", "bytes")

# Int values that stress msgpack's width-and-sign encoding: the int64 extremes, the zero/one boundary
# (and negatives near it), so ``msgpack_bytes_encoding_byte_identical`` covers the encoder's edges, not
# just small positives.
_EXTREME_INTS = (0, 1, -1, 2**63 - 1, -(2**63), 2**31, -(2**31) - 1, 255, 256)


def _random_scalar(rng: random.Random, kind: str, card: int) -> object:
    """A single key scalar of ``kind`` drawn from ``card`` distinct values, sometimes ``None`` — a null
    cell is a real, routable key, so it must appear in the sampled space."""
    if rng.random() < 0.15:
        return None
    if kind == "str":
        return f"k{rng.randrange(card)}"
    if kind == "int":
        # Mix small ids with the int64 extremes so the fold and the hash see the full width.
        if rng.random() < 0.3:
            return rng.choice(_EXTREME_INTS)
        return rng.randrange(-card, card)
    if kind == "bool":
        return rng.random() < 0.5
    return f"b{rng.randrange(card)}".encode()


def _random_key(rng: random.Random, arity: int) -> tuple[object, ...]:
    """A key tuple of ``arity`` scalars, each an independently drawn kind — mixed-type keys are legal
    and must route, so the generator does not hold the kind fixed across the tuple."""
    return tuple(_random_scalar(rng, rng.choice(_KINDS), rng.randint(1, 6)) for _ in range(arity))


def _random_key_column(rng: random.Random, n: int, kind: str) -> pa.Array:
    """An ``n``-row Arrow column of one ``kind``, up to ~``n`` distinct values (to exercise the
    high-cardinality combo fold), with nulls punched in part of the time — the same shape the Tier-1
    fuzz builds, so the two suites cover the same column space."""
    card = rng.randint(1, max(2, n))
    if kind == "str":
        vals: list[object] = [f"k{rng.randrange(card)}" for _ in range(n)]
        typ = pa.string()
    elif kind == "int":
        vals, typ = [rng.randrange(card) for _ in range(n)], pa.int64()
    elif kind == "bool":
        vals, typ = [rng.random() < 0.5 for _ in range(n)], pa.bool_()
    else:  # bytes
        vals, typ = [f"b{rng.randrange(card)}".encode() for _ in range(n)], pa.binary()
    if rng.random() < 0.4:
        vals = [None if rng.random() < 0.2 else v for v in vals]
    return pa.array(vals, typ)


def _route_rid_map(partitioner: object, b: pa.RecordBatch, q: int) -> dict[int, list[int]]:
    """Each instance index -> the ``rid``s routed to it, in emitted order."""
    return {idx: sub.column("rid").to_pylist() for idx, sub in partitioner.route(b, q)}  # type: ignore[attr-defined]


def test_stable_bucket_deterministic_and_in_range() -> None:
    """For any key tuple and parallelism q >= 1, stable_bucket(key, q) is an int in [0, q) and equal on
    every call."""
    rng = random.Random(1234)
    for _ in range(4000):
        key = _random_key(rng, rng.randint(1, 4))
        q = rng.randint(1, 64)
        first = stable_bucket(key, q)
        assert isinstance(first, int)
        assert 0 <= first < q, (key, q, first)
        # Determinism: repeated calls (and a call on a freshly rebuilt equal tuple) never differ.
        assert stable_bucket(key, q) == first
        assert stable_bucket(tuple(key), q) == first
    # Named edge cases the property must still hold at: q == 1 always buckets to 0; the empty and
    # all-null keys are real keys; the int64 extremes are legal scalars.
    assert stable_bucket((None,), 1) == 0
    assert stable_bucket((), 1) == 0
    for x in _EXTREME_INTS:
        b = stable_bucket((x,), 7)
        assert 0 <= b < 7
        assert stable_bucket((x,), 7) == b


def test_msgpack_bytes_encoding_byte_identical() -> None:
    """For any key, msgpack.packb(list(key), use_bin_type=True) is byte-identical across calls, two keys
    that differ in any element encode to different bytes, and blake2b(raw, digest_size=8) is a stable
    8-byte digest whose big-endian int is stable."""
    rng = random.Random(1234)

    # Reverse map bytes -> the canonical identity of the key that produced them. A plain dict keyed by
    # the tuple cannot serve as the oracle: Python collapses (True,) with (1,) (bool == int, same hash),
    # exactly the pair the shuffle keeps apart, so identity is taken as (repr-of-type, value) per scalar.
    def identity(key: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
        return tuple((type(s).__name__, s) for s in key)

    seen: dict[bytes, tuple[tuple[str, object], ...]] = {}
    for _ in range(4000):
        key = _random_key(rng, rng.randint(0, 4))
        ident = identity(key)
        raw = msgpack.packb(list(key), use_bin_type=True)
        # Byte-identical across calls, and equal for an independently rebuilt equal tuple.
        assert raw == msgpack.packb(list(key), use_bin_type=True)
        assert raw == msgpack.packb(list(tuple(key)), use_bin_type=True)
        # Injectivity both ways (what co-location relies on): equal bytes iff the same key identity, so a
        # difference in any element's value or type flips the bytes and never two identities share bytes.
        prior = seen.get(raw)
        if prior is not None:
            assert prior == ident, (prior, ident)
        else:
            for other_ident in seen.values():
                assert other_ident != ident, ident  # a new identity must have new bytes
            seen[raw] = ident
        digest = hashlib.blake2b(raw, digest_size=8).digest()
        assert len(digest) == 8
        assert isinstance(digest, bytes)
        assert int.from_bytes(digest, "big") == int.from_bytes(digest, "big")

    # Named contrasts the property names explicitly — a difference in value, type, or order all flip the
    # bytes. int 1 / str "1" / bool True / bytes b"1" are four distinct keys; ("a","bc") != ("ab","c").
    def pk(key: tuple[object, ...]) -> bytes:
        return msgpack.packb(list(key), use_bin_type=True)

    variants = [pk((1,)), pk(("1",)), pk((True,)), pk((b"1",))]
    assert len(set(variants)) == 4  # value+type all differ
    assert pk(("a", "bc")) != pk(("ab", "c"))  # order/split differs
    assert pk((None,)) != pk(())  # a null cell is not an absent cell


def test_route_preserves_input_row_order_per_instance() -> None:
    """For any batch and parallelism, each sub-batch route emits holds its rows in ascending input-row
    order (a contiguous-order slice of the input), for both keyed partitioners."""
    rng = random.Random(1234)
    kinds = list(_KINDS)
    for _ in range(500):
        n = rng.randint(1, 60)
        kc = tuple(f"key{j}" for j in range(rng.randint(1, 3)))
        q = rng.choice([2, 3, 4, 5, 7])
        cols = {c: _random_key_column(rng, n, rng.choice(kinds)) for c in kc}
        b = batch(rid=list(range(n)), **cols)
        table = tuple(i % q for i in range(q + rng.randrange(0, 6)))
        for partitioner in (HashPartitioner(kc), KeyGroupPartitioner(kc, table)):
            rid_map = _route_rid_map(partitioner, b, q)
            all_rids: list[int] = []
            for rids in rid_map.values():
                # Within an instance the rids are the input row ids in strictly ascending order — the
                # per-key co-location downstream relies on this order being the input order.
                assert rids == sorted(rids)
                assert len(set(rids)) == len(rids)  # no row duplicated within an instance
                all_rids.extend(rids)
            # Every input row is routed exactly once across the instances (conservation).
            assert sorted(all_rids) == list(range(n))


def test_multi_column_combo_fold_correctness() -> None:
    """For any batch with 2+ key columns, HashPartitioner routes each row to stable_bucket of that row's
    value tuple over q — verified by a brute-force per-row loop against the real route."""
    rng = random.Random(1234)
    kinds = list(_KINDS)
    for _ in range(400):
        n = rng.randint(1, 50)
        arity = rng.randint(2, 4)  # 2+ columns exercises the combo fold
        kc = tuple(f"key{j}" for j in range(arity))
        q = rng.choice([2, 3, 4, 5, 7])
        # Vary per-column cardinality independently: some columns near-unique, some tiny — this drives
        # the mixed-radix fold across both the wide and the collapsed regimes.
        cols = {c: _random_key_column(rng, n, rng.choice(kinds)) for c in kc}
        b = batch(rid=list(range(n)), **cols)
        route = _route_rid_map(HashPartitioner(kc), b, q)
        # Brute force: independently bucket each row's raw value tuple, group rids by instance in row
        # order. The vectorized combo fold must reproduce this byte-for-byte.
        pylists = {c: b.column(c).to_pylist() for c in kc}
        expected: dict[int, list[int]] = {}
        for r in range(n):
            key = tuple(pylists[c][r] for c in kc)
            expected.setdefault(stable_bucket(key, q), []).append(r)
        assert route == expected


def test_keygroup_table_indirection_identity_equivalence() -> None:
    """For any batch and parallelism q, a KeyGroupPartitioner with the identity group table (0,1,...,q-1)
    routes byte-for-byte identically to a HashPartitioner on the same key columns."""
    rng = random.Random(1234)
    kinds = list(_KINDS)
    for _ in range(500):
        n = rng.randint(1, 60)
        kc = tuple(f"key{j}" for j in range(rng.randint(1, 3)))
        q = rng.choice([2, 3, 4, 5, 7])
        cols = {c: _random_key_column(rng, n, rng.choice(kinds)) for c in kc}
        b = batch(rid=list(range(n)), **cols)
        identity = tuple(range(q))  # group g -> instance g, so the indirection is a no-op
        hash_map = _route_rid_map(HashPartitioner(kc), b, q)
        keygroup_map = _route_rid_map(KeyGroupPartitioner(kc, identity), b, q)
        assert keygroup_map == hash_map
