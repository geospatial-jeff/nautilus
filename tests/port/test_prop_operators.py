"""Tier 4 property-based conformance tests for operators.

Where Tier 1 (``test_operators.py``) pins hand-picked golden outputs, these pin *invariants* that must
hold over the WHOLE input space of the built-in operators in ``nautilus.operators`` — the ones a Rust port
could silently break on an untested input. Each test drives many randomized inputs from a FIXED seed
(deterministic, no hypothesis) and asserts the universally-quantified property, spanning the edge cases the
property names: batch boundaries, the vectorized fast path vs the general per-distinct-key path, empty /
None / whitespace-only rows, negative and dense integer keys, and the int-vs-bool scalar-type distinction.

Operators are driven the two ways the existing suite drives them: two-input operators (HashJoin) directly
through ``open`` / ``process_*`` / ``on_eos`` / ``close`` via :func:`_drive_join`; one-input operators
(KeyedAgg, KeyedCount, Tokenize) through ``open`` / ``process`` / ``on_eos`` via :func:`_drive_one`.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nautilus.core.operator import ListCollector, OperatorContext
from nautilus.operators import HashJoin, KeyedAgg, KeyedCount, Tokenize


def _drive_one(op, batches: list[pa.RecordBatch]) -> list[dict]:
    """Open a one-input operator, feed each batch, flush at EOS, and return the emitted rows as dicts."""
    coll = ListCollector()
    op.open(OperatorContext("o"))
    for b in batches:
        op.process(b, coll)
    op.on_eos(coll)
    return [r for out in coll.drain() for r in out.to_pylist()]


def _drive_join(join: HashJoin, steps: list[tuple[str, pa.RecordBatch]]) -> list[dict]:
    """Open the join, feed ``(side, batch)`` steps, flush at EOS, close, and return the emitted rows as
    dicts — the same driver ``test_operators.py`` uses, projected to rows."""
    coll = ListCollector()
    join.open(OperatorContext("j"))
    for side, b in steps:
        (join.process_left if side == "L" else join.process_right)(b, coll)
    join.on_eos(coll)
    join.close()
    return [r for out in coll.drain() for r in out.to_pylist()]


def test_keyed_agg_row_conservation() -> None:
    """For every input split into any batches with any agg func, KeyedAgg emits exactly one row per
    distinct key group (fast integer path and general path alike)."""
    rng = np.random.default_rng(1234)
    for trial in range(60):
        n_batches = int(rng.integers(1, 5))
        # Alternate the two int-key regimes and the string key, so both the value-indexed fast path
        # (dense non-negative int) and the general per-distinct-key path (string, and a sparse int that
        # _int_fast_ok demotes) are covered across trials.
        regime = trial % 3
        batches, distinct = [], set()
        for _ in range(n_batches):
            m = int(rng.integers(0, 40))
            if regime == 0:  # dense non-negative int → fast path
                kvals: list = rng.integers(0, 8, m).tolist()
                kcol = pa.array(kvals, pa.int64())
            elif regime == 1:  # sparse int (huge value) → fast path demotes to the dict
                kvals = (rng.integers(0, 8, m) + (1 << 40)).tolist()
                kcol = pa.array(kvals, pa.int64())
            else:  # string key → general path
                kvals = [f"g{x}" for x in rng.integers(0, 8, m)]
                kcol = pa.array(
                    kvals, pa.string()
                )  # type an empty batch (real sources carry a schema)
            distinct.update(kvals)
            v = pa.array(rng.normal(0, 1, m), pa.float64())
            batches.append(pa.record_batch({"k": kcol, "v": v}))
        # Every supported agg func at once — the output row count must not depend on which func is asked.
        aggs = {"s": ("v", "sum"), "a": ("v", "mean"), "c": ("v", "count"), "lo": ("v", "min")}
        rows = _drive_one(KeyedAgg(("k",), aggs), batches)
        assert len(rows) == len(distinct)
        assert {
            r["k"] for r in rows
        } == distinct  # exactly the distinct keys, no dupes, none dropped


def test_tokenize_row_count_equals_token_count() -> None:
    """For every input column, Tokenize emits exactly one row per Python ``str.split`` token; an
    empty / None / whitespace-only cell contributes zero rows."""
    rng = np.random.default_rng(1234)
    pieces = ["cat", "dog", "Fish", "a", "  ", "", "\t", "\n", "the quick", "  x  y  "]
    for lowercase in (True, False):
        for _ in range(80):
            m = int(rng.integers(0, 10))
            cells: list[str | None] = []
            for _ in range(m):
                if rng.random() < 0.15:
                    cells.append(None)  # a null cell must contribute nothing
                else:
                    k = int(rng.integers(0, 4))
                    cells.append(" ".join(rng.choice(pieces, size=k)) if k else "")
            b = pa.record_batch({"text": pa.array(cells, pa.string())})
            rows = _drive_one(Tokenize("text", lowercase=lowercase), [b])
            # Reference: exactly str.split() per non-null cell, lowercased iff the operator lowercases.
            expected = [
                (c.lower() if lowercase else c).split()
                for c in cells
                if c  # None and "" are falsy → zero rows, matching the operator's `if s:` guard
            ]
            flat = [w for toks in expected for w in toks]
            assert len(rows) == len(flat)
            assert [r["word"] for r in rows] == flat  # order preserved, one row per token


def test_hash_join_type_distinction() -> None:
    """For every keyed join, two keys match iff they are equal in value AND scalar type — an int ``1``
    and a bool ``True`` never cross-match, while equal-typed equal-valued keys always do."""
    rng = np.random.default_rng(1234)
    for _ in range(60):
        # An Arrow column holds one type, so int and bool keys live in separate join runs. Same-typed
        # runs (int↔int, bool↔bool) must match by value; the cross-typed run (int↔bool) must never match,
        # even though 1 == True in plain Python — the (type, value) interning keeps their id spaces apart.
        n_l, n_r = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        li, ri = rng.integers(0, 2, n_l).tolist(), rng.integers(0, 2, n_r).tolist()
        lb = (rng.integers(0, 2, n_l) == 1).tolist()
        rb = (rng.integers(0, 2, n_r) == 1).tolist()
        int_left = pa.record_batch({"id": pa.array(li, pa.int64()), "lval": pa.array(range(n_l))})
        int_right = pa.record_batch({"id": pa.array(ri, pa.int64()), "rval": pa.array(range(n_r))})
        bool_left = pa.record_batch({"id": pa.array(lb), "lval": pa.array(range(n_l))})
        bool_right = pa.record_batch({"id": pa.array(rb), "rval": pa.array(range(n_r))})

        def matches(steps: list[tuple[str, pa.RecordBatch]]) -> list[tuple[int, int]]:
            return sorted((r["lval"], r["rval"]) for r in _drive_join(HashJoin("id"), steps))

        # Same-typed joins: the plain-Python equi-join by value is the reference for both.
        assert matches([("L", int_left), ("R", int_right)]) == sorted(
            (i, j) for i, a in enumerate(li) for j, b in enumerate(ri) if a == b
        )
        assert matches([("L", bool_left), ("R", bool_right)]) == sorted(
            (i, j) for i, a in enumerate(lb) for j, b in enumerate(rb) if a == b
        )
        # Cross-typed join: int 1 vs bool True are different keys, so nothing matches — regardless of the
        # values drawn, even when an int 1 sits opposite a bool True (1 == True in Python).
        assert matches([("L", int_left), ("R", bool_right)]) == []
        assert matches([("L", bool_left), ("R", int_right)]) == []


def test_keyed_count_row_conservation() -> None:
    """For every input split into any batches, KeyedCount emits one row per distinct key and its counts
    sum to the number of input rows (including a null key, counted as its own group)."""
    rng = np.random.default_rng(1234)
    for trial in range(80):
        n_batches = int(rng.integers(1, 5))
        # Cover the fast path (dense non-negative int) and the demoting cases (a negative key, and a null
        # key counted as its own group). Keys stay dense: KeyedCount's fast path is deliberately unguarded
        # against a sparse/extreme key (its docstring), so a huge key value is outside its safe input space.
        allow_neg = trial % 2 == 0
        allow_null = trial % 3 == 0
        batches, total_rows = [], 0
        for _ in range(n_batches):
            m = int(rng.integers(0, 40))
            lo = -5 if allow_neg else 0
            arr = pa.array(rng.integers(lo, 9, m).tolist(), pa.int64())
            if allow_null and m:  # null a scattering of cells → the null-key group
                mask = pa.array(rng.random(m) < 0.25)
                arr = pc.if_else(mask, pa.scalar(None, pa.int64()), arr)
            batches.append(pa.record_batch({"k": arr}))
            total_rows += m
        rows = _drive_one(KeyedCount("k"), batches)
        assert sum(r["count"] for r in rows) == total_rows  # row conservation
        keys = [r["k"] for r in rows]
        assert len(keys) == len(set(keys))  # one row per distinct key (None is its own key)
