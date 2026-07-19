"""Union: the keyless two-input concatenation (SQL ``UNION ALL``).

Unit tests drive the operator directly to pin that it forwards every batch from both sides unchanged,
keeps duplicates, and rejects a schema mismatch between the sides. The DSL-level behavior (topology, the
keyless edges, parallel/serial and distributed equivalence, the self-union guard) lives in test_dsl.py.
"""

from __future__ import annotations

from collections import Counter

import pytest

from nautilus.core.operator import ListCollector, OperatorContext
from nautilus.operators import Union
from nautilus.testing import batch


def test_union_forwards_both_sides_and_keeps_duplicates() -> None:
    op = Union()
    coll = ListCollector()
    op.open(OperatorContext("u"))
    op.process_left(batch(id=[1, 2]), coll)
    op.process_right(batch(id=[2, 3]), coll)  # 2 is on both sides
    op.on_eos(coll)
    op.close()
    ids = [r["id"] for b in coll.drain() for r in b.to_pylist()]
    assert Counter(ids) == Counter([1, 2, 2, 3])  # every row, duplicates kept


def test_union_rejects_a_schema_mismatch_between_sides() -> None:
    # A single output stream carries one schema; the first batch fixes it and a differing side raises.
    op = Union()
    coll = ListCollector()
    op.open(OperatorContext("u"))
    op.process_left(batch(id=[1]), coll)
    with pytest.raises(ValueError, match="share a schema"):
        op.process_right(batch(other=[1]), coll)
