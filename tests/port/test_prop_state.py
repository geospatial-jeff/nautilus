"""Tier 4 property-based conformance tests for state.

These pin invariants of the MVP in-memory state backend that must hold over the WHOLE input space,
not just the hand-picked slots the Tier-1 characterization tests (``test_state.py``) fix. A future
Rust port replaces :class:`InMemoryStateBackend`; a property that fails only on some inputs is a
divergence a port could ship silently, so each test drives many random inputs from a FIXED seed and
asserts the universally-quantified property on every one.

Inputs are drawn to hit the edges the invariants name: empty batches, stored ``None`` currents,
extreme int64 values, ``None`` inside key tuples, and one key spread across several namespaces.
"""

from __future__ import annotations

import random

from nautilus.state import (
    InMemoryStateBackend,
    Key,
    KeyContext,
    Namespace,
    ReducingState,
    StateBackend,
    StateScope,
)

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def _add(a: object, b: object) -> object:
    return a + b  # type: ignore[operator]


def _boom(a: object, b: object) -> object:
    raise AssertionError("reducer must not be called on a first write")


def _entries_dict(be: StateBackend, op: str, name: str) -> dict[tuple[Key, Namespace], object]:
    """(key, namespace) -> value for one operator/state-name, order-independent for comparison."""
    return {(key, ns): value for key, ns, value in be.entries(op, name)}


def _rand_value(rng: random.Random) -> int:
    """A value drawn to cover the int64 extremes a sum-reducer could overflow in a Rust port."""
    return rng.choice([INT64_MIN, INT64_MAX, 0, -1, 1, rng.randint(-(10**9), 10**9)])


def _rand_key(rng: random.Random) -> Key:
    """A partition key mixing arity, str/int members, and a ``None`` member (a first-class key value)."""
    members: list[object] = [
        rng.choice(["a", "b", "c", "", "unicode-é"]),
        rng.choice([None, 0, -1, rng.randint(-5, 5)]),
    ]
    return tuple(members[: rng.randint(1, 2)])


def test_reduce_all_batch_equivalence_with_sequential_add() -> None:
    """reduce_all(op,name,batch,f) leaves the same per-key state as adding the batch via ReducingState.add in order."""
    rng = random.Random(1234)
    for _ in range(400):
        n = rng.randint(0, 12)  # includes the empty-batch edge
        items = [(_rand_key(rng), _rand_value(rng)) for _ in range(n)]

        bulk = InMemoryStateBackend()
        bulk.reduce_all("op", "r", items, _add)

        seq = InMemoryStateBackend()
        for key, value in items:
            # ReducingState.add binds to one key via a KeyContext; reduce_all folds at namespace None.
            handle: ReducingState[object] = ReducingState(
                seq, "op", "r", KeyContext(key, None), _add
            )
            handle.add(value)

        assert _entries_dict(bulk, "op", "r") == _entries_dict(seq, "op", "r")
        assert bulk.sizes() == seq.sizes()


def test_reducing_state_add_first_write_skips_reducer() -> None:
    """ReducingState.add on a None current stores the value verbatim; the reducer runs only from the 2nd write."""
    rng = random.Random(1234)
    for _ in range(400):
        be = InMemoryStateBackend()
        key = _rand_key(rng)
        first = _rand_value(rng)
        handle: ReducingState[object] = ReducingState(be, "op", "r", KeyContext(key, None), _boom)

        # First write: current is None (unset), so _boom must NOT fire and the value lands verbatim.
        handle.add(first)
        assert handle.get() == first

        # A pre-stored None is also a first write: overwrite it verbatim, still no reducer call.
        be2 = InMemoryStateBackend()
        be2.put(StateScope("op", "r", key, None), None)
        handle2: ReducingState[object] = ReducingState(be2, "op", "r", KeyContext(key, None), _boom)
        handle2.add(first)
        assert handle2.get() == first

        # Second write onto a non-None current does call the reducer.
        summed: ReducingState[object] = ReducingState(be, "op", "r", KeyContext(key, None), _add)
        second = _rand_value(rng)
        summed.add(second)
        assert summed.get() == first + second


def test_put_idempotency_size_counts() -> None:
    """Two puts on one scope bump the entry/distinct-key counters once (on the first), never on the repeat."""
    rng = random.Random(1234)
    for _ in range(400):
        be = InMemoryStateBackend()
        key = _rand_key(rng)
        ns: Namespace = rng.choice([None, "ns1", "ns2"])
        scope = StateScope("op", "v", key, ns)

        v1, v2 = _rand_value(rng), _rand_value(rng)
        be.put(scope, v1)
        after_first = be.sizes()
        assert after_first == {("op", "v"): (1, 1)}  # one slot, one distinct key

        be.put(scope, v2)  # same scope again
        after_second = be.sizes()
        assert after_second == after_first  # repeat put moves no counter
        assert be.get(scope) == v2  # ...but the value is updated


def test_entries_yields_all_namespaces() -> None:
    """entries(op,name) yields a key once per (key, namespace) pair when the key lives in several namespaces."""
    rng = random.Random(1234)
    for _ in range(400):
        be = InMemoryStateBackend()
        key = _rand_key(rng)
        namespaces = rng.sample(["a", "b", "c", "d"], rng.randint(1, 4))
        expected: dict[tuple[Key, Namespace], object] = {}
        for ns in namespaces:
            value = _rand_value(rng)
            be.put(StateScope("op", "v", key, ns), value)
            expected[(key, ns)] = value

        yielded = list(be.entries("op", "v"))
        # One (key, namespace, value) triple per namespace — no dedup, no dropped namespace.
        assert len(yielded) == len(namespaces)
        assert _entries_dict(be, "op", "v") == expected
        # entries + distinct: one entry per namespace, but a single distinct key across all of them.
        assert be.sizes() == {("op", "v"): (len(namespaces), 1)}


def test_snapshot_determinism_for_insertion_order() -> None:
    """For a fixed insertion order, consecutive snapshot() calls on the same backend are byte-identical."""
    rng = random.Random(1234)
    for _ in range(400):
        be = InMemoryStateBackend()
        n = rng.randint(0, 12)  # includes the empty-store edge
        for _ in range(n):
            key = _rand_key(rng)
            ns: Namespace = rng.choice([None, "ns1", "ns2"])
            be.put(StateScope("op", "v", key, ns), _rand_value(rng))

        # Same backend, same order already applied: two reads must produce identical bytes.
        assert be.snapshot() == be.snapshot()
        # Restoring from the blob and re-snapshotting reproduces the same bytes (order preserved).
        restored = InMemoryStateBackend()
        restored.restore(be.snapshot())
        assert restored.snapshot() == be.snapshot()
