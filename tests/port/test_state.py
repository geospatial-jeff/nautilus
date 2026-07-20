"""Characterization tests pinning the MVP in-memory state backend's observable behavior.

A future Rust port replaces :class:`InMemoryStateBackend`; these tests fix the contract it must
reproduce byte-for-byte and slot-for-slot. They focus on ``snapshot``/``restore`` (previously
uncovered) and the size-bookkeeping edge cases (stored ``None`` slots, per-namespace refcounts,
no-op clears) that a reimplementation is most likely to get subtly wrong.

Golden values here were derived by running the real backend and hardcoding what it produced.
"""

from __future__ import annotations

from collections.abc import Iterator

from nautilus.state import (
    InMemoryStateBackend,
    Key,
    Namespace,
    StateBackend,
    StateScope,
)


def _add(a: object, b: object) -> object:
    return a + b  # type: ignore[operator]


def _boom(a: object, b: object) -> object:
    raise AssertionError("reducer must not be called on a first write")


def _entries_dict(be: StateBackend, op: str, name: str) -> dict[tuple[Key, Namespace], object]:
    """(key, namespace) -> value for one operator/state-name, order-independent for comparison."""
    return {(key, ns): value for key, ns, value in be.entries(op, name)}


# --- (a) snapshot -> restore full round-trip, replacing pre-existing state -----------------------


def test_snapshot_restore_reproduces_every_slot_and_rebuilds_sizes():
    src = InMemoryStateBackend()
    src.put(StateScope("op", "v", ("a",), None), 1)
    src.put(StateScope("op", "v", ("a",), "ns1"), 2)  # same key, distinct namespace
    src.put(StateScope("op", "v", ("b",), None), 3)
    src.put(StateScope("op2", "w", ("c",), None), 4)  # a different operator entirely

    blob = src.snapshot()
    assert isinstance(blob, bytes)

    dst = InMemoryStateBackend()
    # Pre-existing junk that restore must throw away, not merge with.
    dst.put(StateScope("junk", "j", ("z",), None), 999)
    dst.restore(blob)

    # Every (op, name) group's entries come back exactly.
    assert _entries_dict(dst, "op", "v") == {
        (("a",), None): 1,
        (("a",), "ns1"): 2,
        (("b",), None): 3,
    }
    assert _entries_dict(dst, "op2", "w") == {(("c",), None): 4}
    # sizes() is rebuilt from the store: ('op','v') has 3 entries but only 2 distinct keys.
    assert dst.sizes() == src.sizes()
    assert dst.sizes() == {("op", "v"): (3, 2), ("op2", "w"): (1, 1)}
    # The pre-existing junk is gone — restore replaced the whole store.
    assert list(dst.entries("junk", "j")) == []
    assert ("junk", "j") not in dst.sizes()


def test_restore_replaces_and_get_reads_through():
    src = InMemoryStateBackend()
    src.put(StateScope("op", "v", ("a",), None), 42)
    blob = src.snapshot()

    dst = InMemoryStateBackend()
    dst.put(StateScope("op", "v", ("a",), None), -1)  # would collide if merged
    dst.put(StateScope("op", "v", ("gone",), None), -2)
    dst.restore(blob)

    assert dst.get(StateScope("op", "v", ("a",), None)) == 42
    assert dst.get(StateScope("op", "v", ("gone",), None)) is None


# --- (b) snapshot() == snapshot() for equal state (order-DEPENDENT here) -------------------------


def test_snapshot_is_deterministic_for_the_same_backend():
    be = InMemoryStateBackend()
    be.put(StateScope("op", "v", ("a",), None), 1)
    be.put(StateScope("op", "v", ("b",), None), 2)
    assert be.snapshot() == be.snapshot()


def test_snapshot_is_order_dependent_across_backends():
    # Two backends with identical logical state but different insertion order pickle DIFFERENTLY:
    # the snapshot is a pickle of the nested dict, which preserves insertion order. A Rust port that
    # emits an order-independent snapshot would diverge here, so the current behavior is pinned.
    a = InMemoryStateBackend()
    a.put(StateScope("op", "v", ("a",), None), 1)
    a.put(StateScope("op", "v", ("b",), None), 2)
    b = InMemoryStateBackend()
    b.put(StateScope("op", "v", ("b",), None), 2)
    b.put(StateScope("op", "v", ("a",), None), 1)

    assert _entries_dict(a, "op", "v") == _entries_dict(b, "op", "v")  # logically equal
    assert a.snapshot() != b.snapshot()  # but byte-unequal — order-dependent

    # Same insertion order does produce identical bytes.
    c = InMemoryStateBackend()
    c.put(StateScope("op", "v", ("a",), None), 1)
    c.put(StateScope("op", "v", ("b",), None), 2)
    assert a.snapshot() == c.snapshot()


# --- (c) restore rebuilds per-namespace distinct-key refcounts ----------------------------------


def test_restore_rebuilds_namespace_refcounts():
    src = InMemoryStateBackend()
    # One key held in three namespaces: entries + 3, distinct_keys + 1.
    src.put(StateScope("op", "v", ("k",), "ns1"), 1)
    src.put(StateScope("op", "v", ("k",), "ns2"), 1)
    src.put(StateScope("op", "v", ("k",), "ns3"), 1)

    dst = InMemoryStateBackend()
    dst.restore(src.snapshot())
    assert dst.sizes() == {("op", "v"): (3, 1)}

    # Clearing from one namespace leaves the key distinct until every namespace is cleared.
    dst.clear(StateScope("op", "v", ("k",), "ns1"))
    assert dst.sizes() == {("op", "v"): (2, 1)}
    dst.clear(StateScope("op", "v", ("k",), "ns2"))
    assert dst.sizes() == {("op", "v"): (1, 1)}
    dst.clear(StateScope("op", "v", ("k",), "ns3"))
    assert dst.sizes() == {("op", "v"): (0, 0)}


# --- (d) stored None is a real slot -------------------------------------------------------------


def test_stored_none_is_a_real_slot():
    be = InMemoryStateBackend()
    be.put(StateScope("op", "v", ("a",), None), None)

    # It occupies a slot: sizes counts it as one entry, one distinct key.
    assert be.sizes() == {("op", "v"): (1, 1)}
    # It appears in entries() with value None.
    assert list(be.entries("op", "v")) == [(("a",), None, None)]
    # But get() returns None — indistinguishable from a missing key by value alone.
    assert be.get(StateScope("op", "v", ("a",), None)) is None
    assert be.get(StateScope("op", "v", ("missing",), None)) is None


# --- (e) reduce_all first-write against a None-holding slot overwrites, never reduces ------------


def test_reduce_all_first_write_over_none_slot_skips_reducer():
    be = InMemoryStateBackend()
    be.put(StateScope("op", "r", ("a",), None), None)  # a stored None current
    # A None current is treated as a first write: the value is stored verbatim, reducer untouched.
    be.reduce_all("op", "r", [(("a",), 5)], _boom)
    assert be.get(StateScope("op", "r", ("a",), None)) == 5
    assert be.sizes() == {("op", "r"): (1, 1)}


# --- (f) a None-containing key tuple is first-class ---------------------------------------------


def test_none_containing_key_tuple_is_first_class():
    be = InMemoryStateBackend()
    be.reduce_all("op", "r", [((None,), 1), ((None,), 2)], _add)
    assert list(be.entries("op", "r")) == [((None,), None, 3)]
    assert be.sizes() == {("op", "r"): (1, 1)}


# --- (g) clear on a missing/never-written key and double-clear are silent no-ops ----------------


def test_clear_missing_key_is_a_noop():
    be = InMemoryStateBackend()
    # Never written: no group exists, so sizes() stays empty and no error is raised.
    be.clear(StateScope("op", "v", ("nope",), None))
    assert be.sizes() == {}


def test_double_clear_is_a_noop_and_sizes_stay_nonnegative():
    be = InMemoryStateBackend()
    be.put(StateScope("op", "v", ("a",), None), 1)
    be.clear(StateScope("op", "v", ("a",), None))
    before = be.sizes()
    be.clear(StateScope("op", "v", ("a",), None))  # second clear of the same key
    after = be.sizes()
    # The group lingers at zero counts (not removed from sizes); the second clear changes nothing.
    assert before == after == {("op", "v"): (0, 0)}
    entries, distinct = after[("op", "v")]
    assert entries >= 0 and distinct >= 0


# --- (h) a minimal get/put-only subclass's default reduce_all folds identically -----------------


class _MiniBackend(StateBackend):
    """A minimal backend using only get/put — exercises StateBackend.reduce_all's default fold."""

    def __init__(self) -> None:
        self._d: dict[tuple[str, str, Key, Namespace], object] = {}

    def get(self, scope: StateScope) -> object | None:
        return self._d.get((scope.operator_id, scope.name, scope.key, scope.namespace))

    def put(self, scope: StateScope, value: object) -> None:
        self._d[(scope.operator_id, scope.name, scope.key, scope.namespace)] = value

    def clear(self, scope: StateScope) -> None:
        self._d.pop((scope.operator_id, scope.name, scope.key, scope.namespace), None)

    def entries(self, operator_id: str, name: str) -> Iterator[tuple[Key, Namespace, object]]:
        for (op, nm, key, ns), value in self._d.items():
            if op == operator_id and nm == name:
                yield key, ns, value

    def snapshot(self) -> bytes:
        return b""

    def restore(self, blob: bytes) -> None:
        self._d = {}


def test_default_reduce_all_folds_like_in_memory_backend():
    items = [(("a",), 10), (("b",), 1), (("a",), 5), (("b",), 4)]

    mini = _MiniBackend()
    inmem = InMemoryStateBackend()
    mini.reduce_all("op", "r", items, _add)
    inmem.reduce_all("op", "r", items, _add)

    assert _entries_dict(mini, "op", "r") == _entries_dict(inmem, "op", "r")
    assert _entries_dict(inmem, "op", "r") == {(("a",), None): 15, (("b",), None): 5}


def test_default_reduce_all_first_write_on_none_matches():
    # Both backends: a pre-seeded None current is a first write in the default fold too.
    mini = _MiniBackend()
    inmem = InMemoryStateBackend()
    mini.put(StateScope("op", "r", ("a",), None), None)
    inmem.put(StateScope("op", "r", ("a",), None), None)
    mini.reduce_all("op", "r", [(("a",), 7)], _boom)
    inmem.reduce_all("op", "r", [(("a",), 7)], _boom)

    assert mini.get(StateScope("op", "r", ("a",), None)) == 7
    assert _entries_dict(mini, "op", "r") == _entries_dict(inmem, "op", "r")


# --- (i) entries() is scoped; clearing a whole namespace removes it from entries() --------------


def test_entries_do_not_leak_across_names_or_operators():
    be = InMemoryStateBackend()
    be.put(StateScope("op", "a", ("k1",), None), 1)
    be.put(StateScope("op", "b", ("k2",), None), 2)  # same operator, different state-name
    be.put(StateScope("op2", "a", ("k3",), None), 3)  # different operator, same state-name

    # entries('op','a') yields only its own slot — not 'op','b' nor 'op2','a'.
    assert list(be.entries("op", "a")) == [(("k1",), None, 1)]


def test_clearing_every_key_removes_the_namespace_from_entries():
    be = InMemoryStateBackend()
    be.put(StateScope("op", "a", ("k1",), "ns"), 1)
    be.put(StateScope("op", "a", ("k2",), "ns"), 2)
    assert _entries_dict(be, "op", "a") == {(("k1",), "ns"): 1, (("k2",), "ns"): 2}

    be.clear(StateScope("op", "a", ("k1",), "ns"))
    be.clear(StateScope("op", "a", ("k2",), "ns"))
    # The emptied namespace group is dropped, so entries() is empty...
    assert list(be.entries("op", "a")) == []
    # ...but sizes() keeps the (op, name) group at zero counts (it is NOT removed from sizes()).
    assert be.sizes() == {("op", "a"): (0, 0)}
