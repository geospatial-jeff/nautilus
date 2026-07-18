"""Keyed state: per-key, per-namespace storage with a pluggable backend.

State is scoped by ``(operator_id, name, key, namespace)``. The *key* is the partitioning key (a
tuple of Python values); the *namespace* separates sub-states of one key — a backend capability no
built-in operator sets today. Access goes through a :class:`KeyContext` captured by each typed handle,
so there is no shared mutable "current key" cursor that could race under async — a handle always refers
to exactly the key/namespace it was created for.

The MVP backend is a plain in-memory dict. The :class:`StateBackend` ABC declares
``snapshot``/``restore`` from day one so a spilling/checkpointing backend can be added later without
changing operators.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import cast

Key = tuple[object, ...]
Namespace = object | None


@dataclass(frozen=True)
class StateScope:
    """Fully-qualified address of one piece of keyed state."""

    operator_id: str
    name: str
    key: Key
    namespace: Namespace = None


@dataclass(frozen=True)
class KeyContext:
    """The key (and optional namespace) a set of state handles is bound to."""

    key: Key
    namespace: Namespace = None


# --- Backend -----------------------------------------------------------------------------------


class StateBackend(ABC):
    """Pluggable per-key state store. Single-writer (the owning actor's event loop); lock-free."""

    @abstractmethod
    def get(self, scope: StateScope) -> object | None: ...

    @abstractmethod
    def put(self, scope: StateScope, value: object) -> None: ...

    @abstractmethod
    def clear(self, scope: StateScope) -> None: ...

    @abstractmethod
    def entries(self, operator_id: str, name: str) -> Iterator[tuple[Key, Namespace, object]]:
        """Iterate ``(key, namespace, value)`` for one operator/state-name — how a keyed operator
        enumerates its state to flush it at end of stream. The caller must NOT mutate this state during
        iteration — collect the keys to change, then clear/put them afterward (as the keyed operators
        do), so a backend may stream entries lazily rather than copy the whole store."""

    def sizes(self) -> dict[tuple[str, str], tuple[int, int]]:
        """Per ``(operator_id, name)``: ``(entries, distinct_keys)`` currently held. ``entries`` counts
        every ``(key, namespace)`` slot — for a keyed aggregation, one per key.
        The actor samples this to emit ``state.entries`` / ``state.keys``; a backend that cannot report
        cheaply returns ``{}`` (the default) rather than walking its store on the hot path."""
        return {}

    def reduce_all(
        self,
        operator_id: str,
        name: str,
        items: Iterable[tuple[Key, object]],
        reducer: Callable[[object, object], object],
    ) -> None:
        """Fold many ``(key, value)`` pairs into reducing state in one pass, at namespace ``None`` (the
        only namespace the built-ins use) — the bulk form of :meth:`ReducingState.add` for a whole batch,
        so a keyed aggregation folds a batch's per-key partials without a ``KeyContext`` and a handle per
        distinct key. The default routes through :meth:`get`/:meth:`put`; a backend overrides it for a
        tighter loop (see :class:`InMemoryStateBackend`). A ``None`` current is a first write, matching
        :meth:`ReducingState.add`."""
        for key, value in items:
            scope = StateScope(operator_id, name, key, None)
            cur = self.get(scope)
            self.put(scope, value if cur is None else reducer(cur, value))

    @abstractmethod
    def snapshot(self) -> bytes: ...

    @abstractmethod
    def restore(self, blob: bytes) -> None: ...


@dataclass
class InMemoryStateBackend(StateBackend):
    """Dict-backed state. MVP only — unbounded memory; documented limitation.

    The store is nested — ``(operator_id, name, namespace) -> {key: value}`` — so the keyed-aggregation
    hot path (:meth:`reduce_all`) folds a whole batch with one *inner-dict* update per key, building and
    hashing no :class:`StateScope`. Incremental per-``(operator_id, name)`` counts of entries and distinct
    keys keep :meth:`sizes` O(state-names); they move only when a key is added or removed (a repeated fold
    touches no counter), so the per-record path stays cheap.
    """

    #: (operator_id, name, namespace) -> {partition key -> value}.
    _store: dict[tuple[str, str, Namespace], dict[Key, object]] = field(default_factory=dict)
    #: (operator_id, name) -> live entry count.
    _entry_count: dict[tuple[str, str], int] = field(default_factory=dict)
    #: (operator_id, name) -> {key: number of namespaces holding that key}; a key is distinct while > 0.
    _key_count: dict[tuple[str, str], dict[Key, int]] = field(default_factory=dict)

    def get(self, scope: StateScope) -> object | None:
        sub = self._store.get((scope.operator_id, scope.name, scope.namespace))
        return None if sub is None else sub.get(scope.key)

    def put(self, scope: StateScope, value: object) -> None:
        outer = (scope.operator_id, scope.name, scope.namespace)
        sub = self._store.get(outer)
        if sub is None:
            sub = self._store[outer] = {}
        if scope.key not in sub:  # new slot: bump the size counters (a repeat put skips it)
            self._track_add(scope.operator_id, scope.name, scope.key)
        sub[scope.key] = value

    def reduce_all(
        self,
        operator_id: str,
        name: str,
        items: Iterable[tuple[Key, object]],
        reducer: Callable[[object, object], object],
    ) -> None:
        # Only a new key touches the size counters. Semantics match ReducingState.add — a None current
        # (whether unset or a stored None) is a first write.
        outer = (operator_id, name, None)
        sub = self._store.get(outer)
        if sub is None:
            sub = self._store[outer] = {}
        for key, value in items:
            if key in sub:
                cur = sub[key]
                sub[key] = value if cur is None else reducer(cur, value)
            else:
                sub[key] = value
                self._track_add(operator_id, name, key)

    def clear(self, scope: StateScope) -> None:
        outer = (scope.operator_id, scope.name, scope.namespace)
        sub = self._store.get(outer)
        if sub is not None and scope.key in sub:
            del sub[scope.key]
            self._track_remove(scope.operator_id, scope.name, scope.key)
            if not sub:  # drop the empty namespace group so entries()/iteration skip it
                del self._store[outer]

    def entries(self, operator_id: str, name: str) -> Iterator[tuple[Key, Namespace, object]]:
        # Lazy, and touches only the matching (operator_id, name) groups — not the whole store. Callers
        # collect-then-clear (the ABC contract), so the store is not mutated mid-iteration.
        for (op, nm, ns), sub in self._store.items():
            if op == operator_id and nm == name:
                for key, value in sub.items():
                    yield key, ns, value

    def sizes(self) -> dict[tuple[str, str], tuple[int, int]]:
        return {
            name: (count, len(self._key_count.get(name, {})))
            for name, count in self._entry_count.items()
        }

    def _track_add(self, operator_id: str, name: str, key: Key) -> None:
        nm = (operator_id, name)
        self._entry_count[nm] = self._entry_count.get(nm, 0) + 1
        keys = self._key_count.setdefault(nm, {})
        keys[key] = keys.get(key, 0) + 1

    def _track_remove(self, operator_id: str, name: str, key: Key) -> None:
        nm = (operator_id, name)
        self._entry_count[nm] = self._entry_count.get(nm, 1) - 1
        keys = self._key_count.get(nm, {})
        remaining = keys.get(key, 1) - 1
        if remaining <= 0:
            keys.pop(key, None)
        else:
            keys[key] = remaining

    def snapshot(self) -> bytes:
        return pickle.dumps(self._store, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, blob: bytes) -> None:
        self._store = pickle.loads(blob)
        self._entry_count = {}
        self._key_count = {}
        for (op, nm, _ns), sub in self._store.items():  # rebuild the size counters from the store
            for key in sub:
                self._track_add(op, nm, key)


# --- Typed handles -----------------------------------------------------------------------------


class _Handle:
    __slots__ = ("_backend", "_operator_id", "_name", "_kctx")

    def __init__(
        self, backend: StateBackend, operator_id: str, name: str, kctx: KeyContext
    ) -> None:
        self._backend = backend
        self._operator_id = operator_id
        self._name = name
        self._kctx = kctx

    def _scope(self) -> StateScope:
        return StateScope(self._operator_id, self._name, self._kctx.key, self._kctx.namespace)


class ValueState[T](_Handle):
    """A single value per key/namespace."""

    def value(self) -> T | None:
        return cast("T | None", self._backend.get(self._scope()))

    def update(self, value: T) -> None:
        self._backend.put(self._scope(), value)

    def clear(self) -> None:
        self._backend.clear(self._scope())


class ReducingState[T](_Handle):
    """A value folded with an associative ``reducer`` as elements are added."""

    def __init__(
        self,
        backend: StateBackend,
        operator_id: str,
        name: str,
        kctx: KeyContext,
        reducer: Callable[[T, T], T],
    ) -> None:
        super().__init__(backend, operator_id, name, kctx)
        self._reducer = reducer

    def add(self, value: T) -> None:
        scope = self._scope()
        current = cast("T | None", self._backend.get(scope))
        self._backend.put(scope, value if current is None else self._reducer(current, value))

    def get(self) -> T | None:
        return cast("T | None", self._backend.get(self._scope()))

    def clear(self) -> None:
        self._backend.clear(self._scope())


# ListState / MapState were removed: they had no callers or accessors and their in-place mutation
# (append/setitem without a re-put) violated the StateBackend get-then-put contract a future copying or
# spilling backend will rely on. Add them back — backend-correct, with OperatorContext accessors and
# tests — when an operator actually needs list/map state.
