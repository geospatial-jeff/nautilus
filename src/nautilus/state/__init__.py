"""Keyed state: per-key, per-namespace storage with a pluggable backend.

State is scoped by ``(operator_id, name, key, namespace)``. The *key* is the partitioning key (a
tuple of Python values); the *namespace* distinguishes sub-states of the same key, e.g. one window
instance. Access goes through a :class:`KeyContext` captured by each typed handle, so there is no
shared mutable "current key" cursor that could race under async — a handle always refers to exactly
the key/namespace it was created for.

The MVP backend is a plain in-memory dict. The :class:`StateBackend` ABC declares
``snapshot``/``restore`` from day one so a spilling/checkpointing backend can be added later without
changing operators.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

T = TypeVar("T")

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
        """Iterate ``(key, namespace, value)`` for one operator/state-name. Used to enumerate, e.g.,
        all open windows when flushing. The caller must NOT mutate this state during iteration — collect
        the keys to change, then clear/put them afterward (as the keyed operators do), so a backend may
        stream entries lazily rather than copy the whole store."""

    def sizes(self) -> dict[tuple[str, str], tuple[int, int]]:
        """Per ``(operator_id, name)``: ``(entries, distinct_keys)`` currently held. ``entries`` counts
        every ``(key, namespace)`` slot — for a keyed aggregation, one per key.
        The actor samples this to emit ``state.entries`` / ``state.keys``; a backend that cannot report
        cheaply returns ``{}`` (the default) rather than walking its store on the hot path."""
        return {}

    @abstractmethod
    def snapshot(self) -> bytes: ...

    @abstractmethod
    def restore(self, blob: bytes) -> None: ...


@dataclass
class InMemoryStateBackend(StateBackend):
    """Dict-backed state. MVP only — unbounded memory; documented limitation.

    Alongside the store it keeps incremental per-``(operator_id, name)`` counts of entries and distinct
    keys, updated on ``put``/``clear`` of a *new*/removed slot, so :meth:`sizes` is O(state-names) and
    needs no walk of the store. The bookkeeping adds one membership test per ``put`` (an existing key's
    repeated ``put`` — every reducing-state fold — touches no counter), keeping the per-record path cheap.
    """

    _store: dict[StateScope, object] = field(default_factory=dict)
    #: (operator_id, name) -> live entry count.
    _entry_count: dict[tuple[str, str], int] = field(default_factory=dict)
    #: (operator_id, name) -> {key: number of namespaces holding that key}; a key is distinct while > 0.
    _key_count: dict[tuple[str, str], dict[Key, int]] = field(default_factory=dict)

    def get(self, scope: StateScope) -> object | None:
        return self._store.get(scope)

    def put(self, scope: StateScope, value: object) -> None:
        if (
            scope not in self._store
        ):  # a new slot — update the size counters (existing folds skip this)
            self._track_add(scope)
        self._store[scope] = value

    def clear(self, scope: StateScope) -> None:
        if scope in self._store:
            self._track_remove(scope)
        self._store.pop(scope, None)

    def entries(self, operator_id: str, name: str) -> Iterator[tuple[Key, Namespace, object]]:
        # Lazy: no full-store copy. Callers collect-then-clear (see the ABC contract), so the store is
        # not mutated mid-iteration — avoids an O(store) allocation on the end-of-stream flush.
        for scope, value in self._store.items():
            if scope.operator_id == operator_id and scope.name == name:
                yield scope.key, scope.namespace, value

    def sizes(self) -> dict[tuple[str, str], tuple[int, int]]:
        return {
            name: (count, len(self._key_count.get(name, {})))
            for name, count in self._entry_count.items()
        }

    def _track_add(self, scope: StateScope) -> None:
        name = (scope.operator_id, scope.name)
        self._entry_count[name] = self._entry_count.get(name, 0) + 1
        keys = self._key_count.setdefault(name, {})
        keys[scope.key] = keys.get(scope.key, 0) + 1

    def _track_remove(self, scope: StateScope) -> None:
        name = (scope.operator_id, scope.name)
        self._entry_count[name] = self._entry_count.get(name, 1) - 1
        keys = self._key_count.get(name, {})
        remaining = keys.get(scope.key, 1) - 1
        if remaining <= 0:
            keys.pop(scope.key, None)
        else:
            keys[scope.key] = remaining

    def snapshot(self) -> bytes:
        return pickle.dumps(self._store, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, blob: bytes) -> None:
        self._store = pickle.loads(blob)
        self._entry_count = {}
        self._key_count = {}
        for scope in self._store:  # rebuild the size counters from the restored store
            self._track_add(scope)


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


class ValueState(_Handle, Generic[T]):
    """A single value per key/namespace."""

    def value(self) -> T | None:
        return cast("T | None", self._backend.get(self._scope()))

    def update(self, value: T) -> None:
        self._backend.put(self._scope(), value)

    def clear(self) -> None:
        self._backend.clear(self._scope())


class ReducingState(_Handle, Generic[T]):
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
