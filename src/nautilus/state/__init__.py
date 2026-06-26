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

    def with_namespace(self, namespace: Namespace) -> KeyContext:
        return KeyContext(self.key, namespace)


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
        all open windows when flushing. The iterable is a snapshot; mutating during iteration is
        safe."""

    @abstractmethod
    def snapshot(self) -> bytes: ...

    @abstractmethod
    def restore(self, blob: bytes) -> None: ...


@dataclass
class InMemoryStateBackend(StateBackend):
    """Dict-backed state. MVP only — unbounded memory; documented limitation."""

    _store: dict[StateScope, object] = field(default_factory=dict)

    def get(self, scope: StateScope) -> object | None:
        return self._store.get(scope)

    def put(self, scope: StateScope, value: object) -> None:
        self._store[scope] = value

    def clear(self, scope: StateScope) -> None:
        self._store.pop(scope, None)

    def entries(self, operator_id: str, name: str) -> Iterator[tuple[Key, Namespace, object]]:
        for scope, value in list(self._store.items()):
            if scope.operator_id == operator_id and scope.name == name:
                yield scope.key, scope.namespace, value

    def snapshot(self) -> bytes:
        return pickle.dumps(self._store, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, blob: bytes) -> None:
        self._store = pickle.loads(blob)


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


class ListState(_Handle, Generic[T]):
    """An append-only list per key/namespace."""

    def add(self, value: T) -> None:
        scope = self._scope()
        current = cast("list[T] | None", self._backend.get(scope))
        if current is None:
            self._backend.put(scope, [value])
        else:
            current.append(value)

    def get(self) -> list[T]:
        return cast("list[T]", self._backend.get(self._scope()) or [])

    def clear(self) -> None:
        self._backend.clear(self._scope())


class MapState(_Handle, Generic[T]):
    """A dict per key/namespace."""

    def _dict(self) -> dict[object, T] | None:
        return cast("dict[object, T] | None", self._backend.get(self._scope()))

    def get(self, map_key: object) -> T | None:
        d = self._dict()
        return None if d is None else d.get(map_key)

    def put(self, map_key: object, value: T) -> None:
        scope = self._scope()
        d = self._dict()
        if d is None:
            self._backend.put(scope, {map_key: value})
        else:
            d[map_key] = value

    def items(self) -> list[tuple[object, T]]:
        d = self._dict()
        return [] if d is None else list(d.items())

    def clear(self) -> None:
        self._backend.clear(self._scope())
