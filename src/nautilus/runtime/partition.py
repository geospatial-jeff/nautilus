"""Partitioners: pure, local routing of a data batch to downstream instances.

A partitioner decides, on the *sender*, which downstream instance(s) each row goes to. It is a pure
function of the batch and the downstream fan-out — no central entity is consulted. Routing each batch
is therefore a local decision, which is what "no central scheduler on the data path" means here.

Stage 0 ships :class:`Forward` (1:1) and :class:`Broadcast`. Stage 1.5 adds the keyed shuffle
(:class:`HashPartitioner`, direct ``hash(key) mod Q``) and :class:`RoundRobin` rebalancing, which
multi-instance parallelism needs. Key-*group* indirection for rescaling (``hash(key) mod G`` then a
group→instance table) is deferred to Stage 2.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

import msgpack
import pyarrow as pa

#: Key scalars the keyed shuffle accepts — exactly what the keyed operators form via
#: ``KeyContext((value,))`` from ``to_pylist`` / ``value_counts``. ``bool`` is included implicitly
#: (``bool`` ⊆ ``int``) and kept distinct from ``int`` by msgpack's type tags; ``None`` (a null key
#: cell) is included because it co-partitions cleanly — msgpack packs nil canonically and ``None ==
#: None`` under the state backend's dict equality, with no ``-0.0``/``NaN`` ambiguity. ``float`` (and
#: Arrow ``timestamp`` / ``decimal``, which surface from ``to_pylist`` as ``datetime`` / ``Decimal``)
#: are rejected: the backend keys state by Python dict equality (where ``-0.0 == 0.0`` collapse and
#: ``NaN != NaN``), which disagrees with the shuffle's msgpack bytes, so a key counted once at P=1
#: could split across instances at P=N.
_ALLOWED_KEY_SCALARS = (str, bytes, int, type(None))


def stable_bucket(key: tuple[object, ...], num_downstream: int) -> int:
    """Map a key tuple to its owning downstream instance with a process-, seed-, and platform-stable
    hash.

    Never Python's builtin :func:`hash`, which salts ``str``/``bytes`` per process via
    ``PYTHONHASHSEED`` and would route the same key to different instances in a parent versus a
    spawned worker, splitting that key's state. ``msgpack`` (``use_bin_type=True``) canonicalizes the
    key to identical, type-tagged, length-prefixed bytes in any process — so ``int 1`` ≠ ``str "1"`` ≠
    ``bool True`` ≠ ``bytes b"1"`` and ``("a", "bc")`` ≠ ``("ab", "c")`` — and ``blake2b`` is stdlib
    and unsalted.
    """
    raw: bytes = msgpack.packb(list(key), use_bin_type=True)
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_downstream


class Partitioner(ABC):
    """Splits a batch into ``(downstream_index, sub_batch)`` pairs."""

    @abstractmethod
    def route(
        self, batch: pa.RecordBatch, num_downstream: int
    ) -> list[tuple[int, pa.RecordBatch]]: ...


class Forward(Partitioner):
    """1:1 forwarding. Requires a single downstream instance (upstream P == downstream Q == 1 in
    Stage 0); the whole batch goes to instance 0."""

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        if num_downstream != 1:
            raise ValueError(f"Forward requires a single downstream instance, got {num_downstream}")
        return [(0, batch)]


class Broadcast(Partitioner):
    """Sends a full copy of every batch to every downstream instance."""

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        return [(i, batch) for i in range(num_downstream)]


class HashPartitioner(Partitioner):
    """The keyed shuffle: route each row to the instance that owns ``hash(key) mod Q``, so every row
    with a given key lands on the same instance (co-location) and that instance owns the whole key
    range ``{k : stable_bucket(k, Q) == i}``.

    The key is extracted per row with ``to_pylist`` (Python scalars — never ``.to_numpy()``, whose
    numpy scalars msgpack cannot pack), matching the keyed operators' own ``to_pylist`` keying so the
    shuffle and the operator agree on every key exactly. Per-row Python hashing is the accepted MVP
    cost; the Arrow-vectorized hot path is Stage 3.
    """

    def __init__(self, key_columns: Sequence[str]) -> None:
        if not key_columns:
            raise ValueError("HashPartitioner needs at least one key column")
        self._key_columns = tuple(key_columns)

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        if num_downstream == 1:
            return [(0, batch)]  # one owner: skip per-row hashing entirely
        cols = [batch.column(c).to_pylist() for c in self._key_columns]
        buckets: list[list[int]] = [[] for _ in range(num_downstream)]
        for r in range(batch.num_rows):
            key = tuple(col[r] for col in cols)
            for scalar in key:
                if not isinstance(scalar, _ALLOWED_KEY_SCALARS):
                    raise TypeError(
                        f"HashPartitioner cannot route on key scalar {scalar!r} of type "
                        f"{type(scalar).__name__}; allowed key scalars are str/int/bool/bytes/null"
                    )
            buckets[stable_bucket(key, num_downstream)].append(r)
        return [
            (i, batch.take(pa.array(rows, pa.int64()))) for i, rows in enumerate(buckets) if rows
        ]


class RoundRobin(Partitioner):
    """Rotates whole batches across downstream instances (keyless N-way rebalancing). It carries a
    rotation cursor, so each :class:`~nautilus.runtime.actor.Output` builds its own instance and the
    cursor is never shared across senders.
    """

    def __init__(self) -> None:
        self._next = 0

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        idx = self._next
        self._next = (self._next + 1) % num_downstream
        return [(idx, batch)]
