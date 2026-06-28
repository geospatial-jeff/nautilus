"""Partitioners: pure, local routing of a data batch to downstream instances.

A partitioner decides, on the *sender*, which downstream instance(s) each row goes to. It is a pure
function of the batch and the downstream fan-out — no central entity is consulted. Routing each batch
is therefore a local decision, which is what "no central scheduler on the data path" means here.

Stage 0 ships :class:`Forward` (1:1). Stage 1.5 added :class:`RoundRobin` rebalancing and the keyed
shuffle, generalized in Stage 2 to :class:`KeyGroupPartitioner` — the keyed shuffle with a
``group → instance`` indirection table that is the rescale seam (its class docstring has the mechanism;
``DESIGN.md`` has the decision). :class:`HashPartitioner` is the direct ``hash(key) mod Q`` form it
generalizes; it is kept only as the ``G == Q`` equivalence oracle for tests and is not wired into the
runtime (the compiler emits a :class:`~nautilus.compile.plan.KeyGroupSpec`, never a hash spec).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import msgpack
import pyarrow as pa
import pyarrow.compute as pc

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


def _bucket_per_row(
    batch: pa.RecordBatch,
    key_columns: tuple[str, ...],
    bucket_of: Callable[[tuple[object, ...]], int],
) -> pa.Array:
    """The owning instance for every row, as an ``int32`` column, computed once per *distinct* key
    rather than once per row.

    ``bucket_of`` still runs in Python — it hashes the key with ``stable_bucket`` and validates it — but
    Arrow finds the distinct keys and maps each row back to its key for us, so that Python call is made
    once per distinct key, not once per row. The distinct keys are read back with ``to_pylist`` (Python
    scalars — never ``.to_numpy()``, whose numpy scalars ``msgpack`` cannot pack), so the shuffle hashes
    exactly the scalars the keyed operators key on. ``null_encoding="encode"`` gives a null cell its own
    dictionary slot (the default masks it, which ``take``/``filter`` would then drop) — a null key is a
    real key the operators count, so it must route and co-locate like any other.
    """
    if len(key_columns) == 1:
        # The common case (one key column): a single dictionary_encode yields the distinct values and a
        # per-row index into them; bucket each distinct value once, then take to expand back to per-row.
        enc = pc.dictionary_encode(batch.column(key_columns[0]), null_encoding="encode")
        bucket_by_distinct = pa.array(
            [bucket_of((v,)) for v in enc.dictionary.to_pylist()], pa.int32()
        )
        return pc.take(bucket_by_distinct, enc.indices)

    # Several key columns: fold each column's per-row dictionary index into one compact combo id. The
    # mixed-radix fold (combo*card + index) uniquely identifies the index tuple; re-encoding the running
    # combo after each column keeps it within [0, distinct-so-far) ≤ num_rows, so it can never overflow
    # int64 no matter how wide or high-cardinality the key is. We never decode the combo back to a tuple
    # — that would lose the per-column values — so each distinct combo's key is read from a representative
    # row (the first one in that group) and bucketed once.
    encs = [pc.dictionary_encode(batch.column(c), null_encoding="encode") for c in key_columns]
    combo = encs[0].indices
    num_distinct = len(encs[0].dictionary)
    for enc in encs[1:]:
        folded = pc.add(pc.multiply(combo.cast(pa.int64()), len(enc.dictionary)), enc.indices)
        recompressed = pc.dictionary_encode(folded)
        combo = recompressed.indices
        num_distinct = len(recompressed.dictionary)
    rids = pa.array(range(batch.num_rows), pa.int64())
    grouped = pa.table({"combo": combo, "rid": rids}).group_by("combo").aggregate([("rid", "min")])
    # group_by returns a Table, whose columns are chunked; combine to one Array for RecordBatch.take.
    reps = batch.take(grouped.column("rid_min").combine_chunks())  # one row per distinct combo
    rep_cols = [reps.column(c).to_pylist() for c in key_columns]
    bucket_by_combo = [0] * num_distinct
    for m, combo_id in enumerate(grouped.column("combo").to_pylist()):
        bucket_by_combo[combo_id] = bucket_of(tuple(col[m] for col in rep_cols))
    return pc.take(pa.array(bucket_by_combo, pa.int32()), combo)


def _route_keyed(
    batch: pa.RecordBatch,
    num_downstream: int,
    key_columns: tuple[str, ...],
    bucket_of: Callable[[tuple[object, ...]], int],
) -> list[tuple[int, pa.RecordBatch]]:
    """The shared core of the keyed partitioners: compute each row's owning instance, then ``filter`` the
    batch once per instance into its sub-batch.

    ``filter`` keeps the surviving rows in their original order, so a sub-batch holds its rows in input
    order — matching the per-key co-location the keyed operators downstream rely on, and conserving every
    row exactly once across the instances.
    """
    if batch.num_rows == 0:
        return []
    bucket_per_row = _bucket_per_row(batch, key_columns, bucket_of)
    out: list[tuple[int, pa.RecordBatch]] = []
    for i in range(num_downstream):
        sub = batch.filter(pc.equal(bucket_per_row, pa.scalar(i, pa.int32())))
        if sub.num_rows:
            out.append((i, sub))
    return out


def _validate_key(key: tuple[object, ...]) -> None:
    for scalar in key:
        if not isinstance(scalar, _ALLOWED_KEY_SCALARS):
            raise TypeError(
                f"cannot route on key scalar {scalar!r} of type {type(scalar).__name__}; "
                "allowed key scalars are str/int/bool/bytes/null"
            )


class Partitioner(ABC):
    """Splits a batch into ``(downstream_index, sub_batch)`` pairs."""

    @abstractmethod
    def route(
        self, batch: pa.RecordBatch, num_downstream: int
    ) -> list[tuple[int, pa.RecordBatch]]: ...


class _KeyedPartitioner(Partitioner):
    """Shared machinery for the keyed shuffles: a per-instance cache from key tuple to owning downstream
    index. ``_bucket(key)`` is the subclass's pure mapping; the cache means it (and the per-key
    validation) runs once per distinct key for the life of the partitioner, not once per row. The cache
    is bounded by the key cardinality — the same order as the keyed state it routes to."""

    _key_columns: tuple[str, ...]

    def __init__(self) -> None:
        self._cache: dict[tuple[object, ...], int] = {}
        self._bucket_of_fn: Callable[[tuple[object, ...]], int] | None = None

    def _bucket(self, key: tuple[object, ...], num_downstream: int) -> int:
        raise NotImplementedError

    def _bucket_of(self, num_downstream: int) -> Callable[[tuple[object, ...]], int]:
        # Built once and reused: num_downstream is constant for an edge's life, so the closure (and the
        # per-key cache it closes over) is the same every batch — no need to re-create it per route().
        if self._bucket_of_fn is None:
            cache = self._cache

            def bucket_of(key: tuple[object, ...]) -> int:
                idx = cache.get(key)
                if (
                    idx is None
                ):  # a key never seen: validate and compute its owner once, then memoize
                    _validate_key(key)
                    idx = self._bucket(key, num_downstream)
                    cache[key] = idx
                return idx

            self._bucket_of_fn = bucket_of
        return self._bucket_of_fn


class Forward(Partitioner):
    """1:1 forwarding. Requires a single downstream instance (upstream P == downstream Q == 1 in
    Stage 0); the whole batch goes to instance 0."""

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        if num_downstream != 1:
            raise ValueError(f"Forward requires a single downstream instance, got {num_downstream}")
        return [(0, batch)]


class HashPartitioner(_KeyedPartitioner):
    """The direct keyed shuffle: route each row to the instance that owns ``hash(key) mod Q``, so every
    row with a given key lands on the same instance (co-location) and that instance owns the whole key
    range ``{k : stable_bucket(k, Q) == i}``. :class:`KeyGroupPartitioner` generalizes it with a
    ``group → instance`` indirection table and is what the runtime actually builds; this class is kept
    only as the ``G == Q`` equivalence oracle for tests (no spec maps to it).
    """

    def __init__(self, key_columns: Sequence[str]) -> None:
        if not key_columns:
            raise ValueError("HashPartitioner needs at least one key column")
        super().__init__()
        self._key_columns = tuple(key_columns)

    def _bucket(self, key: tuple[object, ...], num_downstream: int) -> int:
        return stable_bucket(key, num_downstream)

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        if num_downstream == 1:
            return [(0, batch)]  # one owner: skip per-row hashing entirely
        return _route_keyed(
            batch, num_downstream, self._key_columns, self._bucket_of(num_downstream)
        )


class KeyGroupPartitioner(_KeyedPartitioner):
    """The keyed shuffle with group indirection: hash each key to one of ``G`` key groups
    (``stable_bucket(key, G)``), then route by a static ``group → instance`` table. The table is fixed
    for the run — this never migrates live state (a rescale is a new job; see ``DESIGN.md``) — and at
    ``G == Q`` with the identity table it routes byte-for-byte like :class:`HashPartitioner`.

    ``group_table`` has length ``G`` and maps each group to an instance index; every value must be a
    valid instance (``0 <= group_table[g] < Q``). The compiler guarantees this when it builds the table
    from the chosen ``G`` and the operator's parallelism ``Q``, so ``route`` does not re-check it.
    """

    def __init__(self, key_columns: Sequence[str], group_table: Sequence[int]) -> None:
        if not key_columns:
            raise ValueError("KeyGroupPartitioner needs at least one key column")
        if not group_table:
            raise ValueError("KeyGroupPartitioner needs a non-empty group table")
        super().__init__()
        self._key_columns = tuple(key_columns)
        self._group_table = tuple(group_table)
        self._num_groups = len(self._group_table)

    def _bucket(self, key: tuple[object, ...], num_downstream: int) -> int:
        # The group's owner comes from the fixed table — independent of num_downstream — so a cached
        # entry stays valid for the run (num_downstream is constant per edge anyway).
        return self._group_table[stable_bucket(key, self._num_groups)]

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        if num_downstream == 1:
            return [(0, batch)]  # one owner: skip per-row hashing entirely
        return _route_keyed(
            batch, num_downstream, self._key_columns, self._bucket_of(num_downstream)
        )


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
