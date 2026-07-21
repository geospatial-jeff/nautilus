"""Partitioners: pure, local routing of a data batch to downstream instances.

A partitioner decides, on the *sender*, which downstream instance(s) each row goes to. It is a pure
function of the batch and the downstream fan-out — no central entity is consulted. Routing each batch
is therefore a local decision, which is what "no central scheduler on the data path" means here.

The three cover the routing choices the compiler makes (``compile.lower._spec_for``; ``DESIGN.md`` has
the decision). :class:`Forward` co-locates — sender ``i`` to downstream ``i`` — so a same-width keyless
hop moves no data; it is what an edge uses unless the work forces a redistribution. :class:`RoundRobin`
(Stage 1.5) rebalances a keyless batch across the downstream when the widths differ (the source's fan-out
to every instance). The keyed shuffle (Stage 1.5, generalized in Stage 2 to :class:`KeyGroupPartitioner`)
groups each key onto one instance through a ``group → instance`` indirection table that is the rescale
seam (its class docstring has the mechanism). :class:`HashPartitioner` hashes each key straight to an
instance (modulo the parallelism), the form it generalizes; it is kept only as the equivalence oracle for
when the key-group count equals the parallelism, and is not wired into the runtime (the compiler emits a
:class:`~nautilus.compile.plan.KeyGroupSpec`, never a hash spec).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import msgpack
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

#: Key scalars the keyed shuffle accepts — exactly what the keyed operators form via
#: ``KeyContext((value,))`` from ``to_pylist`` / ``value_counts``. ``bool`` is included implicitly
#: (``bool`` ⊆ ``int``) and kept distinct from ``int`` by msgpack's type tags; ``None`` (a null key
#: cell) is included because it co-partitions cleanly — msgpack packs nil canonically and ``None ==
#: None`` under the state backend's dict equality, with no ``-0.0``/``NaN`` ambiguity. ``float`` (and
#: Arrow ``timestamp`` / ``decimal``, which surface from ``to_pylist`` as ``datetime`` / ``Decimal``)
#: are rejected: the backend keys state by Python dict equality (where ``-0.0 == 0.0`` collapse and
#: ``NaN != NaN``), which disagrees with the shuffle's msgpack bytes, so a key counted once at parallelism
#: 1 could split across instances at a higher parallelism.
_ALLOWED_KEY_SCALARS = (str, bytes, int, type(None))

#: Cap on the per-partitioner key→instance cache (:meth:`_KeyedPartitioner._distinct_bucketer`). It
#: bounds the ``pc.index_in`` lookup set that resolves a batch's keys, so a near-unique-key stream cannot
#: turn routing into an O(cardinality × batches) set rebuild. Past the cap a genuinely-new key still routes
#: correctly — it is bucketed inline and scattered in — it just is not memoized, degrading gracefully to
#: the pre-cache per-key cost instead of blowing up.
_MAX_KEY_CACHE = 1 << 16


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
    distinct_bucketer: Callable[[pa.Array], pa.Array] | None = None,
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

    For the single-key-column case a ``distinct_bucketer`` (:meth:`_KeyedPartitioner._distinct_bucketer`)
    maps the batch's distinct values to buckets through a process-lifetime cache, so ``bucket_of`` runs
    once per distinct key *ever seen*, not once per distinct key *per batch* — the routing hot-path win.
    Without one (the direct-function call path used in tests) it falls back to the per-batch Python loop.
    """
    if len(key_columns) == 1:
        # The common case (one key column): a single dictionary_encode yields the distinct values and a
        # per-row index into them; bucket each distinct value once, then take to expand back to per-row.
        enc = pc.dictionary_encode(batch.column(key_columns[0]), null_encoding="encode")
        bucket_by_distinct = (
            distinct_bucketer(enc.dictionary)
            if distinct_bucketer is not None
            else pa.array([bucket_of((v,)) for v in enc.dictionary.to_pylist()], pa.int32())
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
    distinct_bucketer: Callable[[pa.Array], pa.Array] | None = None,
) -> list[tuple[int, pa.RecordBatch]]:
    """The shared core of the keyed partitioners: compute each row's owning instance, then group the batch
    by instance in a single ``take`` and hand out each instance's rows as a zero-copy slice.

    The rows for one instance are gathered by ``np.flatnonzero``, which preserves their input order, so a
    sub-batch holds its rows in input order — matching the per-key co-location the keyed operators
    downstream rely on, and conserving every row exactly once across the instances. This replaces a
    ``filter`` per instance (``num_downstream`` full-batch rescans) with one reorder, so the cost stops
    growing with the downstream width.
    """
    if batch.num_rows == 0:
        return []
    bucket_per_row = _bucket_per_row(batch, key_columns, bucket_of, distinct_bucketer).to_numpy(
        zero_copy_only=False
    )
    per_instance = [np.flatnonzero(bucket_per_row == i) for i in range(num_downstream)]
    ordered = batch.take(pa.array(np.concatenate(per_instance)))
    out: list[tuple[int, pa.RecordBatch]] = []
    start = 0
    for i, rows in enumerate(per_instance):
        count = len(rows)
        # skip an instance that owns no rows in this batch (as the per-instance filter did)
        if count:
            out.append((i, ordered.slice(start, count)))
            start += count
    return out


def _validate_key(key: tuple[object, ...]) -> None:
    for scalar in key:
        if not isinstance(scalar, _ALLOWED_KEY_SCALARS):
            raise TypeError(
                f"cannot route on key scalar {scalar!r} of type {type(scalar).__name__}; "
                "allowed key scalars are str/int/bool/bytes/null"
            )


def _validate_batch_keys(batch: pa.RecordBatch, key_columns: tuple[str, ...]) -> None:
    """Validate every *distinct* key scalar in a batch — the fail-fast check the per-row route runs inside
    ``bucket_of``, pulled out so the single-owner (``num_downstream == 1``) path validates too. A bad key
    type (e.g. a float, which co-partitions inconsistently) must error at parallelism 1, not silently work
    there and only fail once the job is scaled to more instances."""
    for column in key_columns:
        encoded = pc.dictionary_encode(batch.column(column), null_encoding="encode")
        for value in encoded.dictionary.to_pylist():
            _validate_key((value,))


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
    is bounded by the key cardinality — the same order as the keyed state it routes to.

    Single-column routing adds a *vectorized* layer on top of that Python cache: ``_distinct_bucketer``
    keeps ``(seen key, its bucket)`` as two parallel Arrow arrays so a batch's recurring keys resolve
    through ``pc.index_in`` with no Python at all — turning the per-batch loop over distinct keys into a
    steady-state Arrow lookup. It falls back to ``_bucket`` only for a key not yet cached."""

    _key_columns: tuple[str, ...]

    def __init__(self) -> None:
        self._cache: dict[tuple[object, ...], int] = {}
        self._bucket_of_fn: Callable[[tuple[object, ...]], int] | None = None
        # The single-column vectorized layer (see class docstring): distinct key value -> its bucket, as
        # two parallel Arrow arrays, capped at _MAX_KEY_CACHE so index_in's per-batch set build is bounded.
        self._seen_keys: pa.Array | None = None
        self._seen_buckets: pa.Array | None = None
        self._distinct_bucketer_fn: Callable[[pa.Array], pa.Array] | None = None
        # Turned off for good once the cache caps and keys still keep missing it: a near-unique-key stream
        # gets no cache hits, so index_in is pure overhead and the plain per-batch loop is cheaper.
        self._vectorize = True

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

    def _distinct_bucketer(self, num_downstream: int) -> Callable[[pa.Array], pa.Array]:
        """Map a batch's *distinct* key values (the ``dictionary`` of a ``dictionary_encode``) to their
        owning-instance ``int32`` buckets, reusing a process-lifetime cache so ``_bucket`` runs once per
        distinct key *ever seen*, not once per batch it appears in — the single-column routing win.

        Recurring keys resolve with ``pc.index_in`` + ``pc.take`` (no Python). A key not in the cache is
        bucketed inline (``bucket_of``) and scattered back with ``pc.replace_with_mask``; it is appended to
        the cache only while under ``_MAX_KEY_CACHE``. Once the cache is full and keys still keep missing it
        the cardinality has outgrown the cache, so ``index_in`` earns nothing — the bucketer then flips
        ``_vectorize`` off and reverts to the plain per-batch loop, so a near-unique-key stream is never
        made *slower* than before. A null key never matches ``index_in`` (Arrow does not match
        null-to-null), so it is bucketed directly — the same ``bucket_of((None,))`` the per-batch loop
        computed, kept co-located like any other key.
        """
        if self._distinct_bucketer_fn is None:
            bucket_of = self._bucket_of(num_downstream)

            def bucketer(distinct: pa.Array) -> pa.Array:
                if not self._vectorize:  # cardinality outgrew the cache — the plain loop is cheaper
                    return pa.array([bucket_of((v,)) for v in distinct.to_pylist()], pa.int32())
                hits = (
                    None
                    if self._seen_keys is None
                    else pc.index_in(distinct, value_set=self._seen_keys)
                )
                non_null = pc.is_valid(distinct)
                # A distinct value needs a fresh Python bucket only if it is a non-null key not yet cached.
                missing = non_null if hits is None else pc.and_(non_null, pc.is_null(hits))
                buckets = (
                    pa.nulls(len(distinct), pa.int32())
                    if hits is None
                    else pc.take(self._seen_buckets, hits)
                )
                if pc.any(missing).as_py():
                    fresh = distinct.filter(missing)
                    fresh_buckets = pa.array(
                        [bucket_of((v,)) for v in fresh.to_pylist()], pa.int32()
                    )
                    buckets = pc.replace_with_mask(buckets, missing, fresh_buckets)
                    if self._seen_keys is None:
                        self._seen_keys, self._seen_buckets = fresh, fresh_buckets
                    elif len(self._seen_keys) < _MAX_KEY_CACHE:  # bounded growth
                        self._seen_keys = pa.concat_arrays([self._seen_keys, fresh])
                        self._seen_buckets = pa.concat_arrays([self._seen_buckets, fresh_buckets])
                    else:
                        # Cache is full yet keys still miss it: cardinality exceeds the cap, so index_in
                        # stops paying off. Fall back to the plain per-batch loop from the next batch on.
                        self._vectorize = False
                if pc.any(pc.is_null(distinct)).as_py():
                    buckets = pc.if_else(
                        pc.is_null(distinct), pa.scalar(bucket_of((None,)), pa.int32()), buckets
                    )
                return buckets

            self._distinct_bucketer_fn = bucketer
        return self._distinct_bucketer_fn


class Forward(Partitioner):
    """Co-located 1:1 forwarding: sender ``i`` hands its whole batch straight to downstream instance
    ``i``, moving nothing off its origin instance — the data-locality edge. A single downstream owner
    (a fan-in, or the trivial one-to-one edge) collapses that to instance 0, since every sender then
    has the same one destination.

    The compiler picks this for a keyless edge whose two stages are the same width (so ``i`` is always a
    valid downstream index) and for any edge into a single instance — never when a width change would
    leave ``sender_index`` past the downstream range. With same-index placement (subtask ``i`` of every
    operator on worker ``i`` modulo the worker count) a forwarded edge is a free in-process channel even
    across workers,
    where :class:`RoundRobin` and the keyed shuffles cross the network. Constructed per
    :class:`~nautilus.runtime.actor.Output` with that output's own sender index.
    """

    def __init__(self, sender_index: int = 0) -> None:
        self._sender_index = sender_index

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        # One owner: every sender routes to instance 0. Otherwise the widths match by construction, so
        # sender i co-locates onto downstream i (sender_index is < num_downstream — the compiler only
        # emits a forward edge for equal widths or a single owner).
        return [(0 if num_downstream == 1 else self._sender_index, batch)]


class HashPartitioner(_KeyedPartitioner):
    """The direct keyed shuffle: route each row to the instance that owns its key (``stable_bucket`` of
    the key, modulo the parallelism), so every row with a given key lands on the same instance
    (co-location) and that instance owns the whole range of keys that hash to its own index.
    :class:`KeyGroupPartitioner` generalizes it with a ``group → instance`` indirection table and is what
    the runtime actually builds; this class is kept only as the equivalence oracle for when the key-group
    count equals the parallelism (no spec maps to it).
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
            _validate_batch_keys(
                batch, self._key_columns
            )  # fail fast on a bad key type even at parallelism 1
            return [(0, batch)]  # one owner: every row to instance 0, no bucketing needed
        distinct_bucketer = (
            self._distinct_bucketer(num_downstream) if len(self._key_columns) == 1 else None
        )
        return _route_keyed(
            batch,
            num_downstream,
            self._key_columns,
            self._bucket_of(num_downstream),
            distinct_bucketer,
        )


class KeyGroupPartitioner(_KeyedPartitioner):
    """The keyed shuffle with group indirection: hash each key to one of the key groups
    (``stable_bucket(key, num_groups)``), then route by a static ``group → instance`` table. The table is
    fixed for the run — this never migrates live state (a rescale is a new job; see ``DESIGN.md``) — and
    when the key-group count equals the parallelism, with the identity table, it routes byte-for-byte like
    :class:`HashPartitioner`.

    ``group_table``'s length is the key-group count, and it maps each group to an instance index; every
    value must be a valid instance (at least 0 and below the parallelism). The compiler guarantees this
    when it builds the table from the chosen key-group count and the operator's parallelism, so ``route``
    does not re-check it.
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
            _validate_batch_keys(
                batch, self._key_columns
            )  # fail fast on a bad key type even at parallelism 1
            return [(0, batch)]  # one owner: every row to instance 0, no bucketing needed
        distinct_bucketer = (
            self._distinct_bucketer(num_downstream) if len(self._key_columns) == 1 else None
        )
        return _route_keyed(
            batch,
            num_downstream,
            self._key_columns,
            self._bucket_of(num_downstream),
            distinct_bucketer,
        )


class RoundRobin(Partitioner):
    """Rotates whole batches across downstream instances (keyless rebalancing). It carries a
    rotation cursor, so each :class:`~nautilus.runtime.actor.Output` builds its own instance and the
    cursor is never shared across senders.
    """

    def __init__(self) -> None:
        self._next = 0

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        idx = self._next
        self._next = (self._next + 1) % num_downstream
        return [(idx, batch)]
