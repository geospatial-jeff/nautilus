"""Nautilus operators for the geospatial comparison — the streaming forms of the SQL the
xarray-sql suite proves are relational.

The xarray-sql benchmarks show each array operation is really a ``GROUP BY`` or a ``JOIN``; these
operators express the same relations in nautilus's Arrow dataflow. :class:`KeyedMean` is the
``AVG(...) GROUP BY key`` those cases lean on (the built-in ``count_by`` only sums occurrence counts,
so a mean needs its own running ``(sum, count)`` fold). :func:`tag_regions` is the raster×vector range
``JOIN`` of case 06: with only a handful of regions the join's build side is tiny, so the natural
streaming form is to broadcast the boxes and tag each pixel in a ``map`` — a broadcast range join —
rather than a hash join, which nautilus builds only for equi-keys.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nautilus.core.operator import (
    AsyncOneInputOperator,
    Collector,
    OneInputOperator,
    OperatorContext,
    SourceOperator,
)
from nautilus.core.records import EOS_FRAME, Batch, Frame


class SlicedSource(SourceOperator):
    """Stream a large unraveled table one outer-slice at a time, building each slice's columns on demand.

    A row engine consuming gridded data must view the grid as rows; doing that for a whole day of ERA5
    (25M cells) at once would materialize hundreds of MB before a single row is processed. ``slice_fn(i)``
    instead returns slice ``i``'s columns (``{name: 1-D np.ndarray}``) only when the source reaches it, so
    the pipeline holds one slice at a time — the streaming a nautilus source is meant to do, and the
    mirror of how xarray-sql reads its Zarr in ``chunks``. Each slice is emitted in ``rows_per_batch``-row
    batches. The per-slice unravel runs inside the pipeline (so it is counted in a timed run, like the
    chunk→Arrow conversion xarray-sql does inside ``.sql()``)."""

    def __init__(self, n_slices: int, slice_fn: Callable[[int], dict[str, np.ndarray]], rows_per_batch: int) -> None:
        self._n_slices = n_slices
        self._slice_fn = slice_fn
        self._rpb = rows_per_batch

    async def frames(self) -> AsyncIterator[Frame]:
        for i in range(self._n_slices):
            cols = self._slice_fn(i)
            names = list(cols)
            rb = pa.RecordBatch.from_arrays([pa.array(cols[k]) for k in names], names=names)
            for j in range(0, rb.num_rows, self._rpb):
                yield Batch(rb.slice(j, self._rpb))
        yield EOS_FRAME


class ZarrChunkSource(SourceOperator):
    """Read a Zarr array's chunks directly with zarr-python's async API, unravel each to Arrow rows, and
    stream them — the source that lets a nautilus pipeline ingest cloud Zarr itself, with no xarray in the
    read path. It exists to exercise nautilus in its design regime: reads run on the actor's event loop
    with up to ``prefetch`` chunk fetches in flight, so the next chunk is being fetched while downstream
    operators compute the current one, and the whole read overlaps the aggregation instead of preceding
    it. ``time_indices`` are the chunk positions along axis 0 to read (each a ``(rows, cols)`` slice here);
    ``slice_columns(pos, data)`` turns one fetched slice into ``{name: 1-D np.ndarray}`` columns."""

    def __init__(
        self,
        url: str,
        var: str,
        time_indices,
        slice_columns: Callable[[int, np.ndarray], dict[str, np.ndarray]],
        *,
        anonymous: bool = True,
        prefetch: int = 8,
        rows_per_batch: int = 262_144,
    ) -> None:
        if not url.startswith("gs://"):
            raise ValueError(f"ZarrChunkSource expects a gs:// URL, got {url!r}")
        self._bucket, self._group_path = url[len("gs://") :].split("/", 1)
        self._var = var
        self._time_indices = list(time_indices)
        self._slice_columns = slice_columns
        self._anonymous = anonymous
        self._prefetch = max(1, prefetch)
        self._rpb = rows_per_batch

    async def frames(self) -> AsyncIterator[Frame]:
        import asyncio

        import zarr.api.asynchronous as za
        from obstore.store import GCSStore
        from zarr.storage import ObjectStore

        # obstore (Rust object-store client) as the Zarr backend: its async has no aiohttp session bound
        # to nautilus's per-run event loop, so it cleans up across the many runs a benchmark drives —
        # unlike an fsspec/gcsfs store, which leaks a closed-loop session between runs.
        store = GCSStore(self._bucket, skip_signature=self._anonymous)
        group = await za.open_group(store=ObjectStore(store, read_only=True), path=self._group_path, mode="r")
        arr = await group.getitem(self._var)
        idxs = self._time_indices
        n = len(idxs)

        def issue(pos: int):
            return asyncio.ensure_future(arr.getitem((idxs[pos], slice(None), slice(None))))

        # Keep `prefetch` reads in flight; yield each in order and refill as it drains, so fetches overlap
        # each other and the downstream compute (bounded — the actor's output channel backpressures too).
        inflight = {pos: issue(pos) for pos in range(min(self._prefetch, n))}
        next_issue = len(inflight)
        for pos in range(n):
            data = np.asarray(await inflight.pop(pos))
            if next_issue < n:
                inflight[next_issue] = issue(next_issue)
                next_issue += 1
            cols = self._slice_columns(idxs[pos], data)  # global chunk index, so sharding is transparent
            names = list(cols)
            rb = pa.RecordBatch.from_arrays([pa.array(cols[k]) for k in names], names=names)
            for j in range(0, rb.num_rows, self._rpb):
                yield Batch(rb.slice(j, self._rpb))
        yield EOS_FRAME


class ZarrReadChunk(AsyncOneInputOperator):
    """Read the Zarr chunks named in each input batch and emit their unraveled rows — the distributed-I/O
    reader. The IR pins a source to one instance, so read *fan-out across workers* is expressed instead as
    a tiny source of chunk indices feeding this async stage at parallelism N: each of the N instances
    (placed on N workers by ``run(workers=N)``) opens its own object-store client and fetches a different
    subset of chunks, so the read genuinely parallelises across processes. Stateless — it runs unordered,
    and ``slice_columns(chunk_index, data)`` turns each fetched slice into ``{name: 1-D np.ndarray}``.

    The store/array is opened once per instance, lazily on the first ``fetch`` (opening needs ``await``,
    which the synchronous ``open`` cannot do), guarded so concurrent first fetches open it only once."""

    def __init__(
        self,
        url: str,
        var: str,
        slice_columns: Callable[[int, np.ndarray], dict[str, np.ndarray]],
        *,
        anonymous: bool = True,
        max_in_flight: int = 8,
    ) -> None:
        if not url.startswith("gs://"):
            raise ValueError(f"ZarrReadChunk expects a gs:// URL, got {url!r}")
        self._bucket, self._group_path = url[len("gs://") :].split("/", 1)
        self._var = var
        self._slice_columns = slice_columns
        self._anonymous = anonymous
        self._cap = max_in_flight

    def open(self, ctx: OperatorContext) -> None:
        self._arr = None
        self._lock = None

    async def _array(self):
        import asyncio

        if self._arr is not None:
            return self._arr
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._arr is None:
                import zarr.api.asynchronous as za
                from obstore.store import GCSStore
                from zarr.storage import ObjectStore

                store = GCSStore(self._bucket, skip_signature=self._anonymous)
                group = await za.open_group(
                    store=ObjectStore(store, read_only=True), path=self._group_path, mode="r"
                )
                self._arr = await group.getitem(self._var)
        return self._arr

    async def fetch(self, batch: pa.RecordBatch) -> object:
        import asyncio

        arr = await self._array()
        idxs = [int(i) for i in batch.column("chunk_idx").to_pylist()]
        chunks = await asyncio.gather(
            *(arr.getitem((i, slice(None), slice(None))) for i in idxs)
        )
        return [(i, np.asarray(c)) for i, c in zip(idxs, chunks, strict=True)]

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        for idx, data in result:  # type: ignore[attr-defined]
            cols = self._slice_columns(idx, data)
            names = list(cols)
            out.emit(pa.RecordBatch.from_arrays([pa.array(cols[k]) for k in names], names=names))

    def max_in_flight(self) -> int:
        return self._cap

    def ordered(self) -> bool:
        return False  # stateless — emission order does not matter, so reads may complete in any order

    def key_columns(self) -> tuple[str, ...] | None:
        return None


def _add_sum_count(a: tuple[float, int], b: tuple[float, int]) -> tuple[float, int]:
    """Merge two per-key ``(sum, count)`` partials. Module-level so the operator cloudpickles to a
    worker at ``parallelism > 1``."""
    return (a[0] + b[0], a[1] + b[1])


class KeyedMean(OneInputOperator):
    """``AVG(value) ... GROUP BY key`` — the mean of a value column per key, emitted at end of stream.

    Modeled on the built-in ``KeyedCount`` — including its integer fast path: for non-negative integer
    keys it accumulates a running per-key ``sum`` and ``count`` in numpy arrays (``np.bincount`` per
    batch), folding a whole batch with no per-key Python object, then divides at end of stream. Other key
    types fall back to an Arrow group-by per batch folded into keyed state. Either way the sum accumulates
    in float64 even for a float32 value column, and ``count``/sum skip *nulls*, like SQL ``AVG``. A *NaN*
    cell, though,
    though, propagates: this operator and DataFusion ``AVG`` both yield NaN for a group containing one,
    whereas xarray ``mean(skipna=True)`` skips it — so the three agree only on gap-free data (the ERA5
    field here has no NaN). Emits one row per key: ``key``, ``mean``, and ``n`` (the contributing count).
    """

    _STATE = "sum_count"

    def __init__(self, key_col: str, value_col: str, mean_col: str = "mean") -> None:
        self.key_col = key_col
        self.value_col = value_col
        self.mean_col = mean_col

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx
        self._key_type: pa.DataType | None = None  # captured from input so output keeps the key's type
        self._sum: np.ndarray | None = None  # running per-key sum, indexed by key (integer fast path)
        self._cnt: np.ndarray | None = None  # running per-key non-null count, same index

    def key_columns(self) -> tuple[str, ...]:
        return (self.key_col,)

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        kcol = batch.column(self.key_col)
        if self._key_type is None:
            self._key_type = kcol.type
            if pa.types.is_integer(kcol.type):  # non-negative integer keys → vectorized numpy accumulator
                self._sum = np.zeros(0, dtype=np.float64)
                self._cnt = np.zeros(0, dtype=np.int64)
        if self._sum is not None:
            vcol = batch.column(self.value_col)
            if kcol.null_count or vcol.null_count:  # drop rows with a null key or value (skipna, like AVG)
                valid = pc.and_(kcol.is_valid(), vcol.is_valid())
                kcol, vcol = kcol.filter(valid), vcol.filter(valid)
            keys = np.asarray(kcol.to_numpy(zero_copy_only=False))
            if keys.size:
                vals = np.asarray(vcol.to_numpy(zero_copy_only=False), dtype=np.float64)
                bs = np.bincount(keys, weights=vals)  # per-key sum; non-negative ints only
                bc = np.bincount(keys)  # per-key count
                if bs.size > self._sum.size:
                    self._sum = np.concatenate([self._sum, np.zeros(bs.size - self._sum.size)])
                    self._cnt = np.concatenate([self._cnt, np.zeros(bs.size - self._cnt.size, np.int64)])
                self._sum[: bs.size] += bs
                self._cnt[: bc.size] += bc
            return
        tbl = pa.table({"k": kcol, "v": batch.column(self.value_col)})
        agg = tbl.group_by("k").aggregate([("v", "sum"), ("v", "count")])
        items = zip(
            ((k,) for k in agg.column("k").to_pylist()),
            zip(agg.column("v_sum").to_pylist(), agg.column("v_count").to_pylist(), strict=True),
            strict=True,
        )
        self._ctx.reduce_all(self._STATE, items, _add_sum_count)

    def on_eos(self, out: Collector) -> None:
        if self._sum is not None:  # integer fast path: mean = sum/count over keys that received rows
            nz = np.nonzero(self._cnt)[0]
            if nz.size:
                out.emit(
                    pa.RecordBatch.from_arrays(
                        [
                            pa.array(nz, self._key_type),
                            pa.array(self._sum[nz] / self._cnt[nz], pa.float64()),
                            pa.array(self._cnt[nz], pa.int64()),
                        ],
                        names=[self.key_col, self.mean_col, "n"],
                    )
                )
            return
        keys: list[object] = []
        means: list[float] = []
        counts: list[int] = []
        fired = []
        for kctx, (total, n) in self._ctx.entries(self._STATE):
            keys.append(kctx.key[0])
            means.append(total / n if n else float("nan"))
            counts.append(n)
            fired.append(kctx)
        for kctx in fired:
            self._ctx.clear_state(self._STATE, kctx)
        if keys:
            out.emit(
                pa.RecordBatch.from_arrays(
                    [
                        pa.array(keys, self._key_type),
                        pa.array(means, pa.float64()),
                        pa.array(counts, pa.int64()),
                    ],
                    names=[self.key_col, self.mean_col, "n"],
                )
            )


def make_region_tagger(regions: list[tuple[str, float, float, float, float]], value_col: str):
    """Build the ``map`` for case 06's raster×vector range join: tag each pixel with the region it
    falls in. ``regions`` is ``(name, lat_min, lat_max, lon_min, lon_max)`` rows; the returned function
    takes a batch with ``latitude``/``longitude``/``value_col`` and emits ``(region_id, value)`` for
    every pixel-in-box pair (a pixel in no box drops out; the boxes here are disjoint, so at most one
    match per pixel). This is the broadcast form of ``JOIN regions ON latitude BETWEEN … AND longitude
    BETWEEN …`` — the build side is five boxes, so broadcasting them beats a shuffle-and-hash join."""
    bounds = [(i, *r[1:]) for i, r in enumerate(regions)]

    def tag(batch: pa.RecordBatch) -> pa.RecordBatch:
        lat = batch.column("latitude").to_numpy(zero_copy_only=False)
        lon = batch.column("longitude").to_numpy(zero_copy_only=False)
        val = batch.column(value_col).to_numpy(zero_copy_only=False)
        ids: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for rid, lat_min, lat_max, lon_min, lon_max in bounds:
            m = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
            if m.any():
                ids.append(np.full(int(m.sum()), rid, dtype=np.int32))
                vals.append(val[m])
        if not ids:
            return pa.RecordBatch.from_arrays(
                [pa.array([], pa.int32()), pa.array([], pa.float64())],
                names=["region_id", value_col],
            )
        return pa.RecordBatch.from_arrays(
            [pa.array(np.concatenate(ids)), pa.array(np.concatenate(vals))],
            names=["region_id", value_col],
        )

    return tag
