"""Real-data sources for the geospatial comparison — the I/O adapters that feed real cloud Zarr into a
nautilus pipeline.

The pipelines themselves reuse the library operators (:class:`nautilus.operators.KeyedMean`, the region
tagger and map functions in :mod:`nautilus.benchmarks`) so nothing here re-implements the nautilus side —
only how a real store becomes a stream, the part the synthetic ``bench-geo-*`` sources cannot cover. Three
read shapes need three sources: an in-memory grid (:class:`SlicedSource`), a direct cloud-Zarr read
(:class:`ZarrSliceSource`, plus :class:`Wb2ForecastSource` for the two-model forecast), and a worker-fanned
read (:class:`ZarrReadChunk`); each class's docstring says how and why.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import numpy as np
import pyarrow as pa

from nautilus.core.operator import (
    AsyncOneInputOperator,
    Collector,
    OperatorContext,
    SourceOperator,
)
from nautilus.core.records import EOS_FRAME, Batch, Frame

Selection = tuple  # a zarr getitem selection, e.g. (t, slice(None), slice(None)) or (yslice, xslice)


def _store_and_prefix(url: str, anonymous: bool):
    """Resolve a store URL to an ``(obstore store, group-path prefix)``. A ``gs://bucket/rest`` URL keys the
    GCS bucket and carries ``rest`` as the prefix of the group path; an ``http(s)://…`` URL is the store
    root itself (no prefix), so the caller passes the in-store group path separately. obstore's Rust client
    (not fsspec/gcsfs) so no aiohttp session binds to nautilus's per-run loop and leaks across benchmark
    runs."""
    if url.startswith("gs://"):
        from obstore.store import GCSStore

        bucket, _, prefix = url[len("gs://") :].partition("/")
        return GCSStore(bucket, skip_signature=anonymous), prefix
    if url.startswith(("http://", "https://")):
        from obstore.store import HTTPStore

        return HTTPStore.from_url(url), ""
    raise ValueError(f"unsupported store URL scheme: {url!r}")


async def _open_arrays(url: str, group_path: str, vars_: list[str], anonymous: bool):
    """Open a Zarr group over GCS or HTTP and return ``{name: async array}`` for each requested var."""
    import zarr.api.asynchronous as za
    from zarr.storage import ObjectStore

    store, prefix = _store_and_prefix(url, anonymous)
    path = "/".join(p for p in (prefix, group_path) if p)
    group = await za.open_group(store=ObjectStore(store, read_only=True), path=path, mode="r")
    return {v: await group.getitem(v) for v in vars_}


class SlicedSource(SourceOperator):
    """Stream a large unraveled table one outer-slice at a time, building each slice's columns on demand.

    A row engine consuming gridded data must view the grid as rows; doing that for a whole day of ERA5
    (25M cells) at once would materialize hundreds of MB before a single row is processed. ``slice_fn(i)``
    instead returns slice ``i``'s columns (``{name: 1-D np.ndarray}``) only when the source reaches it, so
    the pipeline holds one slice at a time — like how xarray-sql reads its Zarr in ``chunks``. Each slice
    is emitted in ``rows_per_batch``-row
    batches. The per-slice unravel runs inside the pipeline (so it is counted in a timed run, like the
    chunk→Arrow conversion xarray-sql does inside ``.sql()``)."""

    def __init__(
        self, n_slices: int, slice_fn: Callable[[int], dict[str, np.ndarray]], rows_per_batch: int
    ) -> None:
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


class ZarrSliceSource(SourceOperator):
    """Read a Zarr store directly with zarr-python's async API, one selection at a time, and stream each
    fetched slice's unraveled rows — the source that lets a nautilus pipeline ingest cloud Zarr itself, with
    no xarray in the read path. Reads run on the actor's event loop with up to ``prefetch`` fetches in
    flight, so the next slice is being fetched while downstream operators compute the current one, and the
    whole read overlaps the aggregation instead of preceding it.

    ``selections`` is the list of zarr getitem selections to read — a time chunk ``(t, slice(None),
    slice(None))``, a ``(time, lead, …)`` forecast slice, or a spatial window ``(yslice, xslice)`` — one per
    fetch, so the same reader covers every case's read shape. All of ``vars`` are read at each selection (two
    reflectance bands for NDVI, one field otherwise) and handed, with the selection, to
    ``slice_columns(selection, {var: np.ndarray})`` which returns ``{name: 1-D np.ndarray}`` columns; the
    selection carries the time/lead indices the callback needs to build a group key. ``group_path`` is the
    Zarr group within the store (empty when the store URL already points at it, as for GCS)."""

    def __init__(
        self,
        url: str,
        vars_: list[str],
        selections: list[Selection],
        slice_columns: Callable[[Selection, dict[str, np.ndarray]], dict[str, np.ndarray]],
        *,
        group_path: str = "",
        anonymous: bool = True,
        prefetch: int = 8,
        rows_per_batch: int = 262_144,
    ) -> None:
        self._url = url
        self._vars = list(vars_)
        self._selections = list(selections)
        self._slice_columns = slice_columns
        self._group_path = group_path
        self._anonymous = anonymous
        self._prefetch = max(1, prefetch)
        self._rpb = rows_per_batch

    async def frames(self) -> AsyncIterator[Frame]:
        import asyncio

        arrs = await _open_arrays(self._url, self._group_path, self._vars, self._anonymous)
        sels = self._selections
        n = len(sels)

        def issue(pos: int):
            sel = sels[pos]
            return asyncio.gather(*(arrs[v].getitem(sel) for v in self._vars))

        # Keep `prefetch` reads in flight; yield each in order and refill as it drains, so fetches overlap
        # each other and the downstream compute (bounded — the actor's output channel backpressures too).
        inflight = {pos: asyncio.ensure_future(issue(pos)) for pos in range(min(self._prefetch, n))}
        next_issue = len(inflight)
        for pos in range(n):
            data = await inflight.pop(pos)
            if next_issue < n:
                inflight[next_issue] = asyncio.ensure_future(issue(next_issue))
                next_issue += 1
            arrays = {v: np.asarray(d) for v, d in zip(self._vars, data, strict=True)}
            cols = self._slice_columns(sels[pos], arrays)
            names = list(cols)
            rb = pa.RecordBatch.from_arrays([pa.array(cols[k]) for k in names], names=names)
            for j in range(0, rb.num_rows, self._rpb):
                yield Batch(rb.slice(j, self._rpb))
        yield EOS_FRAME


class Wb2ForecastSource(SourceOperator):
    """The forecast side of the forecast-skill case (05): read the ``2m_temperature`` field from each model's
    WeatherBench2 store, one ``(init_time, lead)`` slice at a time, and stream ``(mlkey, gid, temp_f)`` rows.

    A single :class:`ZarrSliceSource` cannot express this — the forecasts live in *two* stores (Pangu,
    GraphCast) that a hash join needs merged into one left input — so this reads both and tags each row with
    ``mlkey = model·n_leads + lead`` (the group RMSE is taken over) and ``gid`` (the valid-time cell id the
    truth side is joined on). ``valid_gid(init_i, lead_i, cell_ids)`` maps a forecast slice (by its *local*
    init index and lead index) to its truth ``gid``, returning ``None`` when that valid time has no truth
    step. Each model has its own store time axis, so ``models`` pairs a model index with its store URL and
    its window's init times *as that store's indices* (the local init index is shared — the models cover the
    same init labels — but the positions differ)."""

    def __init__(
        self,
        models: list[tuple[int, str, list[int]]],
        n_leads: int,
        cell_ids: np.ndarray,
        valid_gid: Callable[[int, int, np.ndarray], np.ndarray | None],
        *,
        anonymous: bool = True,
        prefetch: int = 8,
        rows_per_batch: int = 262_144,
    ) -> None:
        self._models = models
        self._n_leads = n_leads
        self._cell_ids = cell_ids
        self._valid_gid = valid_gid
        self._anonymous = anonymous
        self._prefetch = max(1, prefetch)
        self._rpb = rows_per_batch

    async def frames(self) -> AsyncIterator[Frame]:
        import asyncio

        for model_idx, url, init_positions in self._models:
            arrs = await _open_arrays(url, "", ["2m_temperature"], self._anonymous)
            arr = arrs["2m_temperature"]
            # (local init index, store init position, lead index) per forecast slice.
            sels = [
                (ti, pos, li)
                for ti, pos in enumerate(init_positions)
                for li in range(self._n_leads)
            ]
            n = len(sels)

            def issue(pos: int, a=arr, s=sels):
                _, store_pos, li = s[pos]
                return asyncio.ensure_future(a.getitem((store_pos, li, slice(None), slice(None))))

            inflight = {p: issue(p) for p in range(min(self._prefetch, n))}
            next_issue = len(inflight)
            for pos in range(n):
                data = np.asarray(await inflight.pop(pos))
                if next_issue < n:
                    inflight[next_issue] = issue(next_issue)
                    next_issue += 1
                ti, _, li = sels[pos]
                gid = self._valid_gid(ti, li, self._cell_ids)
                if gid is None:  # this (init + lead) valid time has no matching truth step
                    continue
                mlkey = np.full(gid.size, model_idx * self._n_leads + li, dtype=np.int64)
                rb = pa.RecordBatch.from_arrays(
                    [pa.array(mlkey), pa.array(gid), pa.array(data.reshape(-1))],
                    names=["mlkey", "gid", "temp_f"],
                )
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
        chunks = await asyncio.gather(*(arr.getitem((i, slice(None), slice(None))) for i in idxs))
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
