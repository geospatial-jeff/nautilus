"""Average NDVI over a Sentinel-2 scene, read straight from cloud-optimized GeoTIFFs.

A stream of sentinel-2-l2a STAC item ids in, each scene's mean NDVI out, as a four-stage graph:

    Sentinel2ItemSource → AsyncOpenAndDecode → TileNdvi (parallelism N) → MeanNdviByItem
    resolve red/nir hrefs   open + range-read     NDVI per tile, reduced      average over a
    per item                + decode a scene       to (sum, count)             scene's tiles
    ── metadata I/O ──      ── awaited I/O ──      ── CPU, fan-out ──          ── keyed reduce ──

The range-read + decode is an :class:`AsyncOneInputOperator`: its ``fetch`` opens both bands and reads a
tile-row's ranges concurrently (intra-scene overlap), while ``max_in_flight`` scenes decode at once
(inter-scene overlap), and its ``integrate`` emits each decoded tile-row as Arrow tensor columns. That is
why the pipeline is a ``Stream`` graph, not a linear ``(source, transforms)`` chain — an awaiting transform
needs the explicit-edge shape. The source is now a pure lister: it resolves each item's COG hrefs and emits
them, nothing more.

The default run collects each scene's mean NDVI and prints it. ``--write <uri>`` instead terminates in an
:class:`AsyncSink` that writes the means to an external store (one JSON object per item), and reports a
row-count + telemetry summary since the data left the pipeline.

Needs the geo extra (``pip install 'nautilus[geo]'``); run with ``nautilus run sentinel2-ndvi
--parallelism 4`` or ``python examples/sentinel2_ndvi.py <item-id> … [--write <uri>]``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np
import obstore
import pyarrow as pa
from async_geotiff import GeoTIFF
from obstore.store import S3Store, from_url

from nautilus.api import LogicalGraph
from nautilus.core.operator import (
    AsyncOneInputOperator,
    AsyncSink,
    Collector,
    OneInputOperator,
    OperatorContext,
    SourceOperator,
)
from nautilus.core.records import EOS_FRAME, WATERMARK_MAX, Batch, Frame
from nautilus.driver.run import run_plan
from nautilus.dsl import source as stream_source
from nautilus.state import KeyContext
from nautilus.tensors import tensor_array, to_numpy

STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

#: Any sentinel-2-l2a item works; NDVI needs red (B04) and near-infrared (B08), 10 m bands with fill 0.
DEFAULT_ITEM_IDS = ("S2B_2VMJ_20260629_0_L2A",)

#: Read the coarsest overview by default — a few tiles, a few seconds; ``0`` is full resolution.
DEFAULT_LEVEL = -1


@dataclass(frozen=True)
class Cog:
    """An opened COG at one resolution: the reader's opaque ``handle`` (passed back to ``fetch_tile``)
    and ``tile_count`` — the (across, down) tile grid the decode stage walks."""

    handle: Any
    tile_count: tuple[int, int]


class TileReader(Protocol):
    """The decode stage's I/O seam: open a COG at a resolution level, then fetch one internal tile decoded
    to an ``(H, W)`` array. Injecting it runs the pipeline without network in tests; the real one is
    :class:`AsyncGeotiffReader`."""

    async def open(self, href: str, level: int) -> Cog: ...

    async def fetch_tile(self, cog: Cog, x: int, y: int) -> np.ndarray: ...


#: Maps a STAC item id to its ``(red, nir)`` COG urls. Injected like the reader, so a test skips the STAC
#: call.
AssetResolver = Callable[[str], Awaitable[tuple[str, str]]]


class Sentinel2ItemSource(SourceOperator):
    """Lists each scene's COG urls — the metadata stage. For every item id it resolves the red (B04) and
    near-infrared (B08) hrefs and emits one row ``(item_id, red_href, nir_href)``; opening, range-reading,
    and decoding those COGs is the downstream :class:`AsyncOpenAndDecode`'s job. A lister awaits only the
    light STAC lookup, bracketed in ``ctx.io_wait()`` so the report separates that wait from compute.
    """

    def __init__(
        self,
        item_ids: tuple[str, ...] | list[str],
        *,
        resolver: AssetResolver | None = None,
    ) -> None:
        self.item_ids = list(item_ids)
        self._resolver = resolver
        self._ctx = OperatorContext("source")  # replaced in open(); records io.wait_micros once on

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx

    async def frames(self) -> AsyncIterator[Frame]:
        resolve = self._resolver or resolve_sentinel2_assets
        for item_id in self.item_ids:
            async with self._ctx.io_wait():  # the per-item STAC item lookup (metadata, not pixels)
                red_href, nir_href = await resolve(item_id)
            yield Batch(
                pa.record_batch(
                    {
                        "item_id": pa.array([item_id], pa.string()),
                        "red_href": pa.array([red_href], pa.string()),
                        "nir_href": pa.array([nir_href], pa.string()),
                    }
                )
            )
        yield EOS_FRAME  # bounded source: signal completion once every item is listed


class AsyncOpenAndDecode(AsyncOneInputOperator):
    """Opens each scene's COGs and range-reads + decodes its tiles — the awaited I/O stage, split out of
    the source (the Stage 6.4 rework). ``fetch`` opens both bands and reads a tile-row's ranges together
    (``asyncio.gather``, intra-scene overlap), while the engine keeps up to ``max_in_flight`` scenes
    decoding at once (inter-scene overlap); ``integrate`` emits each decoded tile-row as ``(item_id, red,
    nir)`` tensor columns — the fan-out unit :class:`TileNdvi` consumes. The row (not the whole scene) is
    the batch, so a coarse overview stays a handful of small batches.

    Stateless — no keyed state — so it could run ``ordered=False`` (completion order) to let a quick scene
    emit ahead of a slow one; it stays ordered here for a reproducible run. The reader is acquired in
    ``open`` (a fresh one per subtask when parallel), never ``__init__``, so the operator cloudpickles to a
    worker without a live client riding along."""

    def __init__(
        self,
        *,
        reader: TileReader | None = None,
        level: int = DEFAULT_LEVEL,
        max_in_flight: int = 8,
    ) -> None:
        self._reader = reader
        self._level = level
        self._cap = max_in_flight

    def open(self, ctx: OperatorContext) -> None:
        if (
            self._reader is None
        ):  # acquire the client here, not in __init__ — see the class docstring
            self._reader = AsyncGeotiffReader()

    def max_in_flight(self) -> int:
        return self._cap

    async def fetch(self, batch: pa.RecordBatch) -> object:
        reader = self._reader
        assert reader is not None  # set in open(), which the engine calls before any fetch
        scenes: list[tuple[str, list[tuple[list[np.ndarray], list[np.ndarray]]]]] = []
        for item_id, red_href, nir_href in zip(
            batch.column("item_id").to_pylist(),
            batch.column("red_href").to_pylist(),
            batch.column("nir_href").to_pylist(),
            strict=True,
        ):
            red, nir = await asyncio.gather(
                reader.open(red_href, self._level), reader.open(nir_href, self._level)
            )
            nx, ny = red.tile_count
            grid_rows = []
            for y in range(ny):  # a row of tiles for both bands, range-read + decoded concurrently
                tiles = await asyncio.gather(
                    *(reader.fetch_tile(red, x, y) for x in range(nx)),
                    *(reader.fetch_tile(nir, x, y) for x in range(nx)),
                )
                grid_rows.append((tiles[:nx], tiles[nx:]))
            scenes.append((item_id, grid_rows))
        return scenes

    def integrate(
        self, batch: pa.RecordBatch, result: object, ctx: OperatorContext, out: Collector
    ) -> None:
        for item_id, grid_rows in result:  # type: ignore[attr-defined]
            for red_tiles, nir_tiles in grid_rows:
                out.emit(_row_batch(item_id, red_tiles, nir_tiles))


def _row_batch(item_id: str, red: list[np.ndarray], nir: list[np.ndarray]) -> pa.RecordBatch:
    """One tile-grid row: each band's tiles stacked into an ``(n, H, W)`` tensor column, keyed by item."""
    return pa.record_batch(
        {
            "item_id": pa.array([item_id] * len(red), pa.string()),
            "red": tensor_array(red),
            "nir": tensor_array(nir),
        }
    )


class TileNdvi(OneInputOperator):
    """NDVI per tile, reduced to a per-tile ``(sum, valid count)`` — the CPU stage. NDVI is
    ``(nir - red) / (nir + red)`` per pixel, in ``[-1, 1]`` (high on vegetation, low on water and built
    surfaces). Pixels where ``nir + red == 0`` are dropped: Sentinel-2's fill is 0, so this masks off-swath
    fill and edge-tile padding and avoids the divide in one test. Emitting each tile's NDVI *sum* and valid
    *count* — not its mean — lets the scene average be the pixel-weighted ``Σsum / Σcount``, which a
    mean-of-means would get wrong across tiles of different valid counts. Keyless, so a parallel run
    round-robins tiles across instances — the fan-out."""

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        red = to_numpy(batch.column("red")).astype(np.float32)  # (n, H, W)
        nir = to_numpy(batch.column("nir")).astype(np.float32)
        total = nir + red
        valid = total > 0
        ndvi = np.divide(nir - red, total, out=np.zeros_like(total), where=valid)
        out.emit(
            pa.record_batch(
                {
                    "item_id": batch.column("item_id"),
                    "ndvi_sum": pa.array(ndvi.sum(axis=(1, 2), dtype=np.float64), pa.float64()),
                    "valid_count": pa.array(valid.sum(axis=(1, 2), dtype=np.int64), pa.int64()),
                }
            )
        )


class MeanNdviByItem(OneInputOperator):
    """Average NDVI per scene — the keyed reduction. Sums each item's ``(ndvi_sum, valid_count)`` partials
    in keyed state and, at end of stream, emits ``Σsum / Σcount`` per item: the same global keyed
    aggregation as :class:`~nautilus.operators.KeyedCount`, firing once at ``WATERMARK_MAX``. Keyed by
    item, so a parallel run gathers a scene's tiles onto one instance."""

    _STATE = "ndvi_acc"  # keyed state; each entry is a running (sum, count) pair

    def __init__(self, key_col: str = "item_id", mean_col: str = "mean_ndvi") -> None:
        self.key_col = key_col
        self.mean_col = mean_col

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx

    def key_columns(self) -> tuple[str, ...]:
        return (self.key_col,)

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        # Sum this batch's partials per item in Arrow, then fold each item's total into keyed state — one
        # state write per distinct item, not one per tile.
        grouped = (
            pa.table(
                {
                    "item": batch.column(self.key_col),
                    "s": batch.column("ndvi_sum"),
                    "c": batch.column("valid_count"),
                }
            )
            .group_by("item")
            .aggregate([("s", "sum"), ("c", "sum")])
        )
        items = grouped.column("item").to_pylist()
        sums = grouped.column("s_sum").to_pylist()
        counts = grouped.column("c_sum").to_pylist()
        for item, partial_sum, partial_count in zip(items, sums, counts, strict=True):
            self._ctx.reducing_state(self._STATE, KeyContext((item,)), _add_pair).add(
                (partial_sum, partial_count)
            )

    def on_watermark(self, t: int, out: Collector) -> None:
        if t < WATERMARK_MAX:
            return  # global aggregation: only the terminal watermark fires it
        items, means, counts, fired = [], [], [], []
        for kctx, (total, count) in self._ctx.entries(self._STATE):
            items.append(kctx.key[0])
            means.append(total / count if count else float("nan"))
            counts.append(count)
            fired.append(kctx)
        for kctx in fired:
            self._ctx.clear_state(self._STATE, kctx)
        if items:
            out.emit(
                pa.record_batch(
                    {
                        self.key_col: pa.array(items, pa.string()),
                        self.mean_col: pa.array(means, pa.float64()),
                        "valid_count": pa.array(counts, pa.int64()),
                    }
                )
            )


def _add_pair(a: tuple[float, int], b: tuple[float, int]) -> tuple[float, int]:
    return (a[0] + b[0], a[1] + b[1])


#: The sink's write seam: persist one scene's mean-NDVI ``record`` under a key derived from ``item_id``.
#: Injected in tests (an in-memory writer), so the sink path runs without S3.
NdviResultWriter = Callable[[str, dict[str, Any]], Awaitable[None]]


class NdviSink(AsyncSink):
    """Writes each scene's mean NDVI to an external store — the async-sink variant. One JSON object per
    item, keyed by item id, so a whole-job replay overwrites the same object rather than appending a
    duplicate: the at-least-once idempotency an :class:`AsyncSink` requires. ``key_columns`` co-partitions
    each item to one writer instance. The writer is acquired in ``open`` (obstore over ``uri``, or an
    injected one for tests), never ``__init__``, so the sink cloudpickles to a worker cleanly."""

    def __init__(self, uri: str, *, writer: NdviResultWriter | None = None) -> None:
        self._uri = uri
        self._writer = writer

    def open(self, ctx: OperatorContext) -> None:
        if self._writer is None:  # build the real obstore-backed writer here, not in __init__
            self._writer = _obstore_writer(self._uri)

    def key_columns(self) -> tuple[str, ...]:
        return ("item_id",)  # per-item upsert: a key's write lands on one instance

    async def write(self, batch: pa.RecordBatch) -> None:
        assert self._writer is not None  # set in open()
        for row in batch.to_pylist():
            await self._writer(row["item_id"], row)


def _obstore_writer(uri: str) -> NdviResultWriter:
    """The default :data:`NdviResultWriter`: PUT each record as ``<uri>/<item_id>.json`` via obstore, an
    overwrite keyed by item id (idempotent under whole-job replay)."""
    store = from_url(uri)

    async def write(item_id: str, record: dict[str, Any]) -> None:
        await obstore.put_async(store, f"{item_id}.json", json.dumps(record).encode())

    return write


async def resolve_sentinel2_assets(item_id: str) -> tuple[str, str]:
    """Resolve an item id to its ``(red, nir)`` COG urls via the earth-search STAC API. The blocking GET
    runs in a thread so it never stalls the source's event loop."""
    url = f"{STAC_ENDPOINT}/collections/{COLLECTION}/items/{item_id}"
    item = await asyncio.to_thread(_get_json, url)
    assets = item["assets"]
    return assets["red"]["href"], assets["nir"]["href"]


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(
        url, timeout=30
    ) as response:  # noqa: S310 (fixed, trusted endpoint)
        return json.load(response)


class AsyncGeotiffReader:
    """The default reader: opens COGs with async-geotiff over obstore and returns each internal tile
    decoded to an ``(H, W)`` uint16 array. Reads are anonymous range requests against the public
    Sentinel-2 bucket."""

    def __init__(self) -> None:
        self._store = S3Store("sentinel-cogs", region="us-west-2", skip_signature=True)

    async def open(self, href: str, level: int) -> Cog:
        cog = await GeoTIFF.open(urlparse(href).path.lstrip("/"), store=self._store)
        obj = [cog, *cog.overviews][level]  # 0 = full resolution, then overviews, coarsest last
        return Cog(obj, obj.tile_count)

    async def fetch_tile(self, cog: Cog, x: int, y: int) -> np.ndarray:
        tile = await cog.handle.fetch_tile(x, y)  # range request + threaded decode
        return np.asarray(tile.array.data)[0]  # (1, H, W) single band -> (H, W)


def sentinel2_ndvi(
    item_ids: tuple[str, ...] | list[str] = DEFAULT_ITEM_IDS,
    *,
    level: int = DEFAULT_LEVEL,
    reader: TileReader | None = None,
    resolver: AssetResolver | None = None,
    parallelism: int = 1,
    write_uri: str | None = None,
    sink: AsyncSink | None = None,
) -> LogicalGraph:
    """The NDVI graph the CLI and ``main`` run. The default collects each scene's mean NDVI; pass
    ``write_uri`` (or an injected ``sink``) to terminate in an :class:`NdviSink` instead — a write-only run
    whose result carries telemetry and no batches. ``reader``/``resolver`` inject the I/O seams so a test
    runs without network; ``parallelism`` sets the decode/NDVI/reduce width (the source stays single).
    """
    stream = (
        stream_source(Sentinel2ItemSource(item_ids, resolver=resolver))
        .apply_async(AsyncOpenAndDecode(reader=reader, level=level), parallelism=parallelism)
        .apply(TileNdvi(), parallelism=parallelism)
        .apply(MeanNdviByItem(), parallelism=parallelism)
    )
    if write_uri is not None or sink is not None:
        the_sink = sink if sink is not None else NdviSink(write_uri or "")
        return stream.sink(the_sink, parallelism=parallelism).to_graph()
    return stream.to_graph()


def main() -> None:
    parser = argparse.ArgumentParser(description="Average NDVI over Sentinel-2 scenes")
    parser.add_argument("item_ids", nargs="*", default=list(DEFAULT_ITEM_IDS))
    parser.add_argument(
        "--write", metavar="URI", help="write the means to an external store instead"
    )
    parser.add_argument("--parallelism", type=int, default=4)
    args = parser.parse_args()
    item_ids = args.item_ids or list(DEFAULT_ITEM_IDS)
    graph = sentinel2_ndvi(item_ids, parallelism=args.parallelism, write_uri=args.write)
    result = asyncio.run(run_plan(graph))  # NDVI fans out across --parallelism instances
    if args.write:
        # A write-only run's data went to the store, so there are no rows to print — report the shape.
        print(f"wrote mean NDVI for {len(item_ids)} scene(s) to {args.write}")
    else:
        for row in sorted(result.to_pylist(), key=lambda r: r["item_id"]):
            print(
                f"{row['item_id']}: mean NDVI {row['mean_ndvi']:.4f} over {row['valid_count']} pixels"
            )
    print(result.telemetry.summary)


if __name__ == "__main__":
    main()
