"""Average NDVI over a Sentinel-2 scene, read straight from cloud-optimized GeoTIFFs.

A stream of sentinel-2-l2a STAC item ids in, each scene's mean NDVI out, in three stages:

    Sentinel2NdviSource  →  TileNdvi (parallelism N)  →  MeanNdviByItem
    range-read + decode      NDVI per tile, reduced       average over a scene's
    a row of tiles           to (sum, count)              tiles
    ── I/O ──                ── CPU, fan-out ──            ── reduction ──

Fetch and decode both run in the source rather than a separate decode operator: an operator's
``process`` is synchronous and only sees an Arrow batch, but a range request must ``await`` and
async-geotiff's decoder is bound to a tile object that is not an Arrow value, so it cannot cross an edge.
The source emits the decoded red and near-infrared bands as Arrow tensor columns; decoding stays
async-geotiff's job, not ours.

Needs the geo extra (``pip install 'nautilus[geo]'``); run with ``nautilus run sentinel2-ndvi
--parallelism 4`` or ``python examples/sentinel2_ndvi.py <item-id> …``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
from async_geotiff import GeoTIFF
from obstore.store import S3Store

from nautilus.core.operator import Collector, OneInputOperator, OperatorContext, SourceOperator
from nautilus.core.records import EOS_FRAME, WATERMARK_MAX, Batch, Frame
from nautilus.driver.local import run
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
    and ``tile_count`` — the (across, down) tile grid the source walks."""

    handle: Any
    tile_count: tuple[int, int]


class TileReader(Protocol):
    """The source's I/O seam: open a COG at a resolution level, then fetch one internal tile decoded to an
    ``(H, W)`` array. Injecting it runs the pipeline without network in tests; the real one is
    :class:`AsyncGeotiffReader`."""

    async def open(self, href: str, level: int) -> Cog: ...

    async def fetch_tile(self, cog: Cog, x: int, y: int) -> np.ndarray: ...


#: Maps a STAC item id to its ``(red, nir)`` COG urls. Injected like the reader, so a test skips the STAC
#: call.
AssetResolver = Callable[[str], Awaitable[tuple[str, str]]]


class Sentinel2NdviSource(SourceOperator):
    """Streams a scene's tiles as decoded red/nir imagery — the I/O stage. For each item it resolves the
    red (B04) and near-infrared (B08) urls, opens both COGs at ``level``, and walks the tile grid one row
    at a time: a row's tiles are range-read and decoded concurrently, then emitted as one batch of
    ``(row, H, W)`` tensor columns. The row is the fan-out unit, and batching a row into one tensor (rather
    than a batch per tile) amortizes the Arrow construction the downstream stages pay per batch. The two
    10 m bands share a tile grid, so red and nir align by ``(x, y)``."""

    def __init__(
        self,
        item_ids: tuple[str, ...] | list[str],
        *,
        reader: TileReader | None = None,
        resolver: AssetResolver | None = None,
        level: int = DEFAULT_LEVEL,
    ) -> None:
        self.item_ids = list(item_ids)
        self.level = level
        self._reader = reader
        self._resolver = resolver

    async def frames(self) -> AsyncIterator[Frame]:
        reader = self._reader or AsyncGeotiffReader()
        resolve = self._resolver or resolve_sentinel2_assets
        for item_id in self.item_ids:
            red_href, nir_href = await resolve(item_id)
            red, nir = await asyncio.gather(
                reader.open(red_href, self.level), reader.open(nir_href, self.level)
            )
            nx, ny = red.tile_count
            for y in range(ny):
                tiles = await asyncio.gather(
                    *(reader.fetch_tile(red, x, y) for x in range(nx)),
                    *(reader.fetch_tile(nir, x, y) for x in range(nx)),
                )
                yield Batch(_row_batch(item_id, tiles[:nx], tiles[nx:]))
        yield EOS_FRAME


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
) -> tuple[SourceOperator, list[OneInputOperator]]:
    """The ``(source, transforms)`` the CLI and ``main`` run. Pass ``reader``/``resolver`` to run without
    network (the tests do)."""
    source = Sentinel2NdviSource(item_ids, reader=reader, resolver=resolver, level=level)
    return source, [TileNdvi(), MeanNdviByItem()]


def main() -> None:
    item_ids = sys.argv[1:] or list(DEFAULT_ITEM_IDS)
    source, transforms = sentinel2_ndvi(item_ids)
    result = run(source, transforms, parallelism=4)  # NDVI fans out across 4 instances
    for row in sorted(result.to_pylist(), key=lambda r: r["item_id"]):
        print(
            f"{row['item_id']}: mean NDVI {row['mean_ndvi']:.4f} over {row['valid_count']} pixels"
        )
    print(result.telemetry.summary)


if __name__ == "__main__":
    main()
