"""The Sentinel-2 NDVI example is also a custom source + async-transform + async-sink reference, so its
graph is tested here. The async-geotiff / network I/O is injected (a fake reader, resolver, and sink
writer), so these run hermetically; one opt-in test (``-m network``) exercises the real async-geotiff path
against the public bucket.

The example is a ``Stream`` graph (an awaiting decode stage needs explicit edges), so the pipeline is
built with ``sentinel2_ndvi(...) -> LogicalGraph`` and run with ``run_plan`` — not the linear
``run(source, transforms)``. The example module is loaded by file path (examples/ is not an installed
package), matching ``test_examples_geospatial``; it needs the geo extra to import, so the whole module
skips without it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from nautilus.core.operator import ListCollector, OperatorContext
from nautilus.core.records import Batch
from nautilus.driver.local import run
from nautilus.driver.run import run_plan
from nautilus.operators import from_batches
from nautilus.telemetry import TelemetryConfig, Tier

_PATH = Path(__file__).resolve().parent.parent / "examples" / "sentinel2_ndvi.py"
_spec = importlib.util.spec_from_file_location("sentinel2_example", _PATH)
assert _spec and _spec.loader
s2 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = s2  # so the module's @dataclass can resolve its own annotations
try:
    _spec.loader.exec_module(s2)
except ImportError:
    pytest.skip("Sentinel-2 example needs the geo extra (async-geotiff)", allow_module_level=True)

Scene = dict[str, dict[tuple[int, int], np.ndarray]]


class FakeReader:
    """A :class:`TileReader` backed by in-memory tiles, so the decode stage runs without network. The fake
    resolver hands out ``red://<item>`` / ``nir://<item>`` urls; ``open`` parses the band and item into the
    :class:`Cog` handle, and ``fetch_tile`` returns that band's tile for ``(x, y)``."""

    def __init__(self, scenes: dict[str, Scene]) -> None:
        self.scenes = scenes

    async def open(self, href: str, level: int) -> Any:
        band, item = href.split("://")
        coords = list(
            self.scenes[item]["red"]
        )  # both bands share a grid (10 m Sentinel-2 bands do)
        tile_count = (max(x for x, _ in coords) + 1, max(y for _, y in coords) + 1)
        return s2.Cog((band, item), tile_count)

    async def fetch_tile(self, cog: Any, x: int, y: int) -> np.ndarray:
        band, item = cog.handle
        return self.scenes[item][band][(x, y)]


async def _resolver(item_id: str) -> tuple[str, str]:
    return f"red://{item_id}", f"nir://{item_id}"


def _uniform_scene(
    nir: int, red: int, *, grid: int = 2, tile: int = 2, fill_corner: bool = False
) -> Scene:
    """A scene whose every valid pixel has the same NDVI, so the pixel-weighted mean is exactly that
    value. With ``fill_corner`` one pixel of tile (0, 0) is set to the 0 fill value in both bands.
    """
    scene: Scene = {"red": {}, "nir": {}}
    for y in range(grid):
        for x in range(grid):
            scene["red"][(x, y)] = np.full((tile, tile), red, np.uint16)
            scene["nir"][(x, y)] = np.full((tile, tile), nir, np.uint16)
    if fill_corner:
        scene["red"][(0, 0)][0, 0] = 0
        scene["nir"][(0, 0)][0, 0] = 0
    return scene


def _run(scenes: dict[str, Scene], *, parallelism: int = 1, **run_kwargs: Any) -> dict[str, Any]:
    graph = s2.sentinel2_ndvi(
        list(scenes), reader=FakeReader(scenes), resolver=_resolver, parallelism=parallelism
    )
    result = asyncio.run(run_plan(graph, **run_kwargs))
    return {row["item_id"]: row for row in result.to_pylist()}


def test_mean_ndvi_is_pixel_weighted_and_masks_fill() -> None:
    # nir=3000, red=1000 -> NDVI 0.5 for every valid pixel; the one fill pixel is excluded.
    rows = _run({"ITEM_A": _uniform_scene(3000, 1000, fill_corner=True)})
    assert set(rows) == {"ITEM_A"}
    assert rows["ITEM_A"]["mean_ndvi"] == pytest.approx(0.5)
    assert rows["ITEM_A"]["valid_count"] == 2 * 2 * 2 * 2 - 1  # 16 pixels, one fill


def test_fanout_parallel_matches_serial() -> None:
    # Decode and NDVI fan out and the mean shuffles by item; a parallel run must match a serial one.
    scenes = {"ITEM_A": _uniform_scene(4000, 2000, grid=3)}  # NDVI = 2000/6000 = 1/3
    serial = _run(scenes)
    parallel = _run(scenes, parallelism=3)
    assert serial["ITEM_A"]["mean_ndvi"] == pytest.approx(1 / 3)
    assert parallel["ITEM_A"]["mean_ndvi"] == pytest.approx(serial["ITEM_A"]["mean_ndvi"])
    assert parallel["ITEM_A"]["valid_count"] == serial["ITEM_A"]["valid_count"] == 3 * 3 * 2 * 2


def test_multiple_items_each_get_their_own_mean() -> None:
    rows = _run(
        {
            "GREEN": _uniform_scene(8000, 2000),  # NDVI 6000/10000 = 0.6 (vegetated)
            "WATER": _uniform_scene(1000, 1200),  # NDVI -200/2200 < 0 (water/built)
        }
    )
    assert rows["GREEN"]["mean_ndvi"] == pytest.approx(0.6)
    assert rows["WATER"]["mean_ndvi"] == pytest.approx(-200 / 2200)


def test_tile_ndvi_all_fill_tile_contributes_nothing() -> None:
    # A fully-fill tile (both bands 0) reduces to sum 0, count 0 — no divide-by-zero, no NaN leak.
    batch = s2._row_batch("X", [np.zeros((4, 4), np.uint16)], [np.zeros((4, 4), np.uint16)])
    row = run(from_batches(batch), [s2.TileNdvi()]).to_pylist()[0]
    assert row["valid_count"] == 0
    assert row["ndvi_sum"] == 0.0


async def test_source_lists_one_row_per_item() -> None:
    # The reworked source is a pure lister: one row (item_id, red_href, nir_href) per item, no pixels.
    source = s2.Sentinel2ItemSource(["ITEM_A", "ITEM_B"], resolver=_resolver)
    source.open(OperatorContext("source"))
    batches = [f.data async for f in source.frames() if isinstance(f, Batch)]
    assert len(batches) == 2 and all(b.num_rows == 1 for b in batches)
    assert set(batches[0].schema.names) == {"item_id", "red_href", "nir_href"}
    assert batches[0].column("red_href")[0].as_py() == "red://ITEM_A"


async def test_decode_stage_emits_one_batch_per_tile_row() -> None:
    # The fan-out granularity moved from the source to AsyncOpenAndDecode: fetch reads a scene, integrate
    # emits one (item_id, red, nir) tensor batch per tile-grid row (a 2x2 grid -> 2 batches of 2 tiles).
    op = s2.AsyncOpenAndDecode(reader=FakeReader({"ITEM_A": _uniform_scene(3000, 1000)}))
    op.open(OperatorContext("op0"))
    src = pa.record_batch(
        {"item_id": ["ITEM_A"], "red_href": ["red://ITEM_A"], "nir_href": ["nir://ITEM_A"]}
    )
    result = await op.fetch(src)
    collector = ListCollector()
    op.integrate(src, result, OperatorContext("op0"), collector)
    batches = collector.drain()
    assert len(batches) == 2  # two tile-rows
    assert all(b.num_rows == 2 for b in batches)  # two tiles per row
    assert s2.AsyncOpenAndDecode().key_columns() is None  # stateless — keyless, round-robin fan-out


def test_source_io_wait_and_decode_request_micros_recorded() -> None:
    # The source's awaited STAC resolve lands in io.wait_micros (part of its step); the decode stage's
    # awaited range reads land in async.request_micros — the async engine's own I/O attribution.
    async def slow_resolver(item_id: str) -> tuple[str, str]:
        await asyncio.sleep(0.002)  # stands in for the STAC item lookup
        return f"red://{item_id}", f"nir://{item_id}"

    class SlowReader(FakeReader):
        async def fetch_tile(self, cog: Any, x: int, y: int) -> np.ndarray:
            await asyncio.sleep(0.002)  # stands in for a range request
            return await super().fetch_tile(cog, x, y)

    graph = s2.sentinel2_ndvi(
        ["ITEM_A"],
        reader=SlowReader({"ITEM_A": _uniform_scene(3000, 1000, grid=3)}),
        resolver=slow_resolver,
    )
    report = asyncio.run(run_plan(graph, telemetry=TelemetryConfig(tier=Tier.COUNTERS))).telemetry
    src = {
        p.name: p.value for o in report.operators if o.operator_id == "source" for p in o.counters
    }
    assert src.get("io.wait_micros", 0) > 0  # the awaited resolve was captured
    assert src["io.wait_micros"] <= src["runtime.step_micros"]  # part of step, not on top
    request_micros = sum(
        p.value for o in report.operators for p in o.counters if p.name == "async.request_micros"
    )
    assert request_micros > 0  # the decode stage's awaited range reads were attributed


async def test_write_only_sink_writes_each_scene_and_returns_no_batches() -> None:
    # The --write variant terminates in an NdviSink: the means go to the (fake) store, so the run returns
    # no batches, and every scene's record is written under its item id (the idempotent per-item key).
    written: dict[str, dict[str, Any]] = {}

    async def writer(item_id: str, record: dict[str, Any]) -> None:
        written[item_id] = record

    scenes = {"GREEN": _uniform_scene(8000, 2000), "WATER": _uniform_scene(1000, 1200)}
    graph = s2.sentinel2_ndvi(
        list(scenes),
        reader=FakeReader(scenes),
        resolver=_resolver,
        sink=s2.NdviSink("mem://ignored", writer=writer),
    )
    result = await run_plan(graph)
    assert result.to_pylist() == []  # write-only: the data went to the sink, not a collector
    assert set(written) == {"GREEN", "WATER"}
    assert written["GREEN"]["mean_ndvi"] == pytest.approx(0.6)


@pytest.mark.network
def test_real_sentinel2_scene_mean_is_in_range() -> None:
    # End-to-end against the public bucket via async-geotiff, at the coarsest overview (a few tiles).
    rows = asyncio.run(run_plan(s2.sentinel2_ndvi())).to_pylist()
    assert len(rows) == 1
    assert rows[0]["valid_count"] > 0
    assert -1.0 <= rows[0]["mean_ndvi"] <= 1.0
