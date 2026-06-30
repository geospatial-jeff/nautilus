"""The Sentinel-2 NDVI example is also a custom source + operator reference, so its pipeline is tested
here. The async-geotiff / network I/O is injected (a fake reader + resolver), so these run hermetically;
one opt-in test (``-m network``) exercises the real async-geotiff path against the public bucket.

The example module is loaded by file path (examples/ is not an installed package), matching
``test_examples_geospatial``; it needs the geo extra to import, so the whole module skips without it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nautilus.driver.local import run
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
    """A :class:`TileReader` backed by in-memory tiles, so the pipeline runs without network. The fake
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


def _run(scenes: dict[str, Scene], **run_kwargs: Any) -> dict[str, Any]:
    source, transforms = s2.sentinel2_ndvi(
        list(scenes), reader=FakeReader(scenes), resolver=_resolver
    )
    return {row["item_id"]: row for row in run(source, transforms, **run_kwargs).to_pylist()}


def test_mean_ndvi_is_pixel_weighted_and_masks_fill() -> None:
    # nir=3000, red=1000 -> NDVI 0.5 for every valid pixel; the one fill pixel is excluded.
    rows = _run({"ITEM_A": _uniform_scene(3000, 1000, fill_corner=True)})
    assert set(rows) == {"ITEM_A"}
    assert rows["ITEM_A"]["mean_ndvi"] == pytest.approx(0.5)
    assert rows["ITEM_A"]["valid_count"] == 2 * 2 * 2 * 2 - 1  # 16 pixels, one fill


def test_fanout_parallel_matches_serial() -> None:
    # NDVI fans out over tile rows and the mean shuffles by item; a parallel run must match a serial one.
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


async def test_source_emits_one_batch_per_tile_row() -> None:
    # The fan-out granularity: one batch per tile-grid row (a 2x2 grid -> 2 batches of 2 tiles each).
    from nautilus.core.records import Batch

    source = s2.Sentinel2NdviSource(
        ["ITEM_A"], reader=FakeReader({"ITEM_A": _uniform_scene(3000, 1000)}), resolver=_resolver
    )
    batches = [f.data async for f in source.frames() if isinstance(f, Batch)]
    assert len(batches) == 2  # two rows
    assert all(b.num_rows == 2 for b in batches)  # two tiles per row


def test_source_io_wait_is_recorded_separately_from_compute() -> None:
    # ctx.io_wait() must capture the source's awaited I/O as io.wait_micros — the metric that tells an
    # I/O-bound source from a compute-bound one (its runtime.step_micros counts both).
    class SlowReader(FakeReader):
        async def fetch_tile(self, cog: Any, x: int, y: int) -> np.ndarray:
            await asyncio.sleep(0.002)  # stands in for a range request
            return await super().fetch_tile(cog, x, y)

    source, transforms = s2.sentinel2_ndvi(
        ["ITEM_A"],
        reader=SlowReader({"ITEM_A": _uniform_scene(3000, 1000, grid=3)}),
        resolver=_resolver,
    )
    report = run(source, transforms, telemetry=TelemetryConfig(tier=Tier.COUNTERS)).telemetry
    counters = {
        p.name: p.value for o in report.operators if o.operator_id == "source" for p in o.counters
    }
    assert counters.get("io.wait_micros", 0) > 0  # the awaited sleeps were captured
    assert counters["io.wait_micros"] <= counters["runtime.step_micros"]  # part of step, not on top


@pytest.mark.network
def test_real_sentinel2_scene_mean_is_in_range() -> None:
    # End-to-end against the public bucket via async-geotiff, at the coarsest overview (a few tiles).
    rows = run(*s2.sentinel2_ndvi()).to_pylist()
    assert len(rows) == 1
    assert rows[0]["valid_count"] > 0
    assert -1.0 <= rows[0]["mean_ndvi"] <= 1.0
