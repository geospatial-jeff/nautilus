"""Geospatial streaming demos on nautilus.

Three use cases, from simplest to most involved:

1. ``aoi_filter``     — clip a point stream to an area of interest (stateless, vectorized Arrow).
2. ``tile_density``   — bin points to web-mercator map tiles and count per tile (custom map + keyed
                        count): the streaming version of a point-density heatmap.
3. ``vessel_distance``— per-vessel great-circle distance travelled per event-time hour, from an
                        AIS-style position stream (keyed state + tumbling event-time windows).

Run with:  python examples/geospatial.py
"""

from __future__ import annotations

import math

import pyarrow as pa
import pyarrow.compute as pc

from nautilus.core.operator import Collector, OneInputOperator, OperatorContext
from nautilus.operators import FilterRows, KeyedCount, KeyedTumblingSum
from nautilus.runtime.local import run
from nautilus.state import KeyContext
from nautilus.testing import data, from_batches, wm
from nautilus.windows import TumblingEventTimeWindows

MINUTE = 60_000_000  # microseconds
HOUR = 60 * MINUTE


# --- 1. Area-of-interest filter ----------------------------------------------------------------


def within_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> FilterRows:
    """Keep only points whose (lon, lat) fall inside a bounding box. Pure Arrow compute, no Python
    loop — the whole batch is masked at once."""

    def mask(batch: pa.RecordBatch) -> pa.Array:
        lon, lat = batch.column("lon"), batch.column("lat")
        return pc.and_(
            pc.and_(pc.greater_equal(lon, min_lon), pc.less_equal(lon, max_lon)),
            pc.and_(pc.greater_equal(lat, min_lat), pc.less_equal(lat, max_lat)),
        )

    return FilterRows(mask)


def aoi_filter() -> None:
    # GPS pings around San Francisco; the AOI is a box over the city centre.
    source = from_batches(
        data(
            id=[1, 2, 3, 4, 5],
            lon=[-122.42, -122.45, -122.39, -122.51, -122.41],
            lat=[37.77, 37.80, 37.74, 37.70, 37.79],
        ),
    )
    result = run(source, [within_bbox(-122.44, 37.75, -122.40, 37.80)])
    kept = result.to_pylist()
    print(f"area-of-interest filter: {len(kept)}/5 pings inside the box")
    for row in kept:
        print(f"  id={row['id']}  ({row['lon']:.2f}, {row['lat']:.2f})")


# --- 2. Web-mercator tile density --------------------------------------------------------------


class ToTile(OneInputOperator):
    """Map each (lon, lat) point to its web-mercator slippy-map tile ``z/x/y`` at a fixed zoom — the
    same tiling scheme web maps use. Emits a one-column ``tile`` batch to feed a keyed count."""

    def __init__(
        self, zoom: int, lon_col: str = "lon", lat_col: str = "lat", out_col: str = "tile"
    ):
        self.zoom = zoom
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.out_col = out_col

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        n = 2**self.zoom
        lons = batch.column(self.lon_col).to_pylist()
        lats = batch.column(self.lat_col).to_pylist()
        tiles: list[str] = []
        for lon, lat in zip(lons, lats, strict=True):
            x = int((lon + 180.0) / 360.0 * n)
            lat_rad = math.radians(lat)
            y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            x, y = min(max(x, 0), n - 1), min(max(y, 0), n - 1)
            tiles.append(f"{self.zoom}/{x}/{y}")
        out.emit(pa.RecordBatch.from_arrays([pa.array(tiles)], names=[self.out_col]))


def tile_density() -> None:
    # Point clusters in San Francisco, Paris and Tokyo.
    source = from_batches(
        data(
            lon=[-122.41, -122.40, -122.42, 2.35, 2.34, 139.69],
            lat=[37.77, 37.78, 37.76, 48.85, 48.86, 35.68],
        ),
    )
    result = run(source, [ToTile(zoom=6), KeyedCount("tile", "points")])
    rows = sorted(result.to_pylist(), key=lambda r: (-r["points"], r["tile"]))
    print("\ntile density (zoom 6): points per map tile")
    for row in rows:
        print(f"  {row['tile']:>10}  {row['points']} points")


# --- 3. Per-vessel distance per event-time hour ------------------------------------------------


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> int:
    """Great-circle distance between two lon/lat points, in whole metres."""
    r = 6_371_008.8  # mean Earth radius, metres
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)))


class SegmentDistance(OneInputOperator):
    """For an AIS-style stream keyed by ``mmsi`` (vessel id), emit the great-circle distance (metres)
    from each vessel's previous position. The last position per vessel lives in keyed state; a
    vessel's first ping emits 0. Passes ``mmsi`` and ``ts`` through so a window can sum per vessel.
    """

    def open(self, ctx: OperatorContext) -> None:
        self._ctx = ctx

    def process(self, batch: pa.RecordBatch, out: Collector) -> None:
        mmsis = batch.column("mmsi").to_pylist()
        lons = batch.column("lon").to_pylist()
        lats = batch.column("lat").to_pylist()
        tss = pc.cast(batch.column("ts"), pa.int64()).to_pylist()
        out_metres: list[int] = []
        for mmsi, lon, lat in zip(mmsis, lons, lats, strict=True):
            last = self._ctx.value_state("last_pos", KeyContext((mmsi,)))
            prev = last.value()
            out_metres.append(0 if prev is None else _haversine_m(prev[0], prev[1], lon, lat))
            last.update((lon, lat))
        out.emit(
            pa.RecordBatch.from_arrays(
                [pa.array(mmsis), pa.array(out_metres, pa.int64()), pa.array(tss, pa.int64())],
                names=["mmsi", "metres", "ts"],
            )
        )


def vessel_distance() -> None:
    # Two vessels reporting positions over two hours (ts in event-time microseconds).
    source = from_batches(
        data(mmsi=["A", "B"], lon=[-122.40, -122.50], lat=[37.80, 37.70], ts=[0, 5 * MINUTE]),
        data(
            mmsi=["A", "B"],
            lon=[-122.30, -122.45],
            lat=[37.82, 37.72],
            ts=[25 * MINUTE, 30 * MINUTE],
        ),
        wm(HOUR),  # close the first hourly window
        data(
            mmsi=["A", "B"],
            lon=[-122.10, -122.40],
            lat=[37.85, 37.74],
            ts=[70 * MINUTE, 80 * MINUTE],
        ),
        wm(2 * HOUR),  # close the second
    )
    result = run(
        source,
        [
            SegmentDistance(),
            KeyedTumblingSum("mmsi", "metres", "ts", TumblingEventTimeWindows(HOUR)),
        ],
    )
    rows = sorted(result.to_pylist(), key=lambda r: (r["mmsi"], r["window_start"]))
    print("\nvessel distance per event-time hour:")
    for row in rows:
        hour = row["window_start"] // HOUR
        print(f"  vessel {row['mmsi']}  hour {hour}  {row['sum'] / 1000:.2f} km")


def main() -> None:
    aoi_filter()
    tile_density()
    vessel_distance()


if __name__ == "__main__":
    main()
