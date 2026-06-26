"""The geospatial example also serves as a custom-operator reference, so its operators are tested here.
The example module is loaded by file path (examples/ is not an installed package)."""

import importlib.util
from pathlib import Path

from nautilus.operators import KeyedCount, KeyedTumblingSum
from nautilus.runtime.local import run
from nautilus.testing import data, from_batches, wm
from nautilus.windows import TumblingEventTimeWindows

_PATH = Path(__file__).resolve().parent.parent / "examples" / "geospatial.py"
_spec = importlib.util.spec_from_file_location("geospatial_example", _PATH)
assert _spec and _spec.loader
geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geo)


def test_within_bbox_keeps_only_inside_points():
    source = from_batches(
        data(id=[1, 2, 3], lon=[-122.42, -122.50, -122.41], lat=[37.77, 37.70, 37.79])
    )
    result = run(source, [geo.within_bbox(-122.44, 37.75, -122.40, 37.80)])
    assert [r["id"] for r in result.to_pylist()] == [1, 3]  # 2 is outside the box


def test_to_tile_bins_colocated_points_together():
    # two San Francisco points + one Paris point: the SF pair shares a tile at zoom 6
    source = from_batches(data(lon=[-122.41, -122.40, 2.35], lat=[37.77, 37.78, 48.85]))
    result = run(source, [geo.ToTile(zoom=6), KeyedCount("tile", "points")])
    counts = {r["tile"]: r["points"] for r in result.to_pylist()}
    assert sum(counts.values()) == 3
    assert max(counts.values()) == 2


def test_segment_distance_sums_per_vessel_per_window():
    source = from_batches(
        data(mmsi=["A"], lon=[-122.40], lat=[37.80], ts=[0]),
        data(mmsi=["A"], lon=[-122.30], lat=[37.80], ts=[20 * geo.MINUTE]),
        wm(geo.HOUR),
        data(mmsi=["A"], lon=[-122.30], lat=[37.80], ts=[70 * geo.MINUTE]),  # didn't move
        wm(2 * geo.HOUR),
    )
    result = run(
        source,
        [
            geo.SegmentDistance(),
            KeyedTumblingSum("mmsi", "metres", "ts", TumblingEventTimeWindows(geo.HOUR)),
        ],
    )
    by_hour = {r["window_start"] // geo.HOUR: r["sum"] for r in result.to_pylist()}
    assert by_hour[0] > 0  # moved during the first hour
    assert by_hour[1] == 0  # stationary during the second
