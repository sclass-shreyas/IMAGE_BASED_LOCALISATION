import geopandas as gpd
from shapely.geometry import LineString

from src.osm_matching.data_loader import _dedupe


def _frame(rows):
    return gpd.GeoDataFrame(
        {
            "osm_ids": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "highway": "residential",
            "length_m": 10.0,
            "geometry": [LineString(r[2]) for r in rows],
        },
        crs="EPSG:4326",
    )


def test_dedupe_drops_reverse_direction_copy():
    roads = _frame([
        ("1", "Cubbon Road", [(0, 0), (1, 0)]),
        ("1", "Cubbon Road", [(1, 0), (0, 0)]),   # same way, opposite direction
    ])
    assert len(_dedupe(roads)) == 1


def test_dedupe_keeps_different_ways_and_names():
    roads = _frame([
        ("1", "Cubbon Road", [(0, 0), (1, 0)]),
        ("2", "Cubbon Road", [(0, 0), (1, 0)]),   # different OSM way
        ("1", "Other Road", [(0, 0), (1, 0)]),    # different name
        ("3", None, [(0, 0), (1, 0)]),            # unnamed
    ])
    assert len(_dedupe(roads)) == 4