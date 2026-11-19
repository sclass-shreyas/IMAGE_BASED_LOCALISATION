import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from src.osm_matching.geometry import StreetGeometry


def _roads(rows):
    """rows: (name, osm_ids, length_m, [(lon, lat), ...])"""
    return gpd.GeoDataFrame(
        {
            "name": [r[0] for r in rows],
            "osm_ids": [r[1] for r in rows],
            "length_m": [r[2] for r in rows],
            "geometry": [LineString(r[3]) for r in rows],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def geo():
    return StreetGeometry(_roads([
        # piece A: two touching segments
        ("1st Cross Road", "11", 120.0, [(77.5900, 12.9700), (77.5910, 12.9700)]),
        ("1st Cross Road", "12|13", 100.0, [(77.5910, 12.9700), (77.5920, 12.9700)]),
        # piece B: same name, about 3 km away
        ("1st Cross Road", "21", 90.0, [(77.6100, 12.9900), (77.6110, 12.9900)]),
        ("Palace Road", "31", 50.0, [(77.5900, 12.9800), (77.5905, 12.9800)]),
    ]))


def test_same_name_is_split_into_separate_pieces(geo):
    pieces = geo.pieces("1st Cross Road")
    assert [p["n_segments"] for p in pieces] == [2, 1]
    assert [p["piece_index"] for p in pieces] == [0, 1]


def test_piece_contents(geo):
    p = geo.pieces("1st Cross Road")[0]
    assert p["candidate_id"] == "way/11" and p["osm_type"] == "way" and p["osm_id"] == 11
    assert p["osm_way_ids"] == [11, 12, 13]
    assert p["length_m"] == 220.0
    assert p["representative_lat"] == pytest.approx(12.97, abs=1e-6)
    assert p["bbox"][0] <= p["representative_lon"] <= p["bbox"][2]
    assert p["segments"][0][0] == [12.97, 77.59]        # [lat, lon] order
    assert geo.pieces("1st Cross Road")[1]["candidate_id"] == "way/21"


def test_output_is_json_serializable(geo):
    json.dumps(geo.pieces("1st Cross Road"))


def test_unknown_name_returns_empty(geo):
    assert geo.pieces("Nowhere Road") == []
    assert geo.pieces(None) == []


def test_nearby_segments_within_gap_are_joined():
    rows = [
        ("Test Road", "1", 100.0, [(77.5900, 12.9700), (77.5910, 12.9700)]),
        ("Test Road", "2", 100.0, [(77.5911, 12.9700), (77.5921, 12.9700)]),   # about 11 m gap
    ]
    assert len(StreetGeometry(_roads(rows)).pieces("Test Road")) == 1
    assert len(StreetGeometry(_roads(rows), gap_m=5).pieces("Test Road")) == 2