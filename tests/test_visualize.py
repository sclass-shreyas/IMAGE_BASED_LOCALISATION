import geopandas as gpd
from shapely.geometry import LineString

from src.osm_matching.candidates import CandidateGenerator
from src.osm_matching.visualize import make_map


def test_map_with_road_background(tmp_path):
    roads = gpd.GeoDataFrame(
        {
            "name": ["Cubbon Road", "Side Lane"],
            "osm_ids": ["1", "2"],
            "highway": ["primary", "residential"],
            "length_m": [100.0, 50.0],
            "geometry": [
                LineString([(77.5900, 12.9700), (77.5910, 12.9700)]),
                LineString([(77.5905, 12.9700), (77.5905, 12.9710)]),
            ],
        },
        crs="EPSG:4326",
    )
    result = CandidateGenerator(roads).match_cue({"text": "cubbon road"})
    path = make_map(result, tmp_path / "map.html", roads=roads)
    text = path.read_text(encoding="utf-8")
    assert "Cubbon Road" in text
    assert "OSM road network" in text