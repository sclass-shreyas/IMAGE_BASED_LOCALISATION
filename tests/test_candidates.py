import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from src.osm_matching import config
from src.osm_matching.candidates import CandidateGenerator
from src.osm_matching.visualize import make_map

ROWS = [
    ("Mahatma Gandhi Road", "1", 100.0, [(77.5900, 12.9700), (77.5910, 12.9700)]),
    ("1st Cross Road", "11", 120.0, [(77.5900, 12.9800), (77.5910, 12.9800)]),
    ("1st Cross Road", "21", 90.0, [(77.6100, 12.9900), (77.6110, 12.9900)]),   # same name, far away
    ("Cubbon Road", "31", 80.0, [(77.5950, 12.9750), (77.5960, 12.9750)]),
]


def _cue(text, **extra):
    return {"timestamp_s": 12.4, "cue_type": "street_sign", "text": text,
            "ocr_confidence": 0.91, "bbox_xywh": [120, 75, 240, 58], **extra}


@pytest.fixture
def generator():
    roads = gpd.GeoDataFrame(
        {
            "name": [r[0] for r in ROWS],
            "osm_ids": [r[1] for r in ROWS],
            "length_m": [r[2] for r in ROWS],
            "geometry": [LineString(r[3]) for r in ROWS],
        },
        crs="EPSG:4326",
    )
    return CandidateGenerator(roads)


def test_contract_fields(generator):
    result = generator.match_cue(_cue("MG ROAD"))
    assert result["timestamp_s"] == 12.4
    assert result["query_text"] == "MG ROAD"
    assert result["normalized_text"] == "mg road"
    top = result["candidates"][0]
    assert top["name"] == "Mahatma Gandhi Road"
    assert top["candidate_id"] == "way/1" and top["osm_type"] == "way" and top["osm_id"] == 1
    assert 0.0 <= top["match_score"] <= 1.0
    assert top["representative_lat"] == pytest.approx(12.97, abs=1e-6)
    assert result["ambiguous"] is False


def test_ambiguity_is_explicit(generator):
    result = generator.match_cue(_cue("1st cross"))
    assert [c["candidate_id"] for c in result["candidates"]] == ["way/11", "way/21"]
    assert all(c["same_name_pieces"] == 2 for c in result["candidates"])
    assert result["ambiguous"] is True
    assert result["n_candidates_total"] == 2 and result["truncated"] is False


def test_unmatched_text_returns_no_candidates(generator):
    for text in ("zzzzzz", "", None):
        result = generator.match_cue(_cue(text))
        assert result["candidates"] == []
        assert result["ambiguous"] is False and result["n_candidates_total"] == 0


def test_result_is_json_serializable(generator):
    json.dumps(generator.match_cue(_cue("1st cross")))


def test_geometry_can_be_left_out(generator):
    assert "segments" in generator.match_cue(_cue("MG ROAD"))["candidates"][0]
    assert "segments" not in generator.match_cue(_cue("MG ROAD"), include_geometry=False)["candidates"][0]


def test_truncation_is_reported(generator, monkeypatch):
    monkeypatch.setattr(config, "MAX_PIECES_TOTAL", 1)
    result = generator.match_cue(_cue("1st cross"))
    assert len(result["candidates"]) == 1
    assert result["n_candidates_total"] == 2 and result["truncated"] is True


def test_missing_timestamp_does_not_crash(generator):
    cue = _cue("MG ROAD")
    del cue["timestamp_s"]
    assert generator.match_cue(cue)["timestamp_s"] is None


def test_map_is_written(generator, tmp_path):
    result = generator.match_cue(_cue("MG ROAD"))
    path = make_map(result, tmp_path / "map.html")
    assert path.exists()
    assert "Mahatma Gandhi Road" in path.read_text(encoding="utf-8")