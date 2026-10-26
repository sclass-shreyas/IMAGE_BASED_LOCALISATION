import pandas as pd
import pytest

from src.osm_matching.matcher import StreetIndex


def _index(names):
    return StreetIndex(pd.DataFrame({"name": names, "length_m": [1.0] * len(names)}))


@pytest.fixture
def index():
    roads = pd.DataFrame({
        "name": ["Mahatma Gandhi Road", "Mahatma Gandhi Road", "Cubbon Road", "1st Cross Road", None],
        "length_m": [100.0, 50.0, 200.0, 80.0, 30.0],
    })
    return StreetIndex(roads)


def test_initials_cue_finds_full_name(index):
    top = index.match("mgroad")[0]
    assert top["name"] == "Mahatma Gandhi Road"
    assert top["match_type"] == "initials"
    assert top["match_score"] == 1.0
    assert top["segments"] == 2 and top["length_m"] == 150.0


def test_punctuated_cue(index):
    assert index.match("M.G. Rd")[0]["name"] == "Mahatma Gandhi Road"


def test_ocr_character_error(index):
    assert index.match("mgr0ad")[0]["name"] == "Mahatma Gandhi Road"


def test_full_name_match(index):
    top = index.match("Cubbon Road")[0]
    assert top["name"] == "Cubbon Road" and top["match_type"] == "full"


def test_cue_without_road_word(index):
    top = index.match("1st cross")[0]
    assert top["name"] == "1st Cross Road" and top["match_score"] == 1.0


def test_garbage_and_too_short_return_nothing(index):
    assert index.match("zzzzzz") == []
    assert index.match("rd") == []
    assert index.match("") == []
    assert index.match(None) == []


def test_limit(index):
    assert len(index.match("gandhi road", limit=1, min_score=0.3)) == 1


def test_similar_initials_are_not_returned_as_noise():
    index = _index(["Mahatma Gandhi Road", "Annaswamy Mudaliar Road"])
    assert [r["name"] for r in index.match("mgroad")] == ["Mahatma Gandhi Road"]


def test_short_initials_key_does_not_match_long_cue():
    index = _index(["Cubbon Road", "B.B. Naidu Road"])
    assert [r["name"] for r in index.match("cubbon rd")] == ["Cubbon Road"]