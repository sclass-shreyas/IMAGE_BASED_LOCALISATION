from src.osm_matching.normalize import compact_key, name_keys, normalize_text


def test_case_and_punctuation():
    assert normalize_text("M.G. Road") == "mg road"
    assert normalize_text("MG ROAD") == "mg road"
    assert normalize_text("  mg    road ") == "mg road"


def test_abbreviation_expansion():
    assert normalize_text("M G Rd.") == "mg road"
    assert normalize_text("10th Cr Rd") == "10th cross road"


def test_ordinals_are_joined():
    assert normalize_text("1 st Cross Rd") == "1st cross road"
    assert normalize_text("3 RD cross") == "3rd cross"
    assert normalize_text("1ST CROSS") == "1st cross"


def test_positional_abbreviations():
    assert normalize_text("Brigade St.") == "brigade street"
    assert normalize_text("St Marks Road") == "saint marks road"
    assert normalize_text("Dr Rajkumar Road") == "doctor rajkumar road"
    assert normalize_text("Hospital Dr") == "hospital drive"


def test_compact_key():
    assert compact_key("MG ROAD") == "mgroad"
    assert compact_key("M.G. Rd") == "mgroad"


def test_empty_and_non_latin():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    assert normalize_text("\u0cae\u0cb9\u0cbe\u0ca4\u0ccd\u0cae") == ""
    assert name_keys(None) == set()


def test_osm_name_keys_include_initials():
    keys = name_keys("Mahatma Gandhi Road")
    assert "mahatmagandhiroad" in keys
    assert "mgroad" in keys          # matches what the OCR model would output for "MG ROAD"


def test_osm_name_keys_no_false_initials():
    assert name_keys("Cubbon Road") == {"cubbonroad"}
    assert name_keys("1st Cross Road") == {"1stcrossroad"}


def test_ocr_cue_matches_osm_key():
    assert compact_key("mgroad") in name_keys("Mahatma Gandhi Road")


def test_initials_keep_every_letter():
    assert "btmroad" in name_keys("B T Muniyappa Road")
    assert "bbnroad" in name_keys("B.B. Naidu Road")
    assert "bvkiroad" in name_keys("B.V.K. Iyengar Road")