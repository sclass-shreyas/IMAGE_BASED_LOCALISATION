import json

from src.osm_matching.mock_cues import CUE_KEYS, MOCK_CUES, load_cues_from_openvino_json


def test_mock_cues_follow_the_contract():
    for cue in MOCK_CUES:
        assert set(cue) == set(CUE_KEYS)


def test_adapter_reads_person1_json(tmp_path):
    session = {
        "created_utc": "2026-10-24T10:00:00+00:00",
        "frames": [{
            "timestamp_utc": "2026-10-24T10:00:02.500000+00:00",
            "detections": [{
                "text": "mgroad", "confidence": 0.9, "detector_confidence": 0.8,
                "bbox_pixels": [[10, 20], [110, 20], [110, 60], [10, 60]],
            }],
        }],
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session), encoding="utf-8")

    cue = load_cues_from_openvino_json(path)[0]
    assert cue == {"timestamp_s": 2.5, "cue_type": "street_sign", "text": "mgroad",
                   "ocr_confidence": 0.9, "bbox_xywh": [10, 20, 100, 40]}