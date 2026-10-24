"""Cue inputs for testing Person 2's module without Person 1's live OCR."""
import json
from datetime import datetime
from pathlib import Path

CUE_KEYS = ("timestamp_s", "cue_type", "text", "ocr_confidence", "bbox_xywh")

# Cues in the brief's contract format. The street names belong to the PLACEHOLDER area;
# edit them once the team picks the corridor.
MOCK_CUES = [
    {"timestamp_s": 1.0, "cue_type": "street_sign", "text": "MG ROAD", "ocr_confidence": 0.91, "bbox_xywh": [120, 75, 240, 58]},
    {"timestamp_s": 4.0, "cue_type": "street_sign", "text": "mgroad", "ocr_confidence": 0.83, "bbox_xywh": [100, 60, 200, 50]},
    {"timestamp_s": 8.0, "cue_type": "street_sign", "text": "CUBBON RD", "ocr_confidence": 0.88, "bbox_xywh": [90, 70, 220, 55]},
    {"timestamp_s": 12.0, "cue_type": "street_sign", "text": "1st cross", "ocr_confidence": 0.79, "bbox_xywh": [110, 80, 180, 45]},
    {"timestamp_s": 15.0, "cue_type": "street_sign", "text": "palaceroad", "ocr_confidence": 0.74, "bbox_xywh": [95, 65, 210, 52]},
    {"timestamp_s": 18.0, "cue_type": "street_sign", "text": "xqzvkw", "ocr_confidence": 0.40, "bbox_xywh": [60, 40, 90, 30]},  # false detection
]


def _bbox_xywh(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def cue_from_openvino_detection(detection, timestamp_s, cue_type="street_sign"):
    """Convert one detection from Person 1's script into the contract's cue format."""
    return {
        "timestamp_s": round(timestamp_s, 3),
        "cue_type": cue_type,
        "text": detection["text"],
        "ocr_confidence": float(detection["confidence"]),
        "bbox_xywh": _bbox_xywh(detection["bbox_pixels"]),
    }


def load_cues_from_openvino_json(path):
    """Read the JSON written by openvino_webcam_test.py. timestamp_s = seconds since the session started."""
    session = json.loads(Path(path).read_text(encoding="utf-8"))
    start = datetime.fromisoformat(session["created_utc"])
    cues = []
    for frame in session.get("frames", []):
        seconds = (datetime.fromisoformat(frame["timestamp_utc"]) - start).total_seconds()
        for detection in frame.get("detections", []):
            cues.append(cue_from_openvino_detection(detection, seconds))
    return cues