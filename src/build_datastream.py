"""Build a timestamped replay stream from a real OpenVINO OCR session JSON.

Usage: python -m src.build_datastream openvino_webcam.json datastream/session.json
Motion input must share the OCR session clock. Without calibrated route movement,
the resulting stream is for integration/replay only, not accuracy evaluation.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

from src.osm_matching.candidates import CandidateGenerator
from src.pipeline_contract import PipelineEvent


def build_datastream(ocr_path, output_path, motion_path=None, session_id=None):
    source = json.loads(Path(ocr_path).read_text(encoding="utf-8"))
    session_id = session_id or str(uuid.uuid4())
    session_start = datetime.fromisoformat(source["created_utc"])
    motion_events = []
    if motion_path:
        motion_events = json.loads(Path(motion_path).read_text(encoding="utf-8")).get("frames", [])

    matcher = CandidateGenerator()
    frames = []
    for frame_id, snapshot in enumerate(source.get("frames", []), start=1):
        timestamp_s = (datetime.fromisoformat(snapshot["timestamp_utc"]) - session_start).total_seconds()
        observations, responses, events = [], [], []
        event_identity = {"schema_version": 1, "session_id": session_id,
                          "frame_id": frame_id, "timestamp_s": timestamp_s}
        for detection in snapshot.get("detections", []):
            points = detection["bbox_pixels"]
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            cue = {"timestamp_s": timestamp_s, "cue_type": "text_detection",
                   "text": detection["text"], "ocr_confidence": float(detection["confidence"]),
                   "bbox_xywh": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]}
            response = matcher.match_cue(cue, include_geometry=True)
            observations.append(cue)
            responses.append(response)
            events.extend([
                PipelineEvent(**event_identity, source="p1.ocr", payload=cue).to_dict(),
                PipelineEvent(**event_identity, source="p2.osm_candidates", payload=response).to_dict(),
            ])

        # Avoid joining unrelated session clocks. A missing/unaligned motion sample
        # is represented explicitly and has zero confidence.
        motion = None
        if motion_events:
            nearest = min(motion_events, key=lambda item: abs(float(item["timestamp_s"]) - timestamp_s))
            if abs(float(nearest["timestamp_s"]) - timestamp_s) <= 0.25:
                motion = nearest
        motion = motion or {"timestamp_s": timestamp_s, "frame_id": frame_id,
                            "dx_px": 0.0, "dy_px": 0.0, "d_heading_rad": 0.0,
                            "confidence": 0.0, "motion_type": "relative_image"}
        events.append(PipelineEvent(**event_identity, source="p3.relative_motion", payload=motion).to_dict())
        frames.append({"frame_id": frame_id, "timestamp_s": timestamp_s,
                       "observations": observations, "osm_candidates_responses": responses,
                       "observation": observations[0] if observations else {},
                       "osm_candidates_response": responses[0] if responses else {},
                       "motion": motion, "route_delta_m": None, "events": events})

    result = {"schema_version": 1, "session_id": session_id,
              "metadata": {"source": "OpenVINO OCR session + current OSM matcher",
                           "created_utc": datetime.now(timezone.utc).isoformat(),
                           "motion_translation_units": "pixels; ignored by geographic filter",
                           "route_delta_m": "not supplied; localization uses configured area prior and cues"},
              "frames": frames}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ocr_json")
    parser.add_argument("output_json")
    parser.add_argument("--motion-json")
    parser.add_argument("--session-id")
    args = parser.parse_args()
    output = build_datastream(args.ocr_json, args.output_json, args.motion_json, args.session_id)
    print(f"Wrote {len(output['frames'])} integrated frames to {args.output_json}")
