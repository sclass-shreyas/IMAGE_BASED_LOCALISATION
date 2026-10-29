"""Live secondary-webcam coordinator for OCR, OSM candidates, and localization."""
from datetime import datetime, timezone
from collections import deque
from threading import Event, Lock, Thread
import time
import math
import re
import uuid


MIN_LIVE_OCR_CONFIDENCE = 0.25
LOW_CONFIDENCE_OCR_THRESHOLD = 0.45
STRONG_OSM_MATCH_THRESHOLD = 0.90
NORMAL_OSM_MATCH_THRESHOLD = 0.82
LOW_CONFIDENCE_HISTORY_FRAMES = 3
LOW_CONFIDENCE_REQUIRED_HITS = 2


def _generate_phrase_cues(detections, timestamp_s, frame_id):
    """Generate every contiguous word span within each spatially aligned OCR line."""
    boxes = []
    for detection_index, detection in enumerate(detections):
        points = detection.get("bbox_pixels") or []
        if not points:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 <= x1 or y2 <= y1:
            continue
        words = [word.strip(".,;:!?()[]{}\"'") for word in str(detection.get("text", "")).split()]
        words = [word for word in words if word]
        if not words:
            continue
        boxes.append({"detection_index": detection_index, "words": words,
                      "x1": x1, "x2": x2, "y1": y1, "y2": y2,
                      "cy": (y1 + y2) / 2, "height": y2 - y1,
                      "confidence": float(detection.get("confidence", 0.0)),
                      "detector_confidence": float(detection.get("detector_confidence", 0.0))})

    # Cluster boxes by vertical center, then read each line from left to right.
    lines = []
    for box in sorted(boxes, key=lambda item: (item["cy"], item["x1"])):
        possible = [line for line in lines
                    if abs(line["cy"] - box["cy"]) <= max(10.0, 0.65 * max(line["height"], box["height"]))]
        if possible:
            line = min(possible, key=lambda item: abs(item["cy"] - box["cy"]))
            line["boxes"].append(box)
            line["cy"] = sum(item["cy"] for item in line["boxes"]) / len(line["boxes"])
            line["height"] = max(line["height"], box["height"])
        else:
            lines.append({"cy": box["cy"], "height": box["height"], "boxes": [box]})

    cues_by_key = {}
    for line in lines:
        line_boxes = sorted(line["boxes"], key=lambda item: (item["x1"], item["detection_index"]))
        runs = []
        run = []
        previous_box = None
        for box in line_boxes:
            gap = 0.0 if previous_box is None else box["x1"] - previous_box["x2"]
            max_gap = max(14.0, 1.8 * max(box["height"], previous_box["height"] if previous_box else 0.0))
            if run and gap > max_gap:
                runs.append(run)
                run = []
            run.extend({"word": word, "box": box} for word in box["words"])
            previous_box = box
        if run:
            runs.append(run)

        for run in runs:
            words = [item["word"] for item in run]
            for start in range(len(words)):
                for end in range(start + 1, len(words) + 1):
                    selected = run[start:end]
                    text = " ".join(item["word"] for item in selected)
                    key = re.sub(r"[^a-z0-9]", "", text.lower())
                    if len(key) < 2:
                        continue
                    selected_boxes = [item["box"] for item in selected]
                    confidence_values = [max(1e-6, min(1.0, item["box"]["confidence"]))
                                         for item in selected]
                    confidence = math.exp(sum(math.log(value) for value in confidence_values)
                                          / len(confidence_values))
                    x1 = min(item["x1"] for item in selected_boxes)
                    y1 = min(item["y1"] for item in selected_boxes)
                    x2 = max(item["x2"] for item in selected_boxes)
                    y2 = max(item["y2"] for item in selected_boxes)
                    cue = {"timestamp_s": float(timestamp_s), "frame_id": frame_id,
                           "cue_type": "text_detection", "text": text,
                           "ocr_confidence": confidence,
                           "detector_confidence": sum(item["box"]["detector_confidence"] for item in selected)
                           / len(selected),
                           "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                           "word_count": len(selected),
                           "source_detection_indices": sorted({item["box"]["detection_index"]
                                                                 for item in selected})}
                    old = cues_by_key.get(key)
                    if old is None or cue["ocr_confidence"] > old["ocr_confidence"]:
                        cues_by_key[key] = cue
    return list(cues_by_key.values())


class LiveCameraRuntime:
    def __init__(self, localization_session):
        self.session = localization_session
        self._lock = Lock()
        self._stop = Event()
        self._thread = None
        self._preview_lock = Lock()
        self._preview_jpeg = None
        self._status = {"status": "idle", "camera_index": 0}

    def start(self, camera_index=0, backend="dshow", device="GPU", interval_s=0.5):
        if interval_s <= 0:
            raise ValueError("interval_s must be greater than zero")
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("a live camera session is already running")
            session_id = str(uuid.uuid4())
            self._stop.clear()
            self.session.begin_live(session_id)
            self._status = {"status": "starting", "session_id": session_id,
                            "camera_index": camera_index, "backend": backend,
                            "device": device, "error": None}
            self._thread = Thread(target=self._run,
                                  args=(session_id, camera_index, backend, device, interval_s),
                                  daemon=True)
            self._thread.start()
            return dict(self._status)

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._status.get("status") not in ("idle", "stopped", "failed"):
                self._status["status"] = "stopping"
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            if self._status.get("status") == "stopping" and not (thread and thread.is_alive()):
                self._status["status"] = "stopped"
            if self.session.mode == "live" and not (thread and thread.is_alive()):
                self.session.latest["status"] = "stopped"
            return dict(self._status)

    def status(self):
        with self._lock:
            return dict(self._status)

    def preview_jpeg(self):
        """Return the latest annotated OCR frame, if a live camera is running."""
        with self._preview_lock:
            return self._preview_jpeg

    def _publish_preview(self, frame, detections, responses):
        import cv2
        import numpy as np

        preview = frame.copy()
        for detection, response in zip(detections, responses):
            points = np.asarray(detection["bbox_pixels"], dtype=np.int32)
            matched = bool(response.get("candidates")) and response.get("cue_status") in {
                "street_name_candidate", "low_confidence_strong_match",
                "low_confidence_repeated_match",
            }
            color = (40, 210, 80) if matched else (30, 170, 255)
            cv2.polylines(preview, [points], True, color, 2)
            text = f"{detection['text']} ({detection['confidence']:.2f})"
            x = max(4, int(points[:, 0].min()))
            y = max(22, int(points[:, 1].min()) - 7)
            cv2.putText(preview, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, color, 2, cv2.LINE_AA)

        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with self._preview_lock:
                self._preview_jpeg = encoded.tobytes()

    @staticmethod
    def _empty_response(cue, status):
        from src.osm_matching.normalize import normalize_text

        return {"timestamp_s": cue["timestamp_s"], "query_text": cue["text"],
                "normalized_text": normalize_text(cue["text"]), "ocr_confidence": cue["ocr_confidence"],
                "ambiguous": False, "n_candidates_total": 0, "truncated": False,
                "cue_status": status, "candidates": []}

    def _run(self, session_id, camera_index, backend, device, interval_s):
        capture = None
        try:
            # Imports are delayed so the API can start and report errors before a
            # webcam or OpenVINO device is available.
            from openvino_webcam_test import BufferlessCapture, OpenVINOCR, open_camera
            from motion_estimator import MotionEstimator
            from src.osm_matching.candidates import CandidateGenerator
            from src.osm_matching.matcher import GENERIC_WORDS
            from src.osm_matching.normalize import compact_key

            ocr = OpenVINOCR(device)
            requested_indices = [int(camera_index)]
            camera = None
            selected_index = None
            for index in requested_indices:
                camera = open_camera(index, 1280, 720, backend)
                if camera is not None:
                    selected_index = index
                    break
            if camera is None:
                raise RuntimeError(f"Could not open any camera at indices {requested_indices} via {backend}")
            capture = BufferlessCapture(camera)
            matcher = CandidateGenerator()
            estimator = MotionEstimator()
            started = time.perf_counter()
            last_ocr = -interval_s
            frame_id = 0
            recent_match_frames = deque(maxlen=LOW_CONFIDENCE_HISTORY_FRAMES)
            active_low_confidence_matches = set()
            campus_generic_terms = {"building", "department", "block", "hostel", "laboratory",
                                    "lab", "center", "centre", "engineering"}
            with self._lock:
                self._status.update(status="running", camera_index=selected_index,
                                    started_utc=datetime.now(timezone.utc).isoformat())

            while not self._stop.is_set():
                frame = capture.read(timeout=0.5)
                if frame is None:
                    continue
                frame_id += 1
                timestamp_s = time.perf_counter() - started
                motion = estimator.process_frame(frame, timestamp_s, frame_id)
                if timestamp_s - last_ocr < interval_s:
                    continue
                last_ocr = timestamp_s
                detections = ocr.recognize(frame)
                phrase_cues = _generate_phrase_cues(detections, timestamp_s, frame_id)
                records = []
                current_match_keys = set()
                next_active_low_confidence_matches = set()
                for cue in phrase_cues:
                    score = 0.0
                    key = compact_key(cue["text"])
                    if len(key) < 4 or key in GENERIC_WORDS or key in campus_generic_terms:
                        response = self._empty_response(cue, "rejected_generic_text")
                    elif cue["ocr_confidence"] < MIN_LIVE_OCR_CONFIDENCE:
                        response = self._empty_response(cue, "rejected_low_ocr_confidence")
                    else:
                        response = matcher.match_cue(cue, include_geometry=True)
                        candidates = response.get("candidates", [])
                        score = float(candidates[0].get("match_score", 0.0)) if candidates else 0.0
                        if not candidates:
                            response["cue_status"] = "no_osm_match"
                        else:
                            top = candidates[0]
                            kind = str(top.get("candidate_kind", ""))
                            name_key = compact_key(top.get("name", ""))
                            # Road pieces with the same name count as one cue across
                            # frames; buildings are grouped by OSM feature identity.
                            match_key = (str(top.get("candidate_id", "")) if kind == "building"
                                         else f"{kind}:{name_key}")
                            current_match_keys.add(match_key)
                            if cue["ocr_confidence"] >= LOW_CONFIDENCE_OCR_THRESHOLD:
                                if score < NORMAL_OSM_MATCH_THRESHOLD:
                                    response = self._empty_response(cue, "rejected_weak_osm_match")
                                else:
                                    response["cue_status"] = "street_name_candidate"
                            elif (match_key in active_low_confidence_matches
                                  or match_key in next_active_low_confidence_matches):
                                # One persistent low-confidence sign is one update,
                                # not a fresh likelihood multiplication every half-second.
                                response = self._empty_response(cue, "duplicate_low_confidence_cue")
                                next_active_low_confidence_matches.add(match_key)
                            else:
                                recent_hits = sum(match_key in previous for previous in recent_match_frames)
                                strong_and_unambiguous = (
                                    score >= STRONG_OSM_MATCH_THRESHOLD and not response.get("ambiguous", False)
                                )
                                if strong_and_unambiguous:
                                    response["cue_status"] = "low_confidence_strong_match"
                                    next_active_low_confidence_matches.add(match_key)
                                elif recent_hits + 1 >= LOW_CONFIDENCE_REQUIRED_HITS:
                                    response["cue_status"] = "low_confidence_repeated_match"
                                    next_active_low_confidence_matches.add(match_key)
                                else:
                                    response = self._empty_response(cue, "awaiting_temporal_confirmation")
                    records.append({"cue": cue, "response": response, "match_score": score})

                recent_match_frames.append(current_match_keys)
                active_low_confidence_matches = next_active_low_confidence_matches
                accepted_statuses = {"street_name_candidate", "low_confidence_strong_match",
                                     "low_confidence_repeated_match"}
                accepted = [item for item in records
                            if item["response"].get("cue_status") in accepted_statuses
                            and item["response"].get("candidates")]
                # Keep one phrase per independently matched OSM feature. This
                # avoids counting overlapping spans such as "MM", "Foods", and
                # "MM Foods" as three observations, while preserving a separate
                # cue for a nearby Library feature from the same frame.
                accepted.sort(key=lambda item: (
                    item["match_score"], item["cue"].get("ocr_confidence", 0.0),
                    item["cue"].get("word_count", 1)), reverse=True)
                distinct_accepted = []
                seen_feature_ids = set()
                for item in accepted:
                    top = item["response"]["candidates"][0]
                    feature_id = str(top.get("candidate_id") or
                                     f"{top.get('candidate_kind')}:{compact_key(top.get('name', ''))}")
                    if feature_id in seen_feature_ids:
                        continue
                    seen_feature_ids.add(feature_id)
                    distinct_accepted.append(item)
                accepted = distinct_accepted
                status_priority = {"awaiting_temporal_confirmation": 5,
                                   "duplicate_low_confidence_cue": 4,
                                   "rejected_weak_osm_match": 3,
                                   "no_osm_match": 2,
                                   "rejected_low_ocr_confidence": 1,
                                   "rejected_generic_text": 0}
                if accepted:
                    selected = accepted
                elif records:
                    selected = [max(records, key=lambda item: (
                        status_priority.get(item["response"].get("cue_status"), 0),
                        item["match_score"], item["cue"].get("ocr_confidence", 0.0),
                        item["cue"].get("word_count", 1)))]
                else:
                    selected = []
                observations = [item["cue"] for item in selected]
                candidate_responses = [item["response"] for item in selected]

                detection_matches = {}
                for item in selected:
                    if not item["response"].get("candidates"):
                        continue
                    for index in item["cue"].get("source_detection_indices", []):
                        detection_matches[index] = {
                            "candidates": item["response"].get("candidates", []),
                            "cue_status": item["response"].get("cue_status"),
                        }
                preview_responses = [
                    detection_matches.get(index, {"candidates": []})
                    for index, _ in enumerate(detections)
                ]
                # Keep the camera preview faithful to the OCR engine's original
                # detections; phrase combinations are used only for OSM queries.
                self._publish_preview(frame, detections, preview_responses)

                frame_packet = {"frame_id": frame_id, "timestamp_s": timestamp_s,
                                "observations": observations,
                                "osm_candidates_responses": candidate_responses,
                                "motion": motion, "route_delta_m": None}
                latest = self.session.ingest_live(frame_packet)
                with self._lock:
                    self._status.update(status="running", frame_id=frame_id,
                                        timestamp_s=timestamp_s, detections=len(detections),
                                        localization_status=latest["observation"]["status"])
        except Exception as exc:
            with self._lock:
                self._status.update(status="failed", error=str(exc))
            with self.session.lock:
                self.session.latest["status"] = "failed"
                self.session.latest["error"] = str(exc)
        finally:
            if capture is not None:
                capture.release()
            with self._preview_lock:
                self._preview_jpeg = None
            with self._lock:
                if self._stop.is_set() and self._status.get("status") != "failed":
                    self._status["status"] = "stopped"
                    if self.session.mode == "live":
                        self.session.latest["status"] = "stopped"
