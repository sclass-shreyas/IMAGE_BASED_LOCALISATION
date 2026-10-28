"""Live secondary-webcam coordinator for OCR, OSM candidates, and localization."""
from datetime import datetime, timezone
from threading import Event, Lock, Thread
import time
import uuid


class LiveCameraRuntime:
    def __init__(self, localization_session):
        self.session = localization_session
        self._lock = Lock()
        self._stop = Event()
        self._thread = None
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
                observations, candidate_responses = [], []
                for detection in detections:
                    points = detection["bbox_pixels"]
                    xs, ys = [point[0] for point in points], [point[1] for point in points]
                    cue = {"timestamp_s": timestamp_s, "frame_id": frame_id,
                           "cue_type": "text_detection", "text": detection["text"],
                           "ocr_confidence": float(detection["confidence"]),
                           "detector_confidence": float(detection["detector_confidence"]),
                           "bbox_xywh": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]}
                    key = compact_key(cue["text"])
                    if cue["ocr_confidence"] < 0.45:
                        response = self._empty_response(cue, "rejected_low_ocr_confidence")
                    elif len(key) < 4 or key in GENERIC_WORDS or key in campus_generic_terms:
                        response = self._empty_response(cue, "rejected_generic_text")
                    else:
                        response = matcher.match_cue(cue, include_geometry=True)
                        candidates = response.get("candidates", [])
                        if not candidates:
                            response["cue_status"] = "no_osm_match"
                        elif candidates[0]["match_score"] < 0.82:
                            response = self._empty_response(cue, "rejected_weak_osm_match")
                        else:
                            response["cue_status"] = "street_name_candidate"
                    observations.append(cue)
                    candidate_responses.append(response)

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
            with self._lock:
                if self._stop.is_set() and self._status.get("status") != "failed":
                    self._status["status"] = "stopped"
                    if self.session.mode == "live":
                        self.session.latest["status"] = "stopped"
