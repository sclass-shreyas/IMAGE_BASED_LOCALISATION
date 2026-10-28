"""FastAPI adapter for a timestamp-ordered localization replay session."""
import json
import os
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .particlefilter import ParticleFilterEngine

app = FastAPI(title="Integrated Telemetry Localization Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.environ["LOCALIZATION_DATASTREAM"]) if os.environ.get("LOCALIZATION_DATASTREAM") else None


def load_datastream(path=None):
    path = path or DATA_PATH
    if path is None:
        raise ValueError("Set LOCALIZATION_DATASTREAM to an explicitly selected real session stream")
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("sequence_metadata", {}).get("demo_only") or data.get("metadata", {}).get("demo_only"):
        raise ValueError("Synthetic/demo streams are disabled for localization replay")
    frames = data.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("datastream frames must be a list")
    return data


class LocalizationSession:
    """Normalize legacy frames, process each timestamp once, expose latest state."""
    def __init__(self, dataset=None, seed=0):
        self.lock = RLock()
        self.dataset = dataset or {"session_id": "unstarted", "frames": []}
        self.frames = sorted(self.dataset.get("frames", []), key=self._timestamp)
        self.filter = ParticleFilterEngine(seed=seed)
        self.cursor = 0
        self.mode = "idle"
        self.has_location_update = False
        self.latest = {"status": "ready", "session_id": self.dataset.get("session_id", "demo"),
                       "frame_index": 0, "timestamp_s": None, "vehicle": None,
                       "observation": {"status": "waiting", "cue_text": None, "candidate_count": 0}}

    @staticmethod
    def _timestamp(frame):
        value = frame.get("timestamp_s")
        if value is None:
            value = frame.get("osm_candidates_response", {}).get("timestamp_s")
        if value is None:
            raise ValueError("each frame needs timestamp_s (seconds since session start)")
        return float(value)

    def reset(self, seed=0):
        with self.lock:
            self.filter.reset(seed)
            self.cursor = 0
            self.mode = "idle"
            self.has_location_update = False
            self.latest = {"status": "ready", "session_id": self.dataset.get("session_id", "demo"),
                           "frame_index": 0, "timestamp_s": None, "vehicle": None,
                           "observation": {"status": "waiting", "cue_text": None, "candidate_count": 0}}
            return self.latest

    def begin_live(self, session_id):
        with self.lock:
            self.filter.reset()
            self.dataset = {"session_id": session_id, "frames": []}
            self.frames = []
            self.cursor = 0
            self.mode = "live"
            self.has_location_update = False
            self.latest = {"status": "starting", "mode": "live", "session_id": session_id,
                           "frame_index": 0, "timestamp_s": None, "vehicle": None,
                           "observation": {"status": "waiting", "cue_text": None, "candidate_count": 0}}
            return self.latest

    def begin_replay(self, dataset):
        with self.lock:
            self.filter.reset()
            self.dataset = dataset
            self.frames = sorted(dataset.get("frames", []), key=self._timestamp)
            self.cursor = 0
            self.mode = "replay"
            self.has_location_update = False
            self.latest = {"status": "ready", "mode": "replay",
                           "session_id": dataset.get("session_id", "replay"),
                           "frame_index": 0, "timestamp_s": None, "vehicle": None,
                           "observation": {"status": "waiting", "cue_text": None, "candidate_count": 0}}
            return self.latest

    def ingest_live(self, frame):
        with self.lock:
            if self.mode != "live":
                raise ValueError("live session has not been started")
            self.frames.append(frame)
            return self.advance()

    def advance(self):
        with self.lock:
            if self.mode not in ("replay", "live"):
                raise ValueError("start a live session or replay before advancing")
            if self.cursor >= len(self.frames):
                self.latest["status"] = "completed"
                return self.latest
            frame = self.frames[self.cursor]
            timestamp_s = self._timestamp(frame)
            if self.filter.last_timestamp_s is not None and timestamp_s < self.filter.last_timestamp_s:
                raise ValueError("frames must be processed in non-decreasing timestamp order")
            dt = 0.0 if self.filter.last_timestamp_s is None else timestamp_s - self.filter.last_timestamp_s
            motion = frame.get("motion", {})
            route_delta_m = frame.get("route_delta_m")
            self.filter.predict(motion, dt=dt, route_delta_m=route_delta_m)
            observations = frame.get("observations")
            candidates_list = frame.get("osm_candidates_responses")
            if observations is None:
                observations = [frame.get("observation", {})]
                candidates_list = [frame.get("osm_candidates_response", {})]
            if len(observations) != len(candidates_list or []):
                raise ValueError("each OCR observation must have exactly one OSM candidate response")
            diagnostics = [self.filter.update(observation, candidates)
                           for observation, candidates in zip(observations, candidates_list or [])]
            if not diagnostics:
                diagnostics = [self.filter.update({}, {})]
            if any(item["status"] in ("matched", "ambiguous_used") for item in diagnostics):
                self.has_location_update = True
            self.filter.last_timestamp_s = timestamp_s
            lat, lon, uncertainty = self.filter.get_estimated_state()
            status = next((item["status"] for item in reversed(diagnostics)
                           if item["status"] != "no_candidates"), "no_candidates")
            cue_texts = [candidate.get("normalized_text") or observation.get("text")
                         for observation, candidate in zip(observations, candidates_list or [])]
            candidate_count = sum(len(candidate.get("candidates", [])) for candidate in candidates_list or [])
            matched_candidates = []
            seen_candidate_ids = set()
            for candidate_response in candidates_list or []:
                for candidate in candidate_response.get("candidates", []):
                    if candidate.get("candidate_id") not in seen_candidate_ids:
                        seen_candidate_ids.add(candidate.get("candidate_id"))
                        matched_candidates.append({key: candidate.get(key) for key in
                                                   ("candidate_id", "name", "candidate_kind", "match_score")})
            cue_statuses = [candidate.get("cue_status") or item["status"]
                            for candidate, item in zip(candidates_list or [], diagnostics)]
            if candidate_count == 0 and cue_statuses:
                status = cue_statuses[-1]
            self.latest = {
                "status": "running",
                "mode": self.mode,
                "session_id": self.dataset.get("session_id", "demo"),
                "frame_index": self.cursor + 1,
                "timestamp_s": timestamp_s,
                "timestamp": frame.get("timestamp"),
                "vehicle": ({"lat": lat, "lon": lon, "uncertainty_meters": uncertainty}
                            if self.has_location_update else None),
                "observation": {"status": status, "detected": candidate_count > 0,
                                "cue_text": ", ".join(filter(None, cue_texts)) or None,
                                "candidate_count": candidate_count,
                                "matched_candidates": matched_candidates,
                                "cue_statuses": cue_statuses,
                                "ambiguous": any(candidate.get("ambiguous", False)
                                                 for candidate in candidates_list or [])},
                "diagnostics": {"cue_updates": diagnostics},
            }
            self.cursor += 1
            return self.latest

    def state(self):
        with self.lock:
            return self.latest


session = LocalizationSession()


@app.get("/api/state")
def get_state():
    state = dict(session.state())
    if session.mode == "live":
        camera = live_runtime.status()
        state["camera_status"] = camera.get("status")
        state["camera_index"] = camera.get("camera_index")
    return state


@app.post("/api/advance")
def advance():
    try:
        if session.mode != "replay":
            raise HTTPException(status_code=409, detail="/api/advance is available for replay sessions only")
        return session.advance()
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/reset")
def reset(seed: int = 0):
    live_runtime.stop()
    return session.reset(seed)


from .live_runtime import LiveCameraRuntime

live_runtime = LiveCameraRuntime(session)


@app.post("/api/live/start")
def start_live(camera_index: int = 0, backend: str = "dshow", device: str = "GPU",
               interval_s: float = 0.5):
    if not 0 <= camera_index <= 9:
        raise HTTPException(status_code=422, detail="camera_index must be an integer from 0 to 9")
    if backend not in ("dshow", "msmf", "any") or device not in ("GPU", "CPU"):
        raise HTTPException(status_code=422, detail="Unsupported camera backend or OpenVINO device")
    if not 0.05 <= interval_s <= 10.0:
        raise HTTPException(status_code=422, detail="interval_s must be between 0.05 and 10 seconds")
    try:
        return live_runtime.start(camera_index=camera_index, backend=backend,
                                  device=device, interval_s=interval_s)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/live/stop")
def stop_live():
    return live_runtime.stop()


@app.get("/api/live/status")
def live_status():
    return live_runtime.status()


@app.get("/api/live/preview.jpg")
def live_preview():
    frame = live_runtime.preview_jpeg()
    if frame is None:
        raise HTTPException(status_code=503, detail="No live camera preview is available")
    return Response(content=frame, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.post("/api/replay/start")
def start_replay():
    try:
        live_runtime.stop()
        return session.begin_replay(load_datastream())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# Serve the UI from the same localhost origin as the API. This gives browser
# map tile requests a normal HTTP Referer instead of a file:// origin.
app.mount("/", StaticFiles(directory=BASE_DIR / "particlefilterfrontend", html=True), name="frontend")
