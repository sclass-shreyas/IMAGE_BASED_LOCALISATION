import json
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .particlefilter import ParticleFilterEngine

app = FastAPI(title="P4 Integrated Telemetry Localization Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datastream" / "sampledata.json"

def load_datastream():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data stream file not found at {DATA_PATH}")
    with open(DATA_PATH, "r") as f:
        return json.load(f)

dataset = load_datastream()
frames = dataset.get("frames", [])

init_lat = frames[0]["osm_candidates_response"]["candidates"][0]["representative_lat"] if frames else 12.97831
init_lon = frames[0]["osm_candidates_response"]["candidates"][0]["representative_lon"] if frames else 77.60517
pf_engine = ParticleFilterEngine(num_particles=300, initial_lat=init_lat, initial_lon=init_lon)

simulation_start_time = time.time()
TIME_SCALE = 1.0 

@app.get("/api/state")
def get_state():
    if not frames:
        return {"status": "completed"}
    
    elapsed_real_time = (time.time() - simulation_start_time) * TIME_SCALE
    total_frames = len(frames)
    target_index = int(elapsed_real_time // 2)
    
    if target_index >= total_frames:
        target_index = total_frames - 1
        est_lat, est_lon, uncertainty = pf_engine.get_estimated_state()
        return {
            "status": "completed",
            "timestamp": frames[-1].get("timestamp"),
            "vehicle": {"lat": est_lat, "lon": est_lon, "uncertainty_meters": uncertainty},
            "observation": frames[-1].get("observation")
        }
    
    frame = frames[target_index]
    
    # Ingest P1, P2, and P3 outputs
    p1_observation = frame.get("observation", {})
    p2_osm_response = frame.get("osm_candidates_response", {})
    
    # P3 Motion format matching json.dumps output contract
    p3_motion = frame.get("motion", {
        "timestamp": round(time.time(), 2),
        "dx": 0.0,
        "dy": 0.0,
        "d_heading": 0.0,
        "confidence": 0.0
    })

    # Run particle filter prediction (P3) and update (P1 + P2)
    pf_engine.predict(p3_motion)
    pf_engine.update(p1_observation, p2_osm_response)
    
    est_lat, est_lon, uncertainty = pf_engine.get_estimated_state()
    
    cue_label = p2_osm_response.get("normalized_text", p1_observation.get("text", "None"))
    candidates_count = len(p2_osm_response.get("candidates", []))
    
    return {
        "status": "running",
        "timestamp": frame.get("timestamp"),
        "frame_index": target_index + 1,
        "vehicle": {
            "lat": est_lat,
            "lon": est_lon,
            "uncertainty_meters": uncertainty
        },
        "observation": {
            "detected": bool(p2_osm_response.get("candidates")),
            "cue_text": cue_label
        },
        "active_candidates_count": candidates_count
    }