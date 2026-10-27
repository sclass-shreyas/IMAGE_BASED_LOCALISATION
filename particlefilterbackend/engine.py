import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .particlefilter import ParticleFilterEngine

app = FastAPI(title="P4 Particle Filter Backend Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamically locate datastream/sampledata.json relative to workspace root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datastream" / "sampledata.json"

def load_datstream():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data stream file not found at {DATA_PATH}")
    with open(DATA_PATH, "r") as f:
        return json.load(f)

# Load dataset dynamically
dataset = load_datstream()
frames = dataset.get("frames", [])
total_steps = len(frames)

# Initialize filter with first frame's coordinates if available
init_lat = frames[0]["ground_truth"]["lat"] if frames else 12.9716
init_lon = frames[0]["ground_truth"]["lon"] if frames else 77.5946
pf_engine = ParticleFilterEngine(num_particles=150, initial_lat=init_lat, initial_lon=init_lon)

current_frame_index = 0

@app.get("/api/state")
def get_state():
    global current_frame_index
    
    if current_frame_index >= total_steps:
        return {"status": "completed"}
    
    frame = frames[current_frame_index]
    step_num = frame.get("step", current_frame_index + 1)
    motion = frame.get("motion", {"dx_meters": 0.0, "dy_meters": 0.0})
    observation = frame.get("observation", {"detected": False})
    
    # Run particle filter prediction and update steps
    pf_engine.predict(motion.get("dx_meters", 0.0), motion.get("dy_meters", 0.0))
    pf_engine.update(observation.get("detected", False))
    
    est_lat, est_lon, uncertainty = pf_engine.get_estimated_state()
    
    response_data = {
        "status": "running",
        "step": f"{step_num} / {total_steps}",
        "vehicle": {
            "lat": est_lat,
            "lon": est_lon,
            "uncertainty_meters": uncertainty
        },
        "observation": observation
    }
    
    # Advance to next frame for subsequent polls
    current_frame_index += 1
    return response_data

@app.post("/api/reset")
def reset_engine():
    global current_frame_index, pf_engine, frames
    dataset = load_datstream()
    frames = dataset.get("frames", [])
    current_frame_index = 0
    if frames:
        init_lat = frames[0]["ground_truth"]["lat"]
        init_lon = frames[0]["ground_truth"]["lon"]
        pf_engine = ParticleFilterEngine(num_particles=150, initial_lat=init_lat, initial_lon=init_lon)
    return {"status": "reset_successful"}