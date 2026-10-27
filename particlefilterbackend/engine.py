import json
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .particlefilter import ParticleFilterEngine

app = FastAPI(title="P4 Particle Filter Timestamp-Synced Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datastream" / "sampledata.json"

def load_datstream():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data stream file not found at {DATA_PATH}")
    with open(DATA_PATH, "r") as f:
        return json.load(f)

dataset = load_datstream()
frames = dataset.get("frames", [])

# Initialize filter with the first frame coordinates
init_lat = frames[0]["ground_truth"]["lat"] if frames else 12.9716
init_lon = frames[0]["ground_truth"]["lon"] if frames else 77.5946
pf_engine = ParticleFilterEngine(num_particles=150, initial_lat=init_lat, initial_lon=init_lon)

# Simulation clock tracking
simulation_start_time = time.time()
# Factor to control playback speed (1.0 = real-time delta, 2.0 = double speed, etc.)
TIME_SCALE = 1.0 

@app.get("/api/state")
def get_state():
    if not frames:
        return {"status": "completed"}
    
    # Calculate elapsed virtual simulation time based on real wall-clock time
    elapsed_real_time = (time.time() - simulation_start_time) * TIME_SCALE
    
    # Assume each frame is spaced by 2 seconds in your timestamps, or compute relative seconds
    # For a robust approach, we loop or map elapsed time to frame indices
    total_frames = len(frames)
    # Map elapsed seconds directly to frame progression (e.g., 1 frame every 2 seconds)
    target_index = int(elapsed_real_time // 2)
    
    if target_index >= total_frames:
        # Loop simulation or return completed status
        target_index = total_frames - 1
        return {
            "status": "completed",
            "timestamp": frames[-1].get("timestamp"),
            "vehicle": pf_engine.get_estimated_state() and dict(zip(["lat", "lon", "uncertainty_meters"], pf_engine.get_estimated_state())),
            "observation": frames[-1].get("observation")
        }
    
    frame = frames[target_index]
    motion = frame.get("motion", {"dx_meters": 0.0, "dy_meters": 0.0})
    observation = frame.get("observation", {"detected": False})
    
    # Run particle filter prediction and update math
    pf_engine.predict(motion.get("dx_meters", 0.0), motion.get("dy_meters", 0.0))
    pf_engine.update(observation.get("detected", False))
    
    est_lat, est_lon, uncertainty = pf_engine.get_estimated_state()
    
    return {
        "status": "running",
        "timestamp": frame.get("timestamp"),
        "frame_index": target_index + 1,
        "vehicle": {
            "lat": est_lat,
            "lon": est_lon,
            "uncertainty_meters": uncertainty
        },
        "observation": observation
    }