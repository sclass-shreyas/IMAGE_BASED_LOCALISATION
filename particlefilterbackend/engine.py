from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from particlefilter import ParticleFilterEngine
import json
import os

app = FastAPI(title="P4 Localization Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pf = ParticleFilterEngine()

# Robust path pointing to data/localization_stream.json from backend/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAM_FILE = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "localization_stream.json"))

current_step_index = 0

@app.get("/api/state")
def get_current_state():
    global current_step_index
    
    if not os.path.exists(STREAM_FILE):
        return {"error": "Stream feed not found"}
        
    with open(STREAM_FILE, "r") as f:
        data = json.load(f)
        steps = data.get("steps", [])
        
        if current_step_index < len(steps):
            step_data = steps[current_step_index]
            
            # 1. Process Motion (P3)
            motion = step_data.get("motion", {})
            pf.predict(motion.get("dx", 0.0), motion.get("dy", 0.0), motion.get("d_heading", 0.0))
            
            # 2. Process Observation / 25m Cluster (P1 & P2)
            obs = step_data.get("observation", {})
            if obs.get("detected", False):
                pf.update(obs.get("candidates", []))
                pf.resample()
                
            estimate = pf.get_estimate()
            
            response = {
                "step": current_step_index,
                "timestamp": step_data.get("timestamp"),
                "vehicle": estimate,
                "observation": obs
            }
            
            current_step_index += 1
            return response
        else:
            return {"status": "completed"}

@app.post("/api/reset")
def reset_engine():
    global current_step_index, pf
    current_step_index = 0
    pf = ParticleFilterEngine()
    return {"status": "reset successful"}