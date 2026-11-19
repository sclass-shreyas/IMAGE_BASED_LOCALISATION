"""Write example Person 2 outputs for teammates. Run from the repo root."""
import json

from . import config
from .candidates import CandidateGenerator
from .mock_cues import MOCK_CUES

OUTPUT = config.PROJECT_ROOT / "docs" / "examples" / "person2_candidates_sample.json"

if __name__ == "__main__":
    generator = CandidateGenerator()
    results = [generator.match_cue(cue, include_geometry=False) for cue in MOCK_CUES]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} example results to {OUTPUT}")