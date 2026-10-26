"""Central settings for the OSM matching module (Person 2)."""
from pathlib import Path

# Project root = two levels up from this file (src/osm_matching/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSM_CACHE_DIR = PROJECT_ROOT / "data" / "osm"

# ---- TEST AREA ----------------------------------------------------------
# PLACEHOLDER until the team chooses the corridor. Change only this block.
# A centre point (lat, lon) and a radius in metres.
AREA_CENTER = (12.9716, 77.5946)   # placeholder, replace with the chosen corridor
AREA_RADIUS_M = 2500               # roughly a 5 km-wide area

# Name used in the cache filename, so different areas don't overwrite each other
AREA_NAME = "placeholder_area"
# -------------------------------------------------------------------------

# Matching settings
MIN_MATCH_SCORE = 0.75   # similarity (0-1); candidates below this are dropped
MAX_CANDIDATES = 5       # max distinct street names per cue

# Geometry settings
CLUSTER_GAP_M = 25       # same-named segments closer than this belong to one street piece

# Output settings
MAX_PIECES_TOTAL = 30    # max candidates returned per cue (output reports if it was truncated)
AMBIGUITY_MARGIN = 0.05  # candidates within this score of the best count as competing