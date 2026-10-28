"""Central settings for the OSM matching module (Person 2)."""
from pathlib import Path

# Project root = two levels up from this file (src/osm_matching/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSM_CACHE_DIR = PROJECT_ROOT / "data" / "osm"

# ---- TEST AREA: RV College of Engineering, Bengaluru ---------------------
# Starting point from Wikipedia; replace with the middle of the campus from Google Maps if it differs.
AREA_CENTER = (12.923760000292681, 77.49968865500958)   # (lat, lon)
AREA_RADIUS_M = 1000               # campus plus the surrounding roads (e.g. Mysore Road)

# Name used in the cache filename, so different areas don't overwrite each other
AREA_NAME = "rvce_v1"
# -------------------------------------------------------------------------

# Matching settings
MIN_MATCH_SCORE = 0.75   # similarity (0-1); candidates below this are dropped
MAX_CANDIDATES = 5       # max distinct street names per cue

# Geometry settings
CLUSTER_GAP_M = 25       # same-named segments closer than this belong to one street piece

# Output settings
MAX_PIECES_TOTAL = 30    # max candidates returned per cue (output reports if it was truncated)
AMBIGUITY_MARGIN = 0.05  # candidates within this score of the best count as competing

# Optional aliases for signs whose wording differs from the OSM building name.
# Keys must be the exact OSM feature name; values are recognized sign variants.
FEATURE_NAME_ALIASES = {}
