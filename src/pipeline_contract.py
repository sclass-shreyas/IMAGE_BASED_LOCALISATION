"""Shared, lightweight records and validation for the localization pipeline.

All event timestamps are seconds from the start of a session. Geographic
coordinates use WGS84 [latitude, longitude]; pixel coordinates are [x, y].
"""
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class PipelineEvent:
    schema_version: int
    session_id: str
    frame_id: int
    timestamp_s: float
    source: str
    payload: dict[str, Any]

    def to_dict(self):
        return {"schema_version": self.schema_version, "session_id": self.session_id,
                "frame_id": self.frame_id, "timestamp_s": self.timestamp_s,
                "source": self.source, "payload": self.payload}


def validate_candidate_response(response):
    """Validate the Person 2 candidate geometry and coordinate convention."""
    if not isinstance(response, dict) or not isinstance(response.get("candidates", []), list):
        raise ValueError("candidate response must contain a candidates list")
    for candidate in response.get("candidates", []):
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be an object")
        rep_lat, rep_lon = candidate.get("representative_lat"), candidate.get("representative_lon")
        if rep_lat is not None or rep_lon is not None:
            if (not isinstance(rep_lat, (int, float)) or not isinstance(rep_lon, (int, float))
                    or not math.isfinite(rep_lat) or not math.isfinite(rep_lon)
                    or not -90 <= rep_lat <= 90 or not -180 <= rep_lon <= 180):
                raise ValueError("representative coordinate must be a valid WGS84 [lat, lon] pair")
        for segment in candidate.get("segments", []):
            if not isinstance(segment, list):
                raise ValueError("each road segment must be a polyline")
            for point in segment:
                if (not isinstance(point, (list, tuple)) or len(point) != 2
                        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)):
                    raise ValueError("road coordinates must be finite [lat, lon] pairs")
                lat, lon = point
                if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                    raise ValueError("road coordinates are outside WGS84 ranges")
    return response
