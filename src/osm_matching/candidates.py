"""Public entry point of Person 2's module: OCR cue in, geographic candidates out."""
from . import config
from .data_loader import load_roads
from .geometry import StreetGeometry
from .matcher import StreetIndex
from .normalize import normalize_text


def _candidate(piece, hit, n_pieces, include_geometry):
    candidate = {
        "candidate_id": piece["candidate_id"],
        "name": piece["name"],
        "osm_type": piece["osm_type"],
        "osm_id": piece["osm_id"],
        "match_score": hit["match_score"],
        "match_type": hit["match_type"],
        "representative_lat": piece["representative_lat"],
        "representative_lon": piece["representative_lon"],
        "same_name_pieces": n_pieces,
        "length_m": piece["length_m"],
        "osm_way_ids": piece["osm_way_ids"],
        "bbox": piece["bbox"],
    }
    if include_geometry:
        candidate["segments"] = piece["segments"]
    return candidate


def _is_ambiguous(candidates):
    """True when more than one candidate scores within AMBIGUITY_MARGIN of the best."""
    if len(candidates) < 2:
        return False
    top = candidates[0]["match_score"]
    return sum(1 for c in candidates if c["match_score"] >= top - config.AMBIGUITY_MARGIN) > 1


class CandidateGenerator:
    """Build once (loads OSM roads and indexes), then call match_cue for every cue."""

    def __init__(self, roads=None):
        roads = load_roads() if roads is None else roads
        self.index = StreetIndex(roads)
        self.geometry = StreetGeometry(roads)

    def match_cue(self, cue, include_geometry=True):
        """cue: dict with timestamp_s, cue_type, text, ocr_confidence, bbox_xywh (the brief's format).

        Returns the brief's candidate record. Never raises for bad text: it returns no candidates.
        """
        if not isinstance(cue, dict):
            raise TypeError("cue must be a dict in the Person 1 -> Person 2 contract format")
        text = str(cue.get("text") or "")

        best = {}   # candidate_id -> candidate (a way can appear under two names; keep the better score)
        for hit in self.index.match(text):
            pieces = self.geometry.pieces(hit["name"])
            for piece in pieces:
                candidate = _candidate(piece, hit, len(pieces), include_geometry)
                old = best.get(candidate["candidate_id"])
                if old is None or candidate["match_score"] > old["match_score"]:
                    best[candidate["candidate_id"]] = candidate

        ranked = sorted(best.values(), key=lambda c: (-c["match_score"], -c["length_m"]))
        kept = ranked[: config.MAX_PIECES_TOTAL]
        return {
            "timestamp_s": cue.get("timestamp_s"),
            "query_text": text,
            "normalized_text": normalize_text(text),
            "ocr_confidence": cue.get("ocr_confidence"),
            "ambiguous": _is_ambiguous(kept),
            "n_candidates_total": len(ranked),
            "truncated": len(ranked) > len(kept),
            "candidates": kept,
        }


if __name__ == "__main__":
    import sys

    from .mock_cues import MOCK_CUES, load_cues_from_openvino_json

    cues = load_cues_from_openvino_json(sys.argv[1]) if len(sys.argv) > 1 else MOCK_CUES
    generator = CandidateGenerator()
    for cue in cues:
        result = generator.match_cue(cue, include_geometry=False)
        print(f"\nt={result['timestamp_s']}s  {result['query_text']!r} -> {result['normalized_text']!r}  "
              f"ambiguous={result['ambiguous']}  candidates={result['n_candidates_total']}")
        for c in result["candidates"][:6]:
            print(f"   {c['match_score']:.2f} {c['match_type']:8} {c['candidate_id']:16} {c['name']}  ({c['length_m']} m)")
        if result["n_candidates_total"] > 6:
            print(f"   ... and {result['n_candidates_total'] - 6} more")