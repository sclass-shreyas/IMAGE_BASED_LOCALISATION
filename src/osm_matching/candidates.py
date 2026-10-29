"""Public entry point of Person 2's module: OCR cue in, geographic candidates out."""
import geopandas as gpd

from . import config
from .data_loader import load_named_buildings, load_roads
from .geometry import StreetGeometry
from .matcher import StreetIndex
from .normalize import normalize_text


def _candidate(piece, hit, n_pieces, include_geometry):
    candidate = {
        "candidate_id": piece["candidate_id"],
        "candidate_kind": "road",
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


def _building_candidate(feature, hit, same_name_count, include_geometry):
    """Serialize a named OSM building footprint using contract [lat, lon] lines."""
    geom = feature["geometry"]
    point = geom.representative_point()
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    coords = []
    polygons = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for polygon in polygons:
        coords.append([[round(lat, 6), round(lon, 6)] for lon, lat in polygon.exterior.coords])
    osm_type = str(feature.get("osm_type", "way"))
    osm_id = str(feature.get("osm_id", ""))
    try:
        numeric_id = int(osm_id)
    except (TypeError, ValueError):
        numeric_id = osm_id
    candidate = {
        "candidate_id": f"{osm_type}/{osm_id}",
        "candidate_kind": "building",
        "name": feature["name"],
        "matched_as": hit["name"],
        "osm_type": osm_type,
        "osm_id": numeric_id,
        "building_type": str(feature.get("building_type", "yes")),
        "match_score": hit["match_score"],
        "match_type": hit["match_type"],
        "representative_lat": round(point.y, 6),
        "representative_lon": round(point.x, 6),
        "same_name_pieces": same_name_count,
        "length_m": round(float(feature.get("length_m", 0.0)), 1),
        "bbox": [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)],
    }
    if include_geometry:
        candidate["segments"] = coords
    return candidate


def _is_ambiguous(candidates):
    """True when more than one candidate scores within AMBIGUITY_MARGIN of the best."""
    if len(candidates) < 2:
        return False
    top = candidates[0]["match_score"]
    return sum(1 for c in candidates if c["match_score"] >= top - config.AMBIGUITY_MARGIN) > 1


class CandidateGenerator:
    """Build once (loads OSM roads and indexes), then call match_cue for every cue."""

    def __init__(self, roads=None, buildings=None):
        load_area_data = roads is None
        roads = load_roads() if load_area_data else roads
        buildings = (load_named_buildings() if load_area_data and buildings is None else buildings)
        self.index = StreetIndex(roads)
        self.geometry = StreetGeometry(roads)
        self.buildings = buildings
        if buildings is None or buildings.empty:
            self.building_index = None
            self._buildings_by_name = {}
        else:
            feature_rows = []
            self._buildings_by_name = {}
            for name, group in buildings.groupby("name"):
                records = [row for _, row in group.iterrows()]
                self._buildings_by_name[name] = records
                aliases = list(config.FEATURE_NAME_ALIASES.get(name, []))
                # Signs commonly append "Building" to the shorter OSM label.
                if not str(name).lower().endswith(" building"):
                    aliases.append(f"{name} Building")
                aliases = list(dict.fromkeys(aliases))
                for alias in aliases:
                    self._buildings_by_name[alias] = records
                for alias in [name, *aliases]:
                    for feature in records:
                        feature_rows.append({"name": alias, "length_m": feature["length_m"],
                                             "geometry": feature.geometry})
            index_data = gpd.GeoDataFrame(feature_rows, geometry="geometry", crs=buildings.crs)
            self.building_index = StreetIndex(index_data, include_token_spans=True)

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

        if self.building_index is not None:
            for hit in self.building_index.match(text):
                features = self._buildings_by_name.get(hit["name"], [])
                for feature in features:
                    candidate = _building_candidate(feature, hit, len(features), include_geometry)
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
