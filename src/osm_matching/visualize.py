"""Draw candidate roads on a Folium map so matches can be checked by eye.

The background is drawn from the OSM road data we already downloaded, because the public
tile servers either refuse pages opened from a local file (OpenStreetMap) or need an API
key (CARTO). OSM's own tiles are available as an optional layer when the page is served
over http.
"""
import html
from pathlib import Path

import folium

from . import config

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
          "#008080", "#9a6324", "#800000", "#808000", "#000075"]
DEFAULT_OUTPUT = config.PROJECT_ROOT / "results" / "candidate_map.html"   # results/ is git-ignored

BACKGROUND_EXCLUDE = "footway|path|steps|cycleway|pedestrian"   # keep the background readable
PAD_DEG = 0.004          # about 400 m of background around the candidates
SIMPLIFY_DEG = 0.00002   # about 2 m; keeps the HTML file small
PAGE_STYLE = "<style>.leaflet-container { background: #fafafa; }</style>"


def _add_background(fmap, roads, bounds):
    """Draw the OSM roads near the candidates as a thin grey layer."""
    min_lon, min_lat, max_lon, max_lat = bounds
    nearby = roads.cx[min_lon - PAD_DEG: max_lon + PAD_DEG, min_lat - PAD_DEG: max_lat + PAD_DEG]
    if "highway" in nearby.columns:
        nearby = nearby[~nearby["highway"].astype(str).str.contains(BACKGROUND_EXCLUDE)]
    if nearby.empty:
        return
    shapes = nearby[["geometry"]].copy()
    shapes["geometry"] = shapes.geometry.simplify(SIMPLIFY_DEG)
    folium.GeoJson(
        shapes,
        name="OSM road network (background)",
        style_function=lambda _feature: {"color": "#8a9099", "weight": 1.5, "opacity": 0.8},
    ).add_to(fmap)


def make_map(result, output_path=DEFAULT_OUTPUT, roads=None):
    """Write an HTML map of a match_cue() result and return its path.

    roads: the roads GeoDataFrame from load_roads(); if given, it is drawn as the background.
    """
    candidates = result["candidates"]

    if candidates:
        bounds = (
            min(c["bbox"][0] for c in candidates), min(c["bbox"][1] for c in candidates),
            max(c["bbox"][2] for c in candidates), max(c["bbox"][3] for c in candidates),
        )
        min_lon, min_lat, max_lon, max_lat = bounds
        fmap = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2],
                          zoom_start=15, tiles=None)
        fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        if roads is not None:
            _add_background(fmap, roads, bounds)
    else:
        fmap = folium.Map(location=list(config.AREA_CENTER), zoom_start=14, tiles=None)

    fmap.get_root().html.add_child(folium.Element(PAGE_STYLE))
    # off by default; works only when the page is served over http (python -m http.server)
    folium.TileLayer("OpenStreetMap", name="OSM tiles (only when served over http)",
                     overlay=True, show=False).add_to(fmap)

    for rank, c in enumerate(candidates):
        color = COLORS[rank % len(COLORS)]
        label = f"{rank + 1}. {c['name']} ({c['match_score']:.2f})"
        popup = folium.Popup(
            f"<b>{html.escape(c['name'])}</b><br>"
            f"{html.escape(c['candidate_id'])}<br>"
            f"score {c['match_score']:.2f} ({html.escape(c['match_type'])})<br>"
            f"{c['length_m']} m, {c['same_name_pieces']} piece(s) share this name",
            max_width=300,
        )
        group = folium.FeatureGroup(name=label)
        for segment in c.get("segments", []):
            folium.PolyLine(segment, color=color, weight=5, opacity=0.9, tooltip=label).add_to(group)
        folium.CircleMarker(
            [c["representative_lat"], c["representative_lon"]],
            radius=7, color=color, fill=True, fill_opacity=0.9, popup=popup, tooltip=label,
        ).add_to(group)
        group.add_to(fmap)

    folium.LayerControl().add_to(fmap)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    return output_path


if __name__ == "__main__":
    import sys
    import webbrowser

    from .candidates import CandidateGenerator
    from .data_loader import load_roads

    text = " ".join(sys.argv[1:]) or "MG ROAD"
    cue = {"timestamp_s": 0.0, "cue_type": "street_sign", "text": text, "ocr_confidence": 1.0, "bbox_xywh": [0, 0, 0, 0]}
    roads = load_roads()
    result = CandidateGenerator(roads).match_cue(cue)
    path = make_map(result, roads=roads)
    print(f"{result['n_candidates_total']} candidate(s) for {text!r}; map written to {path}")
    webbrowser.open(path.resolve().as_uri())