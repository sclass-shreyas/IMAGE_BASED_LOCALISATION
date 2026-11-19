"""Explain why a street name splits into several pieces: how far apart they are and what lies between."""
import sys
from itertools import combinations

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import nearest_points

from .data_loader import load_roads
from .geometry import StreetGeometry


def _piece_line(piece):
    # segments are [lat, lon] pairs; shapely wants (x=lon, y=lat)
    return MultiLineString([[(lon, lat) for lat, lon in segment] for segment in piece["segments"]])


def gap_report(name, roads=None, max_pairs=6):
    roads = load_roads() if roads is None else roads
    pieces = StreetGeometry(roads).pieces(name)
    print(f"{name}: {len(pieces)} piece(s)")
    if len(pieces) < 2:
        return

    metric_crs = roads.estimate_utm_crs()
    lines = list(gpd.GeoSeries([_piece_line(p) for p in pieces], crs="EPSG:4326").to_crs(metric_crs))
    others = roads[roads["name"].ne(name)].to_crs(metric_crs)   # every road with a different (or no) name

    pairs = sorted((lines[i].distance(lines[j]), i, j) for i, j in combinations(range(len(lines)), 2))
    for distance, i, j in pairs[:max_pairs]:
        a, b = nearest_points(lines[i], lines[j])
        corridor = LineString([a, b]).buffer(30)
        between = others[others.intersects(corridor)]
        labels = sorted({f"{n or '(unnamed)'} [{h}]" for n, h in zip(between["name"], between["highway"])})
        print(f"\n  piece #{i} <-> piece #{j}: nearest points {distance:.0f} m apart")
        print("    roads crossing the gap: " + (", ".join(labels[:8]) if labels else "none"))


if __name__ == "__main__":
    gap_report(" ".join(sys.argv[1:]) or "Palace Road")