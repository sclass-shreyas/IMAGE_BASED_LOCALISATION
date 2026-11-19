"""Turn a street name into separate, map-ready candidate pieces.

A street name such as '1st Cross Road' is often many unrelated streets in
different neighbourhoods. pieces(name) groups the segments that physically
connect, so each piece is one real street with its own location.
"""
from collections import defaultdict

from shapely.strtree import STRtree

from . import config


def _parse_ids(text):
    """'123|456' -> [123, 456]"""
    return [int(p) for p in str(text).split("|") if p.strip().isdigit()]


def _connected_groups(geoms, gap_m):
    """Group indices of geometries that lie within gap_m of each other (transitively)."""
    parent = list(range(len(geoms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = STRtree(geoms)
    for i, geom in enumerate(geoms):
        for j in tree.query(geom.buffer(gap_m)):
            j = int(j)
            if j > i and geom.distance(geoms[j]) <= gap_m:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups = defaultdict(list)
    for i in range(len(geoms)):
        groups[find(i)].append(i)
    return list(groups.values())


class StreetGeometry:
    """Builds candidate pieces (one per physically connected street) from the roads table."""

    def __init__(self, roads, gap_m=None):
        self.gap_m = config.CLUSTER_GAP_M if gap_m is None else gap_m
        named = roads[roads["name"].notna()].reset_index(drop=True)
        projected = named.to_crs(named.estimate_utm_crs())   # metres, so distances are real

        self._geom_ll = list(named.geometry)       # lon/lat, for output
        self._geom_m = list(projected.geometry)    # metric, for distances
        self._length = named["length_m"].astype(float).tolist()
        self._ids = named["osm_ids"].astype(str).tolist()
        self._rows_by_name = {name: list(rows) for name, rows in named.groupby("name").indices.items()}
        self._cache = {}

    def pieces(self, name):
        """Candidate pieces for an exact OSM street name, longest first. [] if unknown."""
        if name in self._cache:
            return self._cache[name]
        rows = self._rows_by_name.get(name)
        if not rows:
            return []

        groups = _connected_groups([self._geom_m[i] for i in rows], self.gap_m)
        pieces = [self._build_piece(name, [rows[k] for k in group]) for group in groups]
        pieces.sort(key=lambda p: -p["length_m"])
        for index, piece in enumerate(pieces):
            piece["piece_index"] = index

        self._cache[name] = pieces
        return pieces

    def _build_piece(self, name, rows):
        lengths = [self._length[i] for i in rows]
        longest = rows[max(range(len(rows)), key=lambda k: lengths[k])]
        way_ids = sorted({w for i in rows for w in _parse_ids(self._ids[i])})
        osm_id = _parse_ids(self._ids[longest])[0]   # OSMnx edges always carry an osmid

        # representative point: midpoint of the segment closest to the piece's centre, so it lies ON the road
        mids = [self._geom_m[i].interpolate(0.5, normalized=True) for i in rows]
        cx = sum(p.x for p in mids) / len(mids)
        cy = sum(p.y for p in mids) / len(mids)
        best = min(range(len(rows)), key=lambda k: (mids[k].x - cx) ** 2 + (mids[k].y - cy) ** 2)
        rep = self._geom_ll[rows[best]].interpolate(0.5, normalized=True)

        bounds = [self._geom_ll[i].bounds for i in rows]   # (minx, miny, maxx, maxy) = lon/lat
        return {
            "candidate_id": f"way/{osm_id}",
            "name": name,
            "osm_type": "way",
            "osm_id": osm_id,
            "osm_way_ids": way_ids,
            "length_m": round(sum(lengths), 1),
            "n_segments": len(rows),
            "representative_lat": round(rep.y, 6),
            "representative_lon": round(rep.x, 6),
            "bbox": [
                round(min(b[0] for b in bounds), 6), round(min(b[1] for b in bounds), 6),
                round(max(b[2] for b in bounds), 6), round(max(b[3] for b in bounds), 6),
            ],
            "segments": [
                [[round(y, 6), round(x, 6)] for x, y in self._geom_ll[i].coords]   # [lat, lon] pairs
                for i in rows
            ],
        }


if __name__ == "__main__":
    import sys

    from .data_loader import load_roads

    geo = StreetGeometry(load_roads())
    for street in sys.argv[1:] or ["1st Cross Road", "Mahatma Gandhi Road", "Cubbon Road"]:
        found = geo.pieces(street)
        print(f"\n{street}: {len(found)} separate piece(s)")
        for p in found[:8]:
            print(f"   #{p['piece_index']}  {p['candidate_id']:14} {p['length_m']:8.1f} m  "
                  f"{p['n_segments']:3} segments  at ({p['representative_lat']:.5f}, {p['representative_lon']:.5f})")
        if len(found) > 8:
            print(f"   ... and {len(found) - 8} more")