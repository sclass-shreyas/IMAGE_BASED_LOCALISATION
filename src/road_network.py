"""Local drivable graph, metric geometry and soft road likelihoods.

Legacy GeoPackages are a geometry-only fallback: direction restrictions were
discarded by their writer. Download GraphML for authoritative OSM topology.
"""
import argparse
import hashlib
from pathlib import Path
import math
import networkx as nx
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from .osm_matching import config


class LocalFrame:
    def __init__(self, lat, lon):
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
        self.forward = Transformer.from_crs(4326, crs, always_xy=True)
        self.inverse = Transformer.from_crs(crs, 4326, always_xy=True)

    def xy(self, lat, lon):
        return np.asarray(self.forward.transform(lon, lat)).T

    def latlon(self, x, y):
        lon, lat = self.inverse.transform(x, y)
        return np.asarray([lat, lon]).T


def graph_path(center=None, radius=None):
    center, radius = center or config.AREA_CENTER, radius or config.AREA_RADIUS_M
    key = hashlib.sha256(f"{center[0]:.6f},{center[1]:.6f},{radius:.0f},drive".encode()).hexdigest()[:12]
    return config.OSM_CACHE_DIR / f"drive_{key}.graphml"


def download_graph(center=None, radius=None):
    import osmnx as ox
    center, radius = center or config.AREA_CENTER, radius or config.AREA_RADIUS_M
    ox.settings.requests_timeout = 60
    graph = ox.graph_from_point(center, dist=radius, network_type="drive", retain_all=True)
    graph.graph.update(localization_source="osm_drive_graph", area_lat=center[0], area_lon=center[1], radius_m=radius)
    path = graph_path(center, radius)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.graphml')
    ox.save_graphml(graph, temporary)
    temporary.replace(path)
    return path


def load_graph(center=None, radius=None, allow_legacy=True):
    import osmnx as ox
    center, radius = center or config.AREA_CENTER, radius or config.AREA_RADIUS_M
    path = graph_path(center, radius)
    if path.exists():
        return ox.load_graphml(path)
    if not allow_legacy or tuple(center) != tuple(config.AREA_CENTER) or radius != config.AREA_RADIUS_M:
        raise ValueError("No cached drive graph for this area. Download it from the dashboard or python -m src.road_network --download")
    import geopandas as gpd
    legacy = config.OSM_CACHE_DIR / f"{config.AREA_NAME}_roads.gpkg"
    if not legacy.exists():
        raise ValueError("No local OSM road cache. Download the drive graph first.")
    roads = gpd.read_file(legacy).to_crs(4326)
    allowed = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
               "secondary", "secondary_link", "tertiary", "tertiary_link", "residential", "service",
               "unclassified", "living_street", "road"}
    graph = nx.MultiDiGraph(crs="EPSG:4326", localization_source="legacy_geometry_bidirectional")
    for _, row in roads.iterrows():
        if not set(str(row.highway).split('|')) & allowed or row.geometry.geom_type != "LineString":
            continue
        coordinates = list(row.geometry.coords)
        u, v = tuple(coordinates[0]), tuple(coordinates[-1])
        if u == v:
            continue
        graph.add_node(u, x=u[0], y=u[1])
        graph.add_node(v, x=v[0], y=v[1])
        graph.add_edge(u, v, geometry=row.geometry, name=str(row.get('name') or ""), length=float(row.length_m))
        graph.add_edge(v, u, geometry=LineString(coordinates[::-1]), name=str(row.get('name') or ""), length=float(row.length_m))
    return graph


class RoadNetwork:
    def __init__(self, graph, origin):
        self.graph = graph
        self.frame = LocalFrame(*origin)
        self.source = graph.graph.get("localization_source", "osm_drive_graph")
        self.edges, starts, ends, edge_ids, offsets = [], [], [], [], []
        self.routing = nx.DiGraph()
        self._transition_cache = {}
        for u, v, key, data in graph.edges(keys=True, data=True):
            a, b = graph.nodes[u], graph.nodes[v]
            geometry = data.get('geometry', LineString([(a['x'], a['y']), (b['x'], b['y'])]))
            coords = np.asarray(geometry.coords)
            xy = self.frame.xy(coords[:, 1], coords[:, 0])
            ux = self.frame.xy(a['y'], a['x'])
            if np.linalg.norm(xy[-1] - ux) < np.linalg.norm(xy[0] - ux):
                xy, coords = xy[::-1], coords[::-1]
            lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            total = float(lengths.sum())
            if total < 0.1:
                continue
            eid = len(self.edges)
            self.edges.append(dict(u=u, v=v, key=key, length=total, xy=xy,
                                   coordinates=coords.tolist(), name=str(data.get('name', ''))))
            if not self.routing.has_edge(u, v) or self.routing[u][v]['weight'] > total:
                self.routing.add_edge(u, v, weight=total, edge_id=eid)
            progress = 0.0
            for p, q, length in zip(xy[:-1], xy[1:], lengths):
                count = max(1, math.ceil(length / 10))
                for j in range(count):
                    starts.append(p + (q-p)*j/count)
                    ends.append(p + (q-p)*(j+1)/count)
                    edge_ids.append(eid)
                    offsets.append(progress + length*j/count)
                progress += length
        if not self.edges:
            raise ValueError("Road graph has no usable drivable geometry")
        self.starts, self.ends = np.asarray(starts), np.asarray(ends)
        self.vectors = self.ends - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.segment_headings = np.arctan2(self.vectors[:, 1], self.vectors[:, 0])
        self.segment_edges, self.offsets = np.array(edge_ids), np.array(offsets)
        self.tree = cKDTree((self.starts + self.ends)/2)

    def match(self, xy, headings, road_sigma=7.0, heading_sigma=0.65):
        _, indices = self.tree.query(xy, k=min(24, len(self.starts)))
        indices = np.asarray(indices).reshape(len(xy), -1)
        starts, vectors = self.starts[indices], self.vectors[indices]
        fractions = np.clip(np.sum((xy[:, None, :] - starts)*vectors, axis=2) /
                            np.maximum(self.lengths[indices]**2, 1e-12), 0, 1)
        distances = np.linalg.norm(xy[:, None, :] - starts - fractions[:, :, None]*vectors, axis=2)
        angles = np.arctan2(np.sin(headings[:, None] - self.segment_headings[indices]),
                            np.cos(headings[:, None] - self.segment_headings[indices]))
        cost = 0.5*(distances/road_sigma)**2 + 0.5*(angles/heading_sigma)**2
        best = cost.argmin(axis=1)
        rows = np.arange(len(xy))
        chosen = indices[rows, best]
        return dict(loglike=-cost[rows, best], distance=distances[rows, best],
                    edge_ids=self.segment_edges[chosen],
                    progress=self.offsets[chosen] + fractions[rows, best]*self.lengths[chosen],
                    headings=self.segment_headings[chosen])

    def transition_penalty(self, previous, current, previous_progress, current_progress, travelled):
        penalties = np.ones(len(current))
        for i in np.flatnonzero((previous >= 0) & (previous != current)):
            old, new = self.edges[int(previous[i])], self.edges[int(current[i])]
            pair = (old['v'], new['u'])
            if pair not in self._transition_cache:
                # A bounded graph search avoids conflating nearby parallel roads.
                reachable = nx.single_source_dijkstra_path_length(self.routing, old['v'], cutoff=250, weight='weight')
                self._transition_cache[pair] = reachable.get(new['u'], float('inf'))
            length = old['length'] - previous_progress[i] + self._transition_cache[pair] + current_progress[i]
            if length > travelled + 25:
                penalties[i] = 0.03
        return penalties

    def geojson(self):
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"edge_id": i, "name": e['name'],
                "heading_deg": float(np.degrees(np.arctan2(*(e['xy'][-1]-e['xy'][0])[::-1])) % 360)},
             "geometry": {"type": "LineString", "coordinates": e['coordinates']}}
            for i, e in enumerate(self.edges)]}

    def route(self, start_latlon, end_latlon):
        nodes = list(self.routing.nodes)
        xy = self.frame.xy([self.graph.nodes[n]['y'] for n in nodes], [self.graph.nodes[n]['x'] for n in nodes])
        start = nodes[int(np.linalg.norm(xy-self.frame.xy(*start_latlon), axis=1).argmin())]
        end = nodes[int(np.linalg.norm(xy-self.frame.xy(*end_latlon), axis=1).argmin())]
        path = nx.shortest_path(self.routing, start, end, weight='weight')
        if len(path) < 2:
            raise ValueError("Choose a destination farther from the start")
        points = []
        for a, b in zip(path, path[1:]):
            edge = self.edges[self.routing[a][b]['edge_id']]
            points.extend(edge['xy'][:-1])
        points.append(edge['xy'][-1])
        return np.asarray(points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        print(download_graph())
    else:
        net = RoadNetwork(load_graph(), config.AREA_CENTER)
        print(f"{len(net.edges)} directed edges; source={net.source}")
