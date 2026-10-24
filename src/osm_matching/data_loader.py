"""Download, clean and cache OSM roads for the test area."""
import geopandas as gpd
import osmnx as ox
import pandas as pd

from . import config


def _cache_path():
    return config.OSM_CACHE_DIR / f"{config.AREA_NAME}_roads.gpkg"


def _to_list(value):
    """OSM attributes can be a single value, a list, or missing. Always return a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)]


def _clean(edges):
    """Turn raw OSMnx edges into one tidy row per (road segment, name)."""
    edges = edges.reset_index(drop=True).copy()

    # osmid / highway can be lists after OSMnx merges segments -> flatten to strings
    edges["osm_ids"] = edges["osmid"].apply(lambda v: "|".join(_to_list(v)))
    edges["highway"] = edges["highway"].apply(lambda v: "|".join(_to_list(v)))

    # a segment may carry several names -> one row per name (None if unnamed)
    if "name" in edges.columns:
        names = edges["name"].apply(_to_list)
    else:
        names = pd.Series([[] for _ in range(len(edges))], index=edges.index)
    edges["name"] = names.apply(lambda lst: lst if lst else [None])
    edges = edges.explode("name")

    edges = edges.rename(columns={"length": "length_m"})
    return edges[["osm_ids", "name", "highway", "length_m", "geometry"]].reset_index(drop=True)


def load_roads(force_download=False):
    """Return a GeoDataFrame of road segments (EPSG:4326). Uses the cache if present."""
    path = _cache_path()
    if path.exists() and not force_download:
        return gpd.read_file(path)

    print(f"Downloading OSM roads around {config.AREA_CENTER}, radius {config.AREA_RADIUS_M} m ...")
    graph = ox.graph_from_point(
        config.AREA_CENTER,
        dist=config.AREA_RADIUS_M,
        network_type="all",
        simplify=True,
    )
    edges = ox.graph_to_gdfs(graph, nodes=False)
    roads = _clean(edges)

    config.OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    roads.to_file(path, driver="GPKG")
    print(f"Saved {len(roads)} rows to {path}")
    return roads


def summarize(roads):
    """Print what is actually in the data: this tells us if the area is usable."""
    named = roads[roads["name"].notna()]
    print(f"\nTotal road rows : {len(roads)}")
    print(f"Named rows      : {len(named)}  ({len(named) / max(len(roads), 1):.0%})")
    print(f"Unique names    : {named['name'].nunique()}")

    print("\nTop 20 named roads by total length (km):")
    top = named.groupby("name")["length_m"].sum().sort_values(ascending=False).head(20) / 1000
    print(top.round(2).to_string())

    print("\nRoad types (highway tag), by row count:")
    print(roads["highway"].value_counts().head(10).to_string())


if __name__ == "__main__":
    summarize(load_roads())