"""Check what OSM actually contains around the test area: named roads and named places/buildings."""
import osmnx as ox
import pandas as pd

from . import config

KIND_KEYS = ("highway", "building", "amenity", "leisure", "tourism", "office", "landuse", "natural", "waterway")
LIMIT = 150


def main():
    print(f"Centre {config.AREA_CENTER}, radius {config.AREA_RADIUS_M} m\n")
    feats = ox.features_from_point(config.AREA_CENTER, tags={"name": True}, dist=config.AREA_RADIUS_M)
    feats = feats[feats["name"].notna()].reset_index()

    def kind(row):
        for key in KIND_KEYS:
            if key in feats.columns and pd.notna(row.get(key)):
                return f"{key}={row[key]}"
        return "other"

    feats["kind"] = feats.apply(kind, axis=1)
    roads = feats[feats["kind"].str.startswith("highway=")]
    other = feats[~feats["kind"].str.startswith("highway=")]

    print(f"Named roads                                   : {roads['name'].nunique()} unique names")
    print(f"Other named features (buildings, places, ...) : {other['name'].nunique()} unique names")

    print("\nNamed roads:")
    for name in sorted(roads["name"].unique())[:LIMIT]:
        print(f"   {name}")

    print("\nOther named features:")
    for name, k in sorted(set(zip(other["name"], other["kind"])))[:LIMIT]:
        print(f"   {name}   [{k}]")


if __name__ == "__main__":
    main()