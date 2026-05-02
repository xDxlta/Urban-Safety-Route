from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import osmnx as ox
import geopandas as gpd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_features import (
    get_osm_context_layers,
    prepare_context_gdf,
    get_city_context_path,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
GRAPHS_DIR = BASE_DIR / "Data" / "graphs"

# --------------------------- LOAD MODEL ---------------------------
print("Loading model...")
xgb_model = joblib.load(PROCESSED_DIR / "xgb_safety_model.pkl")
feature_cols = joblib.load(PROCESSED_DIR / "xgb_feature_cols.pkl")
print(f"Feature cols: {feature_cols}")

# --------------------------- LOAD EDGE FEATURES ---------------------------
print("\nLoading Zürich edge features...")
edges_df = pd.read_csv(PROCESSED_DIR / "zurich_edge_features.csv")
print(f"Edge features shape: {edges_df.shape}")

# --------------------------- CONTEXT FEATURES ---------------------------
print("\nBuilding context features for Zürich...")
context_path = get_city_context_path("Zurich")

if context_path.exists():
    print("Loading cached context for Zürich...")
    context_df = pd.read_csv(context_path)
else:
    print("Downloading OSM context for Zürich...")
    G = ox.load_graphml(GRAPHS_DIR / "Zurich.graphml")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Build points GDF from edge midpoints
    edges_gdf = edges_gdf.to_crs(epsg=3857)
    edges_gdf = edges_gdf.reset_index()  # u, v, key aus Index in Spalten
    edges_gdf["geometry_mid"] = edges_gdf.geometry.interpolate(0.5, normalized=True)
    points_gdf = gpd.GeoDataFrame(
        edges_gdf[["u", "v", "key"]].copy(),
        geometry=edges_gdf["geometry_mid"],
        crs="EPSG:3857"
    ).reset_index(drop=True)
    points_gdf["point_id"] = points_gdf.index

    parks, stations, pois = get_osm_context_layers("Zurich")

    from context_features import prepare_context_gdf
    parks_gdf    = prepare_context_gdf(parks)
    stations_gdf = prepare_context_gdf(stations)
    pois_gdf     = prepare_context_gdf(pois)

    points_gdf["dist_to_park"]    = 9999.0
    points_gdf["dist_to_station"] = 9999.0
    points_gdf["near_park"]       = 0
    points_gdf["near_station"]    = 0
    points_gdf["poi_count_300m"]  = 0

    if not parks_gdf.empty:
        joined = gpd.sjoin_nearest(
            points_gdf[["point_id", "geometry"]],
            parks_gdf[["geometry"]], how="left", distance_col="dist_to_park"
        ).sort_values("dist_to_park").drop_duplicates("point_id")
        points_gdf["dist_to_park"] = points_gdf["point_id"].map(
            joined.set_index("point_id")["dist_to_park"]).fillna(9999.0)
        points_gdf["near_park"] = (points_gdf["dist_to_park"] <= 300).astype(int)

    if not stations_gdf.empty:
        joined = gpd.sjoin_nearest(
            points_gdf[["point_id", "geometry"]],
            stations_gdf[["geometry"]], how="left", distance_col="dist_to_station"
        ).sort_values("dist_to_station").drop_duplicates("point_id")
        points_gdf["dist_to_station"] = points_gdf["point_id"].map(
            joined.set_index("point_id")["dist_to_station"]).fillna(9999.0)
        points_gdf["near_station"] = (points_gdf["dist_to_station"] <= 300).astype(int)

    if not pois_gdf.empty:
        joined = gpd.sjoin_nearest(
            points_gdf[["point_id", "geometry"]],
            pois_gdf[["geometry"]], how="left", distance_col="poi_dist"
        )
        joined = joined[joined["poi_dist"] <= 300]
        counts = joined.groupby("point_id").size()
        points_gdf["poi_count_300m"] = points_gdf["point_id"].map(counts).fillna(0).astype(int)

    context_df = pd.DataFrame(points_gdf[[
        "u", "v", "key", "dist_to_park", "dist_to_station",
        "near_park", "near_station", "poi_count_300m"
    ]])
    context_df.to_csv(context_path, index=False)
    print(f"Saved Zürich context: {context_path}")

print(f"Context shape: {context_df.shape}")

# --------------------------- MERGE ---------------------------
print("\nMerging features...")
df = edges_df.merge(context_df, on=["u", "v", "k"] if "k" in context_df.columns else ["u", "v"],
                    how="left")

# Build derived features
df["log_dist_park"]    = np.log1p(df["dist_to_park"].fillna(9999))
df["log_dist_station"] = np.log1p(df["dist_to_station"].fillna(9999))
df["poi_density_300m"] = df["poi_count_300m"].fillna(0) / (np.pi * (300 ** 2))

# Interaction features
df["lit_and_sidewalk"]     = df["is_lit"] * df["has_sidewalk"]
df["residential_sidewalk"] = df["highway_residential"] * df["has_sidewalk"]
df["footway_lit"]          = df["highway_footway"] * df["is_lit"]
df["road_capacity"]        = df["maxspeed"] * df["lanes"]
df["smooth_and_lit"]       = df["surface_smoothness"] * df["is_lit"]
df["smooth_and_sidewalk"]  = df["surface_smoothness"] * df["has_sidewalk"]
df["busy_road"]            = df["highway_primary"] + df["highway_secondary"] + df["highway_tertiary"]
df["pedestrian_infra"]     = df["highway_footway"] + df["highway_path"] + df["has_sidewalk"]

# --------------------------- PREDICT ---------------------------
print("\nPredicting safety scores...")
X = df[feature_cols].fillna(0)
df["safety_score"] = xgb_model.predict(X)

# Normalize to 0-1
min_s = df["safety_score"].min()
max_s = df["safety_score"].max()
df["safety_score_norm"] = (df["safety_score"] - min_s) / (max_s - min_s)

print(f"\nSafety score stats:")
print(df["safety_score_norm"].describe())

# --------------------------- SAVE ---------------------------
out_cols = ["u", "v", "k", "safety_score", "safety_score_norm"]
out_path = PROCESSED_DIR / "zurich_safety_scores.csv"
df[out_cols].to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {df[out_cols].shape}")