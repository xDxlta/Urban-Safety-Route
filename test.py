import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
import pandas as pd
from pathlib import Path
from pyproj import Transformer
import requests
from datetime import datetime, timezone

#API to determine whether it is day or night, ChatGPT helped finding an API
@st.cache_data(ttl=3600)
def is_night ():

    lat = 47.3779
    lng = 8.5402


    url = "https://api.sunrise-sunset.org/json?lat=" + str(lat) + "&lng=" + str(lng) + "&formatted=0"
    data = requests.get(url).json()
    sunrise = datetime.fromisoformat(data["results"]["sunrise"].replace("Z", "+00:00"))
    sunset = datetime.fromisoformat(data["results"]["sunset"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)

    if sunrise <= now <= sunset:
        return False   # day
    else:
        return True    # night

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"

#loading the data of the lamps. Raw dataset, cleanup neccessary
def load_lamps():
    df = pd.read_csv(
        PROCESSED_DIR / "ewz.ewz_brennstelle_p.csv",
        encoding="latin1",
        sep=None,
        engine="python",
        on_bad_lines="skip"
    )

    # only keep the necessary columns
    df = df[["geometry"]].copy()

    # delete the missing values
    df = df.dropna(subset=["geometry"])

    # preprocessing--> extract coordinates, clean geometry string
    coords = df["geometry"].str.replace("x = ", "")
    coords = coords.str.replace(" = y", "")
    coords = coords.str.strip().str.split(" ", expand=True)

    df["x"] = pd.to_numeric(coords[0], errors="coerce")
    df["y"] = pd.to_numeric(coords[1], errors="coerce")

    # delete invalid values
    df = df.dropna(subset=["x", "y"])
    
    # convert the coordinates correctly into global format--> The suggestion how to convert the coordinates is from ChatGPT
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    lon, lat = transformer.transform(df["x"].values, df["y"].values)
    df["lon"] = lon
    df["lat"] = lat

    return df
 
 #Load the graph and risk
@st.cache_resource
def load_graph_with_scores():
    print("Loading graph...")
    G = ox.load_graphml(BASE_DIR / "Data" / "graphs" / "Zurich.graphml")

    #is it currently night in zurich?
    night = is_night()
    lamps = load_lamps()

    scores_df = pd.read_csv(PROCESSED_DIR / "zurich_safety_scores.csv")
    score_map = {
        (row["u"], row["v"], row["k"]): row["safety_score_norm"]
        for _, row in scores_df.iterrows()
    }



    # mapping
    edges, dists = ox.distance.nearest_edges(
        G,
        lamps["lon"],
        lamps["lat"],
        return_dist=True
    )   
    
    #only keep the edges which are closer than 50m (Radius of 50m)
    lamp_edge_set = set()
    for edge, dist in zip(edges, dists):
        if dist < 50:
            lamp_edge_set.add(edge)
    

    for u, v, k, data in G.edges(keys=True, data=True):
        score = score_map.get((u, v, k), 0.5)
        data["safety_score"] = score

        risk = 1.0 - score

        # is it necessary to adjust the safety score?
        lamp_near = (u, v, k) in lamp_edge_set

        if night:
            if lamp_near:
                risk -= 0.2 #safer
            else:
                risk += 0.2 #more danger

        risk = max(0, min(1, risk)) #the risk cannot be negative

        data["risk"] = risk

    return G
 
 
G = load_graph_with_scores()
 
 
def safe_weight(u, v, data):
    edge_data = min(data.values(), key=lambda x: x.get("length", float("inf")))
    length = edge_data.get("length", 0)
    risk = edge_data.get("risk", 0.5)
    return length * (1 + 10.0 * risk)
 
 
def get_routes(start, end):
    orig = ox.distance.nearest_nodes(G, start[1], start[0])
    dest = ox.distance.nearest_nodes(G, end[1], end[0])
    route_normal = nx.shortest_path(G, orig, dest, weight="length")
    route_safe = nx.shortest_path(G, orig, dest, weight=safe_weight)
    route_normal_gdf = ox.routing.route_to_gdf(G, route_normal)
    route_safe_gdf = ox.routing.route_to_gdf(G, route_safe)
    return route_normal_gdf, route_safe_gdf
 
 
# --------------------------- UI ---------------------------
 
st.title("Safety Routing Zürich")
st.caption("Blau = kürzeste Route | Grün = sicherste Route")
 
if "points" not in st.session_state:
    st.session_state.points = []
 
if st.button("Reset"):
    st.session_state.points = []
    st.rerun()
 
m = folium.Map(location=[47.3769, 8.5417], zoom_start=13)
 
for i, point in enumerate(st.session_state.points):
    color = "green" if i == 0 else "red"
    label = "Start" if i == 0 else "Ziel"
    folium.Marker(location=point, tooltip=label,
                  icon=folium.Icon(color=color)).add_to(m)
 
if len(st.session_state.points) == 2:
    start, end = st.session_state.points
    route_normal_gdf, route_safe_gdf = get_routes(start, end)
 
    for _, row in route_normal_gdf.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(coords, color="blue", weight=3,
                        tooltip="Kürzeste Route").add_to(m)
 
    for _, row in route_safe_gdf.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(coords, color="green", weight=5,
                        tooltip="Sicherste Route").add_to(m)
 
    map_data = st_folium(m, width=700, height=600)
 
    # --------------------------- SIDEBAR ---------------------------
    with st.sidebar:
        st.header("🔍 Routenvergleich")
 
        scores_df = pd.read_csv(PROCESSED_DIR / "zurich_safety_scores.csv")
 
        def get_route_stats(route_gdf):
            merged = route_gdf.reset_index()
            if "u" in merged.columns and "v" in merged.columns:
                merged["k"] = merged.get("key", 0)
                merged = merged.merge(scores_df, on=["u", "v", "k"], how="left")
            return merged
 
        normal_stats = get_route_stats(route_normal_gdf)
        safe_stats = get_route_stats(route_safe_gdf)
 
        normal_score = normal_stats["safety_score_norm"].mean()
        safe_score = safe_stats["safety_score_norm"].mean()
 
        st.metric("Ø Safety Score – Kürzeste Route", f"{normal_score:.2f}")
        st.metric("Ø Safety Score – Sicherste Route", f"{safe_score:.2f}",
                  delta=f"{safe_score - normal_score:+.2f}")
 
        st.divider()
        st.subheader("🚧 Was wurde vermieden?")
 
        feature_labels = {
            "is_tunnel":       "Tunnel",
            "is_bridge":       "Brücken",
            "highway_primary": "Hauptstrassen",
            "highway_secondary": "Nebenstrassen",
            "maxspeed":        "Hohe Geschwindigkeit",
            "busy_road":       "Vielbefahrene Strassen",
            "road_capacity":   "Hohe Strassenkapazität",
            "dead_end":        "Sackgassen",
            "is_oneway":       "Einbahnstrassen",
            "has_sidewalk":    "Kein Gehsteig",
        }
 
        edge_feats = pd.read_csv(PROCESSED_DIR / "zurich_edge_features.csv")
 
        def get_edge_feature_means(route_gdf):
            merged = route_gdf.reset_index()
            if "u" in merged.columns and "v" in merged.columns:
                merged["k"] = merged.get("key", 0)
                merged = merged.merge(edge_feats, on=["u", "v", "k"], how="left")
            cols = [c for c in feature_labels.keys() if c in merged.columns]
            return merged[cols].mean()
 
        normal_feats = get_edge_feature_means(route_normal_gdf)
        safe_feats = get_edge_feature_means(route_safe_gdf)
 
        diff = normal_feats - safe_feats
        if "has_sidewalk" in diff:
            diff["has_sidewalk"] = safe_feats["has_sidewalk"] - normal_feats["has_sidewalk"]
 
        top5 = diff.sort_values(ascending=False).head(5)
 
        if top5.max() < 0.01:
            st.write("Die Routen sind sehr ähnlich – kaum Unterschied.")
        else:
            for feat, val in top5.items():
                if val > 0.01:
                    label = feature_labels.get(feat, feat)
                    st.write(f"**{label}** weniger auf sicherer Route")
 
else:
    map_data = st_folium(m, width=700, height=600)
 
    with st.sidebar:
        st.header("Anleitung")
        st.write("1. Klicke auf die Karte um den **Startpunkt** zu setzen")
        st.write("2. Klicke erneut um den **Zielpunkt** zu setzen")
        st.write("3. Die Routen werden automatisch berechnet")
        st.write("")
        st.write("Blaue Linie = Kürzeste Route")
        st.write("Grüne Linie = Sicherste Route")
 
# --------------------------- KLICK HANDLER ---------------------------
if map_data and map_data.get("last_clicked"):
    clicked = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
    if len(st.session_state.points) < 2:
        if len(st.session_state.points) == 0 or clicked != st.session_state.points[-1]:
            st.session_state.points.append(clicked)
            st.rerun()