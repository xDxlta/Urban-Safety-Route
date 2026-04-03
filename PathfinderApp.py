import streamlit as st
import osmnx as ox
import networkx as nx
import pydeck as pdk
import folium
from streamlit_folium import st_folium


#------Risk logic------
def compute_risk(data, is_night=False):
    risk = 0

    tunnel_value = data.get("tunnel")
    highway = data.get("highway")

    if tunnel_value in ["yes", "building_passage", "covered"]:
        risk += 1.0

    if highway in ["primary", "secondary"]:
        risk += 2.0

    return risk

#Load the Graph and calculate the risk + cache it, because otherwise it takes over a minute to use the map
@st.cache_resource
def load_graph():
    G = ox.load_graphml("zurich_walk.graphml")

    for u, v, k, data in G.edges(keys=True, data=True):
        data["risk"] = compute_risk(data)

    return G
G = load_graph()


# Define a custom weight function that combines length and risk. TODO : Adjust the weight later, so that the risk is not overemphasized. Currently, it adds a large penalty to the length based on the risk, which may lead to very long detours for routes with any risk.
def safe_weight(u, v, data):
    edge_data = min(data.values(), key=lambda x: x.get("length", float("inf")))
    return edge_data.get("length", 0) + 10000 * edge_data.get("risk", 0)

# Define a function to get the normal and safe routes between a start and end point. It uses the nearest nodes in the graph for the start and end coordinates, calculates the shortest path based on length for the normal route and based on the custom safe weight for the safe route, and converts these routes to GeoDataFrames.
def get_routes(start, end):
    orig = ox.distance.nearest_nodes(G, start[1], start[0])
    dest = ox.distance.nearest_nodes(G, end[1], end[0])

    route_normal = nx.shortest_path(G, orig, dest, weight="length")
    route_safe = nx.shortest_path(G, orig, dest, weight=safe_weight)

    route_normal_gdf = ox.routing.route_to_gdf(G, route_normal)
    route_safe_gdf = ox.routing.route_to_gdf(G, route_safe)

    return route_normal_gdf, route_safe_gdf

st.title("Safety Routing Zurich")

if "points" not in st.session_state:
    st.session_state.points = []

if st.button("Reset"):
    st.session_state.points = []
    st.rerun()

# Karte zuerst bauen
m = folium.Map(location=[47.3769, 8.5417], zoom_start=13)

# Bereits gesetzte Punkte anzeigen
for i, point in enumerate(st.session_state.points):
    color = "green" if i == 0 else "red"
    label = "Start" if i == 0 else "Ziel"
    folium.Marker(location=point, tooltip=label, icon=folium.Icon(color=color)).add_to(m)

# Wenn 2 Punkte da sind, Route zeichnen
if len(st.session_state.points) == 2:
    start, end = st.session_state.points
    route_normal_gdf, route_safe_gdf = get_routes(start, end)

    for _, row in route_normal_gdf.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(coords, color="blue", weight=3).add_to(m)

    for _, row in route_safe_gdf.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(coords, color="red", weight=5).add_to(m)

# Karte anzeigen
map_data = st_folium(m, width=900, height=600)

# Klick verarbeiten
if map_data and map_data.get("last_clicked"):
    clicked = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])

    if len(st.session_state.points) < 2:
        if len(st.session_state.points) == 0 or clicked != st.session_state.points[-1]:
            st.session_state.points.append(clicked)
            st.rerun()