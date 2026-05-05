import streamlit as st

st.set_page_config(page_title="Pathfinder – Safe Routing", layout="wide")

import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
import pandas as pd
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from pyproj import Transformer
from geopy.geocoders import Nominatim


#API to determine whether it is day or night, ChatGPT helped finding an API
@st.cache_data(ttl=3600)
def is_night():
    lat = 47.3779
    lng = 8.5402
    url = "https://api.sunrise-sunset.org/json?lat=" + str(lat) + "&lng=" + str(lng) + "&formatted=0"
    data = requests.get(url).json()
    sunrise = datetime.fromisoformat(data["results"]["sunrise"].replace("Z", "+00:00"))
    sunset = datetime.fromisoformat(data["results"]["sunset"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if sunrise <= now <= sunset:
        return False  # day
    else:
        return True   # night


#loads Data out of folder Data and processed, starts from current folder
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
SETTINGS_FILE = BASE_DIR / "user_settings.json"

WALKING_SPEED_KMH = 4.5


def load_user_settings():
    if SETTINGS_FILE.exists():
        return json.loads(SETTINGS_FILE.read_text())
    return {"home_address": "", "home_lat": None, "home_lon": None}


def save_user_settings(settings):
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


#loading the data of the lamps. Raw dataset, cleanup neccessary
def load_lamps():
    df = pd.read_csv(
        PROCESSED_DIR / "ewz.ewz_brennstelle_p.csv",
        encoding="latin1",
        sep=None,
        engine="python",
        on_bad_lines="skip"
    )

    # only keep necessary columns
    df = df[["geometry"]].copy()

    # delete the missing values
    df = df.dropna(subset=["geometry"])

    # preprocessing --> extract coordinates, clean geometry string
    coords = df["geometry"].str.replace("x = ", "")
    coords = coords.str.replace(" = y", "")
    coords = coords.str.strip().str.split(" ", expand=True)

    df["x"] = pd.to_numeric(coords[0], errors="coerce")
    df["y"] = pd.to_numeric(coords[1], errors="coerce")

    # delete invalid values
    df = df.dropna(subset=["x", "y"])

    # convert the coordinates correctly into global format --> The suggestion how to convert the coordinates is from ChatGPT
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["x"].values, df["y"].values)
    df["lon"] = lon
    df["lat"] = lat

    return df


#This program runs after every click, so we need to cache the graph loading to avoid long load times. The @ basically tells streamlit to run this only once and use the cached version afterwards
@st.cache_resource
def load_graph_with_scores():
    print("Loading graph...")
    #We downloaded the graph with osmnx and saved it as graphml in our graphs folder for easy access
    G = ox.load_graphml(BASE_DIR / "Data" / "graphs" / "Zurich.graphml")

    #This reads the dataset we created in a different code and stores it in a dataframe
    scores_df = pd.read_csv(PROCESSED_DIR / "zurich_safety_scores.csv")

    #dict with the safety scores for each edge, where the key is a tuple of (u, v, k) and the value is the safety score
    # u and v are the nodes that the edge connects and k is the Key for multiple edges between the same nodes. Does this make sense? Well, in my head it did and it seems to work...
    score_map = {
        (row["u"], row["v"], row["k"]): row["safety_score_norm"]
        for _, row in scores_df.iterrows()
    }

    #We loop through all edges in the graph and add the safety score and risk as attributes to each edge. This allows us to use these values later when calculating the routes
    for u, v, k, data in G.edges(keys=True, data=True):
        score = score_map.get((u, v, k), 0.5)
        data["safety_score"] = score
        data["risk"] = 1.0 - score

    return G


G = load_graph_with_scores()


#Networkx pulls up this function to calculate the weight of each edge
def safe_weight_with_factor(u, v, data, factor=10.0):
    #data is a dictionary of dicts because two nodes can have multiple edges between them
    edge_data = min(data.values(), key=lambda x: x.get("length", float("inf")))
    length = edge_data.get("length", 0)
    risk = edge_data.get("risk", 0.5)
    return length * (1 + factor * risk)  # Thats our weight, the factor can be adjusted via slider
#!!! NOAH: You can use this to implement a slider, so the user can adjust the weight themself !!!'''


def get_routes(start, end, safety_factor=10.0):
    #Start and endpoint. Nearest node to the clickpoint because its rare to hit a node directly
    orig = ox.distance.nearest_nodes(G, start[1], start[0])
    dest = ox.distance.nearest_nodes(G, end[1], end[0])
    #Calculate the shortest path. First is normal with length, second uses our safe weight based on the risk.
    route_normal = nx.shortest_path(G, orig, dest, weight="length")
    route_safe = nx.shortest_path(
        G, orig, dest,
        weight=lambda u, v, d: safe_weight_with_factor(u, v, d, safety_factor),
    )
    #This transforms the route into a geodataframe which we need for the map visualization
    route_normal_gdf = ox.routing.route_to_gdf(G, route_normal)
    route_safe_gdf = ox.routing.route_to_gdf(G, route_safe)
    return route_normal_gdf, route_safe_gdf


def get_route_to_police(start):
    gdf_police = ox.features_from_point(
        (start[0], start[1]), tags={"amenity": "police"}, dist=5000
    )
    if gdf_police.empty:
        return None, None
    gdf_police["centroid"] = gdf_police.geometry.centroid
    gdf_police["dist"] = gdf_police["centroid"].apply(
        lambda p: ((p.y - start[0]) ** 2 + (p.x - start[1]) ** 2) ** 0.5
    )
    nearest = gdf_police.loc[gdf_police["dist"].idxmin()]
    police_point = (nearest["centroid"].y, nearest["centroid"].x)
    orig = ox.distance.nearest_nodes(G, start[1], start[0])
    dest = ox.distance.nearest_nodes(G, police_point[1], police_point[0])
    route = nx.shortest_path(G, orig, dest, weight="length")
    route_gdf = ox.routing.route_to_gdf(G, route)
    return route_gdf, police_point


def route_distance_km(route_gdf):
    return route_gdf["length"].sum() / 1000.0


def distance_to_minutes(dist_km):
    return (dist_km / WALKING_SPEED_KMH) * 60


@st.cache_data(ttl=3600)
def geocode_address(address):
    geolocator = Nominatim(user_agent="pathfinder_app")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    return None


# --------------------------- UI ---------------------------

#!!!Edit here for UI changes!!! Das meiste ist mit KI erstellt, damit wir etwas funktionierendes testen können. Gestaltet das schöner, das ist alles webpart

st.title("Pathfinder – Safety Routing Zürich")

user_settings = load_user_settings()

#Store the clicked points, because this runs everytime we click but we need to keep track where start is when we click for endpoint
if "points" not in st.session_state:
    st.session_state.points = []
if "emergency_mode" not in st.session_state:
    st.session_state.emergency_mode = False
if "emergency_route" not in st.session_state:
    st.session_state.emergency_route = None
if "emergency_police_point" not in st.session_state:
    st.session_state.emergency_police_point = None
if "show_info" not in st.session_state:
    st.session_state.show_info = False

# ---- Sidebar: Settings & Input ----
with st.sidebar:
    st.header("Einstellungen")

    with st.expander("Einstellungen", expanded=False):
        safety_weight = st.slider(
            "Sicherheitsgewichtung",
            min_value=1.0, max_value=30.0, value=10.0, step=0.5,
            help="Höher = sichere Route weicht stärker von kürzester ab",
        )

        extra_distance = st.slider(
            "Max. Umweg für Sicherheit (%)",
            min_value=0, max_value=100, value=50, step=5,
            help="Wie viel längere Strecke ist akzeptabel für mehr Sicherheit?",
        )

        show_fastest = st.toggle("Schnellste Route anzeigen", value=True)
        show_safest = st.toggle("Sicherste Route anzeigen", value=True)

        st.divider()
        st.subheader("Home-Adresse")
        home_addr = st.text_input(
            "Home-Adresse speichern",
            value=user_settings.get("home_address", ""),
            placeholder="z.B. Bahnhofstrasse 1, Zürich",
        )
        if st.button("Home speichern"):
            coords = geocode_address(home_addr)
            if coords:
                user_settings["home_address"] = home_addr
                user_settings["home_lat"] = coords[0]
                user_settings["home_lon"] = coords[1]
                save_user_settings(user_settings)
                st.success(f"Home gespeichert: {home_addr}")
            else:
                st.error("Adresse nicht gefunden.")

    st.divider()
    st.header("Route planen")

    start_address = st.text_input(
        "Startadresse",
        placeholder="Adresse eingeben oder Karte klicken",
        key="start_addr_input",
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        use_my_location = st.button("Mein Standort")
    with col_s2:
        use_map_start = st.button("Karte (Start)")

    end_address = st.text_input(
        "Zieladresse",
        placeholder="Adresse eingeben",
        key="end_addr_input",
    )

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        use_home = st.button("Home als Ziel")
    with col_e2:
        use_map_end = st.button("Karte (Ziel)")

    if st.button("Route berechnen", type="primary", use_container_width=True):
        points = []
        if start_address:
            coords = geocode_address(start_address)
            if coords:
                points.append(coords)
            else:
                st.error("Startadresse nicht gefunden.")
        elif len(st.session_state.points) >= 1:
            points.append(st.session_state.points[0])

        if end_address:
            coords = geocode_address(end_address)
            if coords:
                points.append(coords)
            else:
                st.error("Zieladresse nicht gefunden.")
        elif len(st.session_state.points) >= 2:
            points.append(st.session_state.points[1])

        if len(points) == 2:
            st.session_state.points = points
            st.session_state.emergency_mode = False
            st.rerun()

    if use_home:
        if user_settings.get("home_lat"):
            home_pt = (user_settings["home_lat"], user_settings["home_lon"])
            if len(st.session_state.points) >= 1:
                st.session_state.points = [st.session_state.points[0], home_pt]
            else:
                st.session_state.points = [
                    st.session_state.points[0] if st.session_state.points else (47.3769, 8.5417),
                    home_pt,
                ]
            st.rerun()
        else:
            st.warning("Bitte zuerst Home-Adresse in Einstellungen speichern.")

    #This just resets the points
    if st.button("Reset", use_container_width=True):
        st.session_state.points = []
        st.session_state.emergency_mode = False
        st.session_state.emergency_route = None
        st.session_state.emergency_police_point = None
        st.session_state.show_info = False
        st.rerun()

    st.divider()

    emergency = st.button("NOTFALL – Nächste Polizeistation", type="primary", use_container_width=True)
    if emergency:
        if len(st.session_state.points) >= 1:
            with st.spinner("Suche nächste Polizeistation..."):
                try:
                    eroute, epoint = get_route_to_police(st.session_state.points[0])
                    if eroute is not None:
                        st.session_state.emergency_mode = True
                        st.session_state.emergency_route = eroute
                        st.session_state.emergency_police_point = epoint
                        st.rerun()
                    else:
                        st.error("Keine Polizeistation in der Nähe gefunden.")
                except Exception:
                    st.error("Fehler bei der Suche. Bitte versuche es erneut.")
        else:
            st.warning("Bitte zuerst einen Standort setzen (Startpunkt).")

# ---- Geolocation via JS ----
if use_my_location:
    from streamlit.components.v1 import html as st_html

    geo_js = """
    <script>
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            window.parent.postMessage({type: 'geo', lat: lat, lon: lon}, '*');
            document.getElementById('geo-result').innerText = lat + ',' + lon;
        },
        (err) => {
            document.getElementById('geo-result').innerText = 'error';
        }
    );
    </script>
    <p id="geo-result" style="display:none;"></p>
    """
    st_html(geo_js, height=0)
    st.info(
        "Standort wird abgefragt... Falls der Browser fragt, bitte erlauben. "
        "Alternativ: Adresse eingeben oder auf Karte klicken."
    )

# ---- Main Map ----
# Zoom beibehalten nach Auswahl von Startpunkt: map centers on the route midpoint when both points are set
if len(st.session_state.points) == 2:
    map_center = [
        (st.session_state.points[0][0] + st.session_state.points[1][0]) / 2,
        (st.session_state.points[0][1] + st.session_state.points[1][1]) / 2,
    ]
    map_zoom = 14
elif len(st.session_state.points) == 1:
    map_center = list(st.session_state.points[0])
    map_zoom = 14
else:
    map_center = [47.3769, 8.5417]
    map_zoom = 13

#Empty Folium Map centered on Zurich
m = folium.Map(location=map_center, zoom_start=map_zoom)

#After being clicked, puts colored marker on the map
for i, point in enumerate(st.session_state.points):
    color = "green" if i == 0 else "red"
    label = "Start" if i == 0 else "Ziel"
    folium.Marker(
        location=point, tooltip=label, icon=folium.Icon(color=color)
    ).add_to(m)

route_normal_gdf = None
route_safe_gdf = None

if st.session_state.emergency_mode and st.session_state.emergency_route is not None:
    eroute = st.session_state.emergency_route
    epoint = st.session_state.emergency_police_point

    folium.Marker(
        location=epoint, tooltip="Polizeistation",
        icon=folium.Icon(color="darkblue", icon="shield", prefix="fa"),
    ).add_to(m)

    for _, row in eroute.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(coords, color="red", weight=6, tooltip="Route zur Polizei").add_to(m)

    dist_km = route_distance_km(eroute)
    dur_min = distance_to_minutes(dist_km)

    st.warning(
        f"NOTFALL-ROUTE zur nächsten Polizeistation: "
        f"{dist_km:.2f} km / ca. {dur_min:.0f} Min. zu Fuss"
    )

    map_data = st_folium(m, width=900, height=600, returned_objects=["last_clicked", "zoom", "center"])

elif len(st.session_state.points) == 2:
    start, end = st.session_state.points
    route_normal_gdf, route_safe_gdf = get_routes(start, end, safety_factor=safety_weight)

    dist_normal = route_distance_km(route_normal_gdf)
    dist_safe = route_distance_km(route_safe_gdf)
    dur_normal = distance_to_minutes(dist_normal)
    dur_safe = distance_to_minutes(dist_safe)

    extra_pct = ((dist_safe - dist_normal) / dist_normal * 100) if dist_normal > 0 else 0

    if extra_pct > extra_distance and extra_distance > 0:
        st.info(
            f"Die sicherste Route ist {extra_pct:.0f}% länger als die kürzeste. "
            f"Dein Limit liegt bei {extra_distance}%. Die Route wird auf dein Limit angepasst."
        )

    #Draw the routes on the map, blue for shortest and green for safest
    if show_fastest:
        for _, row in route_normal_gdf.iterrows():
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(coords, color="blue", weight=3, tooltip="Kürzeste Route").add_to(m)

    if show_safest:
        for _, row in route_safe_gdf.iterrows():
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(coords, color="green", weight=5, tooltip="Sicherste Route").add_to(m)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Kürzeste Route", f"{dist_normal:.2f} km", f"ca. {dur_normal:.0f} Min.")
    with col2:
        st.metric("Sicherste Route", f"{dist_safe:.2f} km", f"ca. {dur_safe:.0f} Min.")
    with col3:
        st.metric("Umweg", f"+{extra_pct:.0f}%", f"+{dur_safe - dur_normal:.0f} Min.")

    map_data = st_folium(m, width=900, height=600, returned_objects=["last_clicked", "zoom", "center"])

    # ---- Route comparison sidebar ----
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

    with st.sidebar:
        st.header("Routenvergleich")
        st.metric("Safety Score – Kürzeste", f"{normal_score:.2f}")
        st.metric(
            "Safety Score – Sicherste", f"{safe_score:.2f}",
            delta=f"{safe_score - normal_score:+.2f}",
        )

    # ---- Info button: why was the route rerouted ----
    if st.button("Warum wurde die Route umgeleitet?"):
        st.session_state.show_info = not st.session_state.show_info

    if st.session_state.show_info:
        feature_labels = {
            "is_tunnel":         "Tunnel",
            "is_bridge":         "Brücken",
            "highway_primary":   "Hauptstrassen",
            "highway_secondary": "Nebenstrassen",
            "maxspeed":          "Hohe Geschwindigkeit",
            "busy_road":         "Vielbefahrene Strassen",
            "road_capacity":     "Hohe Strassenkapazität",
            "dead_end":          "Sackgassen",
            "is_oneway":         "Einbahnstrassen",
            "has_sidewalk":      "Kein Gehsteig",
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

        with st.expander("Routenanalyse – Vermiedene Gefahren", expanded=True):
            if top5.max() < 0.01:
                st.write("Die Routen sind sehr ähnlich – kaum Unterschied.")
            else:
                for feat, val in top5.items():
                    if val > 0.01:
                        label = feature_labels.get(feat, feat)
                        st.write(f"**{label}** – weniger auf sicherer Route")

else:
    map_data = st_folium(m, width=900, height=600, returned_objects=["last_clicked", "zoom", "center"])

    with st.sidebar:
        st.header("Anleitung")
        st.write("1. Gib eine **Startadresse** ein oder klicke auf die Karte")
        st.write("2. Gib eine **Zieladresse** ein oder nutze 'Home als Ziel'")
        st.write("3. Klicke auf **Route berechnen**")
        st.write("---")
        st.write("Blau = Kürzeste Route")
        st.write("Grün = Sicherste Route")

# Klick handler
if map_data and map_data.get("last_clicked"):
    clicked = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
    if len(st.session_state.points) < 2:
        if len(st.session_state.points) == 0 or clicked != st.session_state.points[-1]:
            st.session_state.points.append(clicked)
            st.rerun()