<<<<<<< Updated upstream

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
from streamlit_js_eval import get_geolocation


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

# Approximate bounding box of the city of Zürich (lat_min, lat_max, lon_min, lon_max)
ZURICH_BOUNDS = (47.32, 47.435, 8.44, 8.625)


def is_in_zurich(lat, lon):
    lat_min, lat_max, lon_min, lon_max = ZURICH_BOUNDS
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


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
    if not address or not address.strip():
        return None
    geolocator = Nominatim(user_agent="pathfinder_app")
    # Bounding-Box für Zürich: SW- und NE-Ecke als (lat, lon)
    lat_min, lat_max, lon_min, lon_max = ZURICH_BOUNDS
    viewbox = [(lat_min, lon_min), (lat_max, lon_max)]

    location = None
    # 1) Direkt auf Zürich beschränkt suchen
    try:
        location = geolocator.geocode(
            address, viewbox=viewbox, bounded=True, country_codes="ch"
        )
    except Exception:
        location = None

    # 2) Fallback: Stadt explizit anhängen, falls nicht schon enthalten
    if location is None and "zürich" not in address.lower() and "zurich" not in address.lower():
        try:
            location = geolocator.geocode(
                f"{address}, Zürich, Schweiz",
                viewbox=viewbox, bounded=True, country_codes="ch",
            )
        except Exception:
            location = None

    if location:
        return (location.latitude, location.longitude)
    return None


# --------------------------- UI ---------------------------


st.title("Pathfinder – Safety Routing Zürich")

map_tiles = "OpenStreetMap"
legend_bg = "white"
legend_text = "#111"
legend_border = "#888"

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
if "show_safety_tips" not in st.session_state:
    st.session_state.show_safety_tips = False
if "geo_request" not in st.session_state:
    st.session_state.geo_request = None

# Einheitliche Fehlermeldung für Punkte ausserhalb der Stadt Zürich
ZURICH_BOUNDS_ERROR = (
    "Der ausgewählte Punkt liegt ausserhalb der Stadt Zürich, "
    "bitte einen Start-und Endpunkt innerhalb der Stadtgrenzen wählen"
)

# ---- Sidebar: Anleitung → Route planen → Homeadresse → Einstellungen ----
with st.sidebar:
    # 1. Anleitung
    st.header("Anleitung")
    st.write(
        "Wähle **zwei Punkte direkt auf der Karte** – die Route wird automatisch berechnet."
    )
    st.write(
        "**Oder** gib unter **Route planen** Start- und Zieladresse manuell ein und klicke auf *Route berechnen*."
    )

    st.divider()

    # 2. Route planen
    st.header("Route planen")

    # Prefill-Werte aus vorigem Run anwenden, BEVOR die Text-Inputs gerendert werden
    if "_prefill_start" in st.session_state:
        st.session_state["start_addr_input"] = st.session_state.pop("_prefill_start")
    if "_prefill_end" in st.session_state:
        st.session_state["end_addr_input"] = st.session_state.pop("_prefill_end")

    start_address = st.text_input(
        "Startadresse",
        placeholder="Adresse eingeben oder Karte klicken",
        key="start_addr_input",
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        use_my_loc_start = st.button("Mein Standort", key="my_loc_start")
    with col_s2:
        use_home_start = st.button("Zuhause", key="home_start")

    end_address = st.text_input(
        "Zieladresse",
        placeholder="Adresse eingeben",
        key="end_addr_input",
    )

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        use_my_loc_end = st.button("Mein Standort", key="my_loc_end")
    with col_e2:
        use_home_end = st.button("Zuhause", key="home_end")

    if st.button("Route berechnen", type="primary", use_container_width=True):
        points = []
        out_of_bounds = False
        if start_address:
            coords = geocode_address(start_address)
            if coords:
                if not is_in_zurich(coords[0], coords[1]):
                    st.error(ZURICH_BOUNDS_ERROR)
                    out_of_bounds = True
                else:
                    points.append(coords)
            else:
                st.error("Startadresse nicht gefunden.")
        elif len(st.session_state.points) >= 1:
            points.append(st.session_state.points[0])

        if end_address:
            coords = geocode_address(end_address)
            if coords:
                if not is_in_zurich(coords[0], coords[1]):
                    st.error(ZURICH_BOUNDS_ERROR)
                    out_of_bounds = True
                else:
                    points.append(coords)
            else:
                st.error("Zieladresse nicht gefunden.")
        elif len(st.session_state.points) >= 2:
            points.append(st.session_state.points[1])

        if len(points) == 2 and not out_of_bounds:
            st.session_state.points = points
            st.session_state.emergency_mode = False
            st.rerun()

    # ---- Zuhause als Start oder Ziel ----
    def _set_home_point(slot):
        """slot: 0 = Start, 1 = Ziel"""
        if not user_settings.get("home_lat"):
            st.warning("Bitte zuerst Home-Adresse speichern.")
            return
        home_pt = (user_settings["home_lat"], user_settings["home_lon"])
        if not is_in_zurich(home_pt[0], home_pt[1]):
            st.error(ZURICH_BOUNDS_ERROR)
            return
        pts = list(st.session_state.points)
        other_slot = 1 - slot
        if len(pts) > other_slot and pts[other_slot] == home_pt:
            st.error("Start- und Zielpunkt sind identisch - bitte einen anderen Punkt wählen")
            return
        while len(pts) <= slot:
            pts.append(home_pt)
        pts[slot] = home_pt
        st.session_state.points = pts[:2]
        # Adressfeld vorbefüllen (wird beim nächsten Run vor dem Text-Input gesetzt)
        home_addr_str = user_settings.get("home_address", "")
        if home_addr_str:
            prefill_key = "_prefill_start" if slot == 0 else "_prefill_end"
            st.session_state[prefill_key] = home_addr_str
        st.rerun()

    if use_home_start:
        _set_home_point(0)
    if use_home_end:
        _set_home_point(1)

    # ---- "Mein Standort"-Buttons setzen Anfrage-Flag ----
    if use_my_loc_start:
        st.session_state.geo_request = "start"
        st.rerun()
    if use_my_loc_end:
        st.session_state.geo_request = "end"
        st.rerun()

    #This just resets the points
    if st.button("Reset", use_container_width=True):
        st.session_state.points = []
        st.session_state.emergency_mode = False
        st.session_state.emergency_route = None
        st.session_state.emergency_police_point = None
        st.session_state.show_info = False
        st.session_state.geo_request = None
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

    # Platzhalter für den Routenvergleich – wird gefüllt sobald eine Route berechnet wurde
    routenvergleich_container = st.container()

    st.divider()

    # 3. Homeadresse
    st.header("Homeadresse")

    # Prefill anwenden, bevor das Text-Input gerendert wird (z.B. nach Reset)
    if "_prefill_home" in st.session_state:
        st.session_state["home_addr_input"] = st.session_state.pop("_prefill_home")

    home_addr = st.text_input(
        "Home-Adresse",
        value=user_settings.get("home_address", ""),
        placeholder="z.B. Bahnhofstrasse 1, Zürich",
        key="home_addr_input",
    )

    home_col1, home_col2 = st.columns(2)
    with home_col1:
        save_home_clicked = st.button("Adresse speichern", use_container_width=True)
    with home_col2:
        reset_home_clicked = st.button("Reset", use_container_width=True, key="home_reset_btn")

    if save_home_clicked:
        coords = geocode_address(home_addr)
        if coords:
            if not is_in_zurich(coords[0], coords[1]):
                st.error(ZURICH_BOUNDS_ERROR)
            else:
                user_settings["home_address"] = home_addr
                user_settings["home_lat"] = coords[0]
                user_settings["home_lon"] = coords[1]
                save_user_settings(user_settings)
                st.success(f"Home gespeichert: {home_addr}")
        else:
            st.error("Adresse nicht gefunden.")

    if reset_home_clicked:
        user_settings["home_address"] = ""
        user_settings["home_lat"] = None
        user_settings["home_lon"] = None
        save_user_settings(user_settings)
        st.session_state["_prefill_home"] = ""
        st.rerun()

    st.divider()

    # 4. Einstellungen
    with st.expander("Einstellungen", expanded=False):
        # CSS um die Min/Max-Werte und Slider-Tooltips auszublenden
        st.markdown(
            """
            <style>
                div[data-testid="stExpander"] [data-testid="stSlider"] [data-testid="stTickBar"] { display: none !important; }
                div[data-testid="stExpander"] [data-testid="stSlider"] [data-testid="stThumbValue"] { display: none !important; }
                div[data-testid="stExpander"] [data-testid="stSlider"] [data-baseweb="tooltip"] { display: none !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        def _qual_slider(label, key):
            st.markdown(f"**{label}**")
            val = st.slider(
                label, min_value=1, max_value=10, value=5, step=1,
                format=" ", label_visibility="collapsed", key=key,
            )
            col_l, col_r = st.columns(2)
            col_l.caption("klein")
            col_r.markdown(
                "<div style='text-align:right; color:rgb(163,168,184); font-size:0.875em;'>hoch</div>",
                unsafe_allow_html=True,
            )
            return val

        # 1–10 (Slider) → in interne Wertebereiche umrechnen
        sw_level = _qual_slider("Sicherheitsgewichtung", key="sw_slider")
        safety_weight = 1.0 + (sw_level - 1) * (29.0 / 9.0)  # 1–10 → 1.0–30.0

        ed_level = _qual_slider("Zusätzlicher Umweg für Sicherheit", key="ed_slider")
        extra_distance = int(round((ed_level - 1) * (100.0 / 9.0)))  # 1–10 → 0–100

        show_fastest = st.toggle("Schnellste Route anzeigen", value=True)
        show_safest = st.toggle("Sicherste Route anzeigen", value=True)

# ---- Geolocation: Browser-Standort abrufen, wenn angefordert ----
if st.session_state.geo_request:
    loc = get_geolocation()
    if loc and isinstance(loc, dict) and loc.get("coords"):
        coords = loc["coords"]
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        if lat is not None and lon is not None:
            pt = (float(lat), float(lon))
            slot = 0 if st.session_state.geo_request == "start" else 1
            other_slot = 1 - slot
            target = st.session_state.geo_request
            st.session_state.geo_request = None  # Flag immer löschen
            if not is_in_zurich(pt[0], pt[1]):
                st.error(ZURICH_BOUNDS_ERROR)
            else:
                pts = list(st.session_state.points)
                if len(pts) > other_slot and pts[other_slot] == pt:
                    st.error("Start- und Zielpunkt sind identisch - bitte einen anderen Punkt wählen")
                else:
                    while len(pts) <= slot:
                        pts.append(pt)
                    pts[slot] = pt
                    st.session_state.points = pts[:2]
                    st.rerun()
    else:
        st.info("Standort wird abgefragt... Bitte im Browser erlauben.")

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

#Empty Folium Map centered on Zurich, theme depends on time of day
m = folium.Map(location=map_center, zoom_start=map_zoom, tiles=map_tiles)

#After being clicked, puts colored marker on the map
for i, point in enumerate(st.session_state.points):
    color = "green" if i == 0 else "red"
    label = "Start" if i == 0 else "Ziel"
    folium.Marker(
        location=point, tooltip=label, icon=folium.Icon(color=color)
    ).add_to(m)

# ---- Dauerhafter Zuhause-Marker, falls eine Home-Adresse gespeichert ist ----
home_saved = user_settings.get("home_lat") is not None and user_settings.get("home_lon") is not None
if home_saved:
    folium.Marker(
        location=(user_settings["home_lat"], user_settings["home_lon"]),
        tooltip=f"Zuhause: {user_settings.get('home_address', '')}",
        icon=folium.Icon(color="purple", icon="home", prefix="fa"),
    ).add_to(m)

# ---- Legende als HTML-Overlay direkt auf der Karte ----
in_emergency = st.session_state.emergency_mode and st.session_state.emergency_route is not None
emergency_legend = (
    '<span style="background:#ff4136; width:22px; height:4px; display:inline-block; vertical-align:middle; margin-right:6px;"></span>Notfall-Route<br>'
    '<span style="color:#0074d9; font-size:18px; margin-right:4px;">●</span>Polizeistation<br>'
) if in_emergency else ""

home_legend = (
    '<span style="color:#9b59b6; font-size:18px; margin-right:4px;">⌂</span>Zuhause<br>'
) if home_saved else ""

legend_html = f"""
<div style="position: absolute; bottom: 24px; left: 24px; width: 220px;
            background-color: {legend_bg}; color: {legend_text};
            border: 1px solid {legend_border}; z-index: 9999;
            padding: 10px 12px; font-size: 13px; border-radius: 6px;
            font-family: sans-serif; box-shadow: 0 2px 6px rgba(0,0,0,0.25);
            line-height: 1.7;">
  <b>Legende</b><br>
  <span style="background:#3388ff; width:22px; height:4px; display:inline-block; vertical-align:middle; margin-right:6px;"></span>Kürzeste Route<br>
  <span style="background:#2ecc40; width:22px; height:4px; display:inline-block; vertical-align:middle; margin-right:6px;"></span>Sicherste Route<br>
  <span style="color:#2ecc40; font-size:18px; margin-right:4px;">●</span>Startpunkt<br>
  <span style="color:#ff4136; font-size:18px; margin-right:4px;">●</span>Zielpunkt<br>
  {home_legend}{emergency_legend}
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

route_normal_gdf = None
route_safe_gdf = None
emergency_info = None
route_info = None

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

    map_data = st_folium(m, width=900, height=600, returned_objects=["last_clicked", "zoom", "center"])

    # Notfall-Info unterhalb der Karte
    st.warning(
        f"NOTFALL-ROUTE zur nächsten Polizeistation: "
        f"{dist_km:.2f} km / ca. {dur_min:.0f} Min. zu Fuss"
    )

elif len(st.session_state.points) == 2:
    start, end = st.session_state.points
    route_normal_gdf, route_safe_gdf = get_routes(start, end, safety_factor=safety_weight)

    dist_normal = route_distance_km(route_normal_gdf)
    dist_safe = route_distance_km(route_safe_gdf)
    dur_normal = distance_to_minutes(dist_normal)
    dur_safe = distance_to_minutes(dist_safe)

    extra_pct = ((dist_safe - dist_normal) / dist_normal * 100) if dist_normal > 0 else 0

    #Draw the routes on the map, blue for shortest and green for safest
    if show_fastest:
        for _, row in route_normal_gdf.iterrows():
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(coords, color="blue", weight=3, tooltip="Kürzeste Route").add_to(m)

    if show_safest:
        for _, row in route_safe_gdf.iterrows():
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(coords, color="green", weight=5, tooltip="Sicherste Route").add_to(m)

    map_data = st_folium(m, width=900, height=600, returned_objects=["last_clicked", "zoom", "center"])

    # ---- Infos UNTER der Karte ----
    if extra_pct > extra_distance and extra_distance > 0:
        st.info(
            f"Die sicherste Route ist {extra_pct:.0f}% länger als die kürzeste. "
            f"Dein Limit liegt bei {extra_distance}%. Die Route wird auf dein Limit angepasst."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Kürzeste Route", f"{dist_normal:.2f} km", f"ca. {dur_normal:.0f} Min.")
    with col2:
        st.metric("Sicherste Route", f"{dist_safe:.2f} km", f"ca. {dur_safe:.0f} Min.")
    with col3:
        st.metric("Umweg", f"+{extra_pct:.0f}%", f"+{dur_safe - dur_normal:.0f} Min.")

    # ---- Routenvergleich (in Sidebar-Container, zwischen Route planen und Einstellungen) ----
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

    with routenvergleich_container:
        st.header("Routenvergleich")
        st.metric("Safety Score – Kürzeste", f"{normal_score:.2f}")
        st.metric(
            "Safety Score – Sicherste", f"{safe_score:.2f}",
            delta=f"{safe_score - normal_score:+.2f}",
        )

    # ---- Info-Buttons: Reroute-Begründung und Safety-Tipps ----
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Warum wurde die Route umgeleitet?", use_container_width=True):
            st.session_state.show_info = not st.session_state.show_info
    with btn_col2:
        if st.button("Sicher nach Hause – Tipps", use_container_width=True):
            st.session_state.show_safety_tips = not st.session_state.show_safety_tips

    if st.session_state.show_safety_tips:
        with st.expander("Verhaltensregeln für eine sichere Heimreise", expanded=True):
            st.markdown(
                """
                - **Grössere Personengruppen meiden** und nicht durch sie hindurchgehen
                - **Nicht von Fremden ansprechen lassen** – freundlich, aber bestimmt weitergehen
                - **Kopfhörer leise** lassen oder ganz abnehmen, um die Umgebung wahrzunehmen
                - **Hauptstrassen und beleuchtete Wege** bevorzugen, dunkle Abkürzungen meiden
                - **Wertgegenstände** (Handy, Schmuck, Portemonnaie) nicht offen tragen
                - **Handy aufgeladen** und mit etwas Akku-Reserve dabeihaben
                - **Vertrauensperson informieren** über Route und voraussichtliche Ankunftszeit
                - **Selbstsicher gehen** – aufrechte Haltung, zielgerichteter Blick
                - **Im Zweifel** in ein offenes Geschäft, Restaurant oder zur Polizei gehen
                - **Notrufnummer 117** (Polizei) griffbereit halten
                - Wenn dir jemand folgt: **Strassenseite wechseln** oder Umweg über belebte Gegend nehmen
                - Bei akuter Gefahr: **laut um Hilfe rufen** und Aufmerksamkeit erzeugen
                """
            )

    if st.session_state.show_info:
        # Pro Feature: ("more on safe"-Phrase, "less on safe"-Phrase)
        feature_phrases = {
            "is_tunnel":         ("Mehr Tunnel",                   "Weniger Tunnel"),
            "is_bridge":         ("Mehr Brücken",                  "Weniger Brücken"),
            "highway_primary":   ("Mehr Hauptstrassen",            "Weniger Hauptstrassen"),
            "highway_secondary": ("Mehr Nebenstrassen",            "Weniger Nebenstrassen"),
            "maxspeed":          ("Höhere Geschwindigkeiten",      "Niedrigere Geschwindigkeiten"),
            "busy_road":         ("Mehr vielbefahrene Strassen",   "Weniger vielbefahrene Strassen"),
            "road_capacity":     ("Höhere Strassenkapazität",      "Niedrigere Strassenkapazität"),
            "dead_end":          ("Mehr Sackgassen",               "Weniger Sackgassen"),
            "is_oneway":         ("Mehr Einbahnstrassen",          "Weniger Einbahnstrassen"),
            "has_sidewalk":      ("Mehr Gehweg",                   "Weniger Gehweg"),
        }

        edge_feats = pd.read_csv(PROCESSED_DIR / "zurich_edge_features.csv")

        def get_edge_feature_means(route_gdf):
            merged = route_gdf.reset_index()
            if "u" in merged.columns and "v" in merged.columns:
                merged["k"] = merged.get("key", 0)
                merged = merged.merge(edge_feats, on=["u", "v", "k"], how="left")
            cols = [c for c in feature_phrases.keys() if c in merged.columns]
            return merged[cols].mean()

        normal_feats = get_edge_feature_means(route_normal_gdf)
        safe_feats = get_edge_feature_means(route_safe_gdf)

        # diff > 0: sichere Route hat MEHR davon, diff < 0: sichere Route hat WENIGER davon
        diff = safe_feats - normal_feats
        threshold = 0.01

        # Top-Unterschiede nach Betrag sortieren
        top = diff.abs().sort_values(ascending=False).head(5)

        reasons = []
        for feat in top.index:
            val = diff[feat]
            if abs(val) < threshold:
                continue
            more_phrase, less_phrase = feature_phrases.get(feat, (feat, feat))
            reasons.append(more_phrase if val > 0 else less_phrase)

        with st.expander("Routenanalyse – Gründe für die Umleitung", expanded=True):
            if not reasons:
                st.write("Die Routen sind sehr ähnlich – kaum Unterschied.")
            else:
                st.markdown("**Die Route wurde umgeleitet, da:**")
                for r in reasons:
                    st.markdown(f"- {r}")

else:
    map_data = st_folium(m, width=900, height=600, returned_objects=["last_clicked", "zoom", "center"])

# Klick handler – 1. Klick = Start, ab 2. Klick = Ziel (wird bei jedem weiteren Klick aktualisiert)
if map_data and map_data.get("last_clicked"):
    clicked = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
    if not is_in_zurich(clicked[0], clicked[1]):
        st.error(ZURICH_BOUNDS_ERROR)
    else:
        pts = list(st.session_state.points)
        if len(pts) == 0:
            pts.append(clicked)
            st.session_state.points = pts
            st.rerun()
        elif len(pts) == 1:
            if clicked != pts[0]:
                pts.append(clicked)
                st.session_state.points = pts
                st.rerun()
        else:
            # Ab 3. Klick: Zielpunkt aktualisieren, Startpunkt bleibt
            if clicked != pts[1]:
                pts[1] = clicked
                st.session_state.points = pts
>>>>>>> Stashed changes
                st.rerun()