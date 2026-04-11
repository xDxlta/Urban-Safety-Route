from pathlib import Path
import re
import pandas as pd
import osmnx as ox


# This code is responsible for taking the training base and building a feature table by looking up
# the nearest OSM edge for each point and extracting features from that edge and its nodes.
# The resulting feature table will be saved as a CSV for use in model training.

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
GRAPHS_DIR = BASE_DIR / "Data" / "graphs"

GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Set useful tags GLOBALLY so they apply to every graph download.
# Must be at module level — not inside a function — so it is always active.
ox.settings.useful_tags_way = [
    "highway", "oneway", "maxspeed", "lanes", "bridge", "tunnel",
    "lit", "sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both",
    "footway", "surface", "width", "access", "junction", "service", "length",
]


def load_training_base() -> pd.DataFrame:
    file_path = PROCESSED_DIR / "training_base.csv"
    df = pd.read_csv(file_path)
    return df


# --------------------------- CITY / GRAPH HELPERS ---------------------------

def sanitize_city_name(city_name: str) -> str:
    city_name = str(city_name).strip()
    city_name = re.sub(r"[^A-Za-z0-9]+", "_", city_name)
    return city_name.strip("_")


def get_place_query(city_name: str):
    custom_map = {
        "Amsterdam": "Amsterdam, Netherlands",
        "New York": "New York City, USA",
        "Sao Paulo": "São Paulo, Brazil",
        "Rio De Janeiro": "Rio de Janeiro, Brazil",
        "Washington DC": "Washington, District of Columbia, USA",
        "Mexico City": "Mexico City, Mexico",
        "Philadelphia": "Philadelphia, Pennsylvania, USA",
        "Atlanta": "Atlanta, Georgia, USA",
        "Berlin": "Berlin, Germany",
        "Singapore": "Singapore",
        "Moscow": "Moscow, Russia",
        "Cape Town": "Cape Town, South Africa",
        "Dublin": "Dublin, Ireland",
        "Tokyo": "Tokyo, Japan",
        "Bucharest": "Bucharest, Romania",
        "Melbourne": "Melbourne, Australia",
        "Warsaw": "Warsaw, Poland",
        "Taipei": "Taipei, Taiwan",
        "Johannesburg": "Johannesburg, South Africa",
        "Santiago": "Santiago, Chile",
        "Toronto": "Toronto, Canada",
        "Milan": "Milan, Italy",
        "Portland": "Portland, Oregon, USA",
        "Los Angeles": "Los Angeles, California, USA",
        "Bangkok": "Bangkok, Thailand",
        "London": "London, UK",
        "Madrid": "Madrid, Spain",
        "Boston": "Boston, Massachusetts, USA",
        "Houston": "Houston, Texas, USA",
        "Paris": "Paris, France",
        "Belo Horizonte": "Belo Horizonte, Brazil",
        "Rome": "Rome, Italy",
        "Kyoto": "Kyoto, Japan",
        "Chicago": "Chicago, Illinois, USA",
        "Minneapolis": "Minneapolis, Minnesota, USA",
        "Montreal": "Montreal, Quebec, Canada",
        "San Francisco": "San Francisco, California, USA",
        "Lisbon": "Lisbon, Portugal",
        "Guadalajara": "Guadalajara, Mexico",
        "Seattle": "Seattle, Washington, USA",
        "Gaborone": "Gaborone, Botswana",
        "Barcelona": "Barcelona, Spain",
        "Zagreb": "Zagreb, Croatia",
        "Copenhagen": "Copenhagen, Denmark",
        "Sydney": "Sydney, Australia",
        "Valparaiso": "Valparaiso, Chile",
        "Denver": "Denver, Colorado, USA",
        "Munich": "Munich, Germany",
        "Stockholm": "Stockholm, Sweden",
        "Tel Aviv": "Tel Aviv, Israel",
        "Prague": "Prague, Czech Republic",
        "Hong Kong": "Hong Kong",
        "Glasgow": "Glasgow, UK",
        "Bratislava": "Bratislava, Slovakia",
        "Kiev": "Kyiv, Ukraine",
        "Helsinki": "Helsinki, Finland",
        "Zurich": "Zurich, Switzerland",
    }
    return custom_map.get(city_name, city_name)


def get_graph_path(city_name: str) -> Path:
    safe_name = sanitize_city_name(city_name)
    return GRAPHS_DIR / f"{safe_name}.graphml"


def load_or_download_graph(city_name: str):
    graph_path = get_graph_path(city_name)
    if graph_path.exists():
        print(f"Loading cached graph for {city_name}")
        G = ox.load_graphml(graph_path)
        return G
    print(f"Downloading graph for {city_name}")
    place_query = get_place_query(city_name)
    print(f"Place query used for {city_name}: {place_query}")
    G = ox.graph_from_place(place_query, network_type="walk")
    ox.save_graphml(G, graph_path)
    return G


# --------------------------- FEATURE HELPERS ---------------------------

def _safe_first(value):
    """Extract a single scalar from lists/arrays OSMnx may return for edge attributes."""
    import numpy as np
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) > 0 else None
    if isinstance(value, np.ndarray):
        return value.flat[0] if value.size > 0 else None
    return value


def normalize_highway(value):
    value = _safe_first(value)
    if value is None:
        return "missing"
    try:
        if pd.isna(value):
            return "missing"
    except (TypeError, ValueError):
        pass
    value = str(value)
    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in value.split(",")]
        return parts[0] if parts else "missing"
    return value


def normalize_osm_value(value):
    value = _safe_first(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in value.split(",")]
        return parts[0].lower() if parts else None
    return value.lower()


def osm_equals(value, valid_values):
    value = normalize_osm_value(value)
    if value is None:
        return 0
    return 1 if value in valid_values else 0


def is_lit_feature(value):
    value = normalize_osm_value(value)
    lit_positive = {"yes", "24/7", "automatic", "limited", "interval", "dusk-dawn", "sunrise-sunset"}
    return 1 if value in lit_positive else 0


def to_numeric(value, default=0.0):
    value = _safe_first(value)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in value.split(",")]
        value = parts[0] if parts else str(default)
    value = value.split()[0]
    try:
        return float(value)
    except Exception:
        return default


def has_sidewalk_tag(value):
    """Check OSM sidewalk=* tag on the road edge itself."""
    value = normalize_osm_value(value)
    if value is None:
        return 0
    return 1 if value in {"yes", "both", "left", "right", "separate"} else 0


def is_oneway(value):
    value = normalize_osm_value(value)
    if value is None:
        return 0
    return 1 if value in {"yes", "true", "1", "-1", "reversible"} else 0


def surface_to_smoothness(value):
    """
    Convert OSM surface tag to a numeric smoothness score.
    0 = unknown/rough, 1 = medium (cobblestone etc), 2 = smooth (asphalt etc).
    Smoother surfaces correlate with more formal, maintained streets -> safer perception.
    """
    value = normalize_osm_value(value)
    if value is None:
        return 0
    smooth = {"asphalt", "concrete", "paving_stones", "concrete:plates", "concrete:lanes"}
    medium = {"cobblestone", "sett", "unhewn_cobblestone", "metal", "wood", "compacted"}
    rough  = {"gravel", "fine_gravel", "pebblestone", "dirt", "grass", "sand", "mud", "ground"}
    if value in smooth:
        return 2
    if value in medium:
        return 1
    if value in rough:
        return 0
    return 0


# --------------------------- GRAPH FEATURES ---------------------------

def edge_to_features(edge_attrs: dict) -> dict:
    highway  = normalize_highway(edge_attrs.get("highway"))
    tunnel   = edge_attrs.get("tunnel")
    lit      = edge_attrs.get("lit")
    bridge   = edge_attrs.get("bridge")
    footway  = edge_attrs.get("footway")
    oneway   = edge_attrs.get("oneway")
    maxspeed = edge_attrs.get("maxspeed")
    lanes    = edge_attrs.get("lanes")
    surface  = edge_attrs.get("surface")
    width    = edge_attrs.get("width")

    sidewalk       = edge_attrs.get("sidewalk")
    sidewalk_left  = edge_attrs.get("sidewalk:left")
    sidewalk_right = edge_attrs.get("sidewalk:right")
    sidewalk_both  = edge_attrs.get("sidewalk:both")

    is_tunnel_val  = osm_equals(tunnel, {"yes", "building_passage", "covered"})
    is_lit_val     = is_lit_feature(lit)
    is_bridge_val  = osm_equals(bridge, {"yes"})
    is_oneway_val  = is_oneway(oneway)

    footway_sidewalk_val = osm_equals(footway, {"sidewalk"})
    footway_crossing_val = osm_equals(footway, {"crossing"})

    has_sidewalk_val = int(
        has_sidewalk_tag(sidewalk)
        or osm_equals(sidewalk_left,  {"yes", "separate"})
        or osm_equals(sidewalk_right, {"yes", "separate"})
        or osm_equals(sidewalk_both,  {"yes", "separate"})
        or footway_sidewalk_val == 1
    )

    features = {
        "edge_length":        to_numeric(edge_attrs.get("length"), 0.0),
        "is_tunnel":          is_tunnel_val,
        "is_lit":             is_lit_val,
        "is_bridge":          is_bridge_val,
        "is_oneway":          is_oneway_val,
        "has_sidewalk":       has_sidewalk_val,
        "maxspeed":           to_numeric(maxspeed, 0.0),
        "lanes":              to_numeric(lanes, 1.0),
        "width":              to_numeric(width, 0.0),
        "surface_smoothness": surface_to_smoothness(surface),
        "highway_primary":     1 if highway == "primary"     else 0,
        "highway_secondary":   1 if highway == "secondary"   else 0,
        "highway_tertiary":    1 if highway == "tertiary"    else 0,
        "highway_residential": 1 if highway == "residential" else 0,
        "highway_service":     1 if highway == "service"     else 0,
        "highway_footway":     1 if highway == "footway"     else 0,
        "highway_path":        1 if highway == "path"        else 0,
        "footway_sidewalk":   footway_sidewalk_val,
        "footway_crossing":   footway_crossing_val,
    }

    return features


def get_node_context_features(G, u, v):
    degree_u = G.degree[u]
    degree_v = G.degree[v]
    avg_degree = (degree_u + degree_v) / 2
    return {
        "degree_u":   degree_u,
        "degree_v":   degree_v,
        "avg_degree": avg_degree,
        "dead_end":   1 if min(degree_u, degree_v) <= 1 else 0,
    }


def get_nearest_edge_features(G, lat: float, lon: float) -> dict:
    u, v, k = ox.distance.nearest_edges(G, lon, lat)
    edge_attrs = G.get_edge_data(u, v, k)
    features = edge_to_features(edge_attrs)
    node_context = get_node_context_features(G, u, v)
    features.update(node_context)
    return features


# --------------------------- MAIN FEATURE BUILDING ---------------------------

# Null-feature dict for when extraction fails — must stay in sync with edge_to_features()
NULL_FEATS = {
    "edge_length":         None,
    "is_tunnel":           None,
    "is_lit":              None,
    "is_bridge":           None,
    "is_oneway":           None,
    "has_sidewalk":        None,
    "maxspeed":            None,
    "lanes":               None,
    "width":               None,
    "surface_smoothness":  None,
    "highway_primary":     None,
    "highway_secondary":   None,
    "highway_tertiary":    None,
    "highway_residential": None,
    "highway_service":     None,
    "highway_footway":     None,
    "highway_path":        None,
    "footway_sidewalk":    None,
    "footway_crossing":    None,
    "degree_u":            None,
    "degree_v":            None,
    "avg_degree":          None,
    "dead_end":            None,
}


def build_feature_table_for_city(city_df: pd.DataFrame) -> pd.DataFrame:
    city_name = city_df["city_name"].iloc[0]
    print(f"\nProcessing city: {city_name} | rows: {len(city_df)}")

    G = load_or_download_graph(city_name)

    lons = city_df["lon"].tolist()
    lats = city_df["lat"].tolist()

    edge_matches = ox.distance.nearest_edges(G, X=lons, Y=lats)

    feature_rows = []

    for row, edge_match in zip(city_df.itertuples(index=False), edge_matches):
        try:
            u, v, k = edge_match
            edge_attrs = G.get_edge_data(u, v, k)
            feats = edge_to_features(edge_attrs)
            node_context = get_node_context_features(G, u, v)
            feats.update(node_context)
        except Exception as e:
            print(f"Feature extraction failed for {city_name}, image_id={row.image_id}: {e}")
            feats = NULL_FEATS.copy()

        feature_rows.append({"image_id": row.image_id, **feats})

    features_df = pd.DataFrame(feature_rows)
    city_full_df = city_df.merge(features_df, on="image_id", how="left")
    print(f"{city_name}: nodes={len(G.nodes)}, edges={len(G.edges)}")
    return city_full_df


def build_feature_table_all_cities(training_base: pd.DataFrame) -> pd.DataFrame:
    city_dfs = []

    for city_name, city_df in training_base.groupby("city_name"):
        if city_name in ["Taipei", "Tokyo"]:
            print(f"Skipping {city_name} for now")
            continue
        try:
            city_result = build_feature_table_for_city(city_df.copy())
            city_dfs.append(city_result)
        except Exception as e:
            print(f"Skipping city {city_name} due to error: {e}")

    if not city_dfs:
        return pd.DataFrame()

    return pd.concat(city_dfs, ignore_index=True)


def save_features_csv(df: pd.DataFrame, filename: str) -> None:
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


# --------------------------- RUN ---------------------------

if __name__ == "__main__":
    training_base_df = load_training_base()
    print("Training base loaded:", training_base_df.shape)
    print("Unique cities:", training_base_df["city_name"].nunique())

    feature_df = build_feature_table_all_cities(training_base_df)

    print("\nFeature table shape:", feature_df.shape)
    print("\nFeature table head:")
    print(feature_df.head())

    print("\nMissing values per column:")
    print(feature_df.isna().sum())

    save_features_csv(feature_df, "feature_full_multicity.csv")
    