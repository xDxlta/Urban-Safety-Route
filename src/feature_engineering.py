from pathlib import Path
import re
import pandas as pd
import osmnx as ox


#This code is responsible for taking the training base and building a feature table by looking up the nearest OSM edge for each point and extracting features from that edge and its nodes. The resulting feature table will be saved as a CSV for use in model training.
# Path stuff again

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
GRAPHS_DIR = BASE_DIR / "Data" / "graphs"

GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

def load_training_base() -> pd.DataFrame:
    file_path = PROCESSED_DIR / "training_base.csv"
    df = pd.read_csv(file_path)
    return df


# --------------------------- CITY / GRAPH HELPERS ---------------------------

#Turn city names into safe filenames.
def sanitize_city_name(city_name: str) -> str:
    city_name = str(city_name).strip()
    city_name = re.sub(r"[^A-Za-z0-9]+", "_", city_name)
    return city_name.strip("_")



# I really dont know why OSMnx has such inconsistent place query requirements across cities, so we need this mapping to get the correct graph for some cities. For most cities, the name alone works fine, but somehow OSM thinks New York and Amsterdam are the same place or something
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
        

    }

    return custom_map.get(city_name, city_name)

#Download the graph for a city if we dont have it cached, otherwise load from disk. Caching is important because OSMnx can be very slow to download and process graphs, especially for large cities. We save the graphs in GraphML format which preserves all the attributes we need.
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
       

def normalize_highway(value):
    if pd.isna(value):
        return "missing"

    value = str(value)

    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in value.split(",")]
        return parts[0] if parts else "missing"

    return value

def normalize_osm_value(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    # Falls OSMnx/Writing Listen als String speichert: "['yes', 'limited']"
    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in value.split(",")]
        return parts[0].lower() if parts else None

    return value.lower()


def is_lit_feature(value):
    value = normalize_osm_value(value)

    # OSM lit-Werte, die echte Beleuchtung anzeigen
    lit_positive = {"yes", "24/7", "automatic", "limited", "interval"}

    return 1 if value in lit_positive else 0

def to_numeric(value, default=0.0):
    if pd.isna(value):
        return default

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in value.split(",")]
        value = parts[0] if parts else str(default)

    value = value.split()[0]

    try:
        return float(value)
    except:
        return default


def has_sidewalk(value):
    if pd.isna(value):
        return 0

    value = str(value).lower()
    return 1 if value in ["yes", "both", "left", "right", "separate"] else 0


def is_oneway(value):
    if pd.isna(value):
        return 0

    value = str(value).lower()
    return 1 if value in ["yes", "true", "1"] else 0


def edge_to_features(edge_attrs: dict) -> dict:
    highway = normalize_highway(edge_attrs.get("highway"))
    tunnel = edge_attrs.get("tunnel")
    lit = edge_attrs.get("lit")
    bridge = edge_attrs.get("bridge")
    footway = edge_attrs.get("footway")
    sidewalk = edge_attrs.get("sidewalk")
    oneway = edge_attrs.get("oneway")
    maxspeed = edge_attrs.get("maxspeed")
    lanes = edge_attrs.get("lanes")

    features = {
        "edge_length": to_numeric(edge_attrs.get("length"), 0.0),

        "is_tunnel": 1 if str(tunnel) in ["yes", "building_passage", "covered"] else 0,
        "is_lit": is_lit_feature(lit),
        "is_bridge": 1 if str(bridge) == "yes" else 0,
        "is_oneway": is_oneway(oneway),

        "has_sidewalk": has_sidewalk(sidewalk),
        "maxspeed": to_numeric(maxspeed, 0.0),
        "lanes": to_numeric(lanes, 1.0),

        "highway_primary": 1 if highway == "primary" else 0,
        "highway_secondary": 1 if highway == "secondary" else 0,
        "highway_tertiary": 1 if highway == "tertiary" else 0,
        "highway_residential": 1 if highway == "residential" else 0,
        "highway_service": 1 if highway == "service" else 0,
        "highway_footway": 1 if highway == "footway" else 0,
        "highway_path": 1 if highway == "path" else 0,

        "footway_sidewalk": 1 if str(footway) == "sidewalk" else 0,
        "footway_crossing": 1 if str(footway) == "crossing" else 0,
    }

    return features


def get_node_context_features(G, u, v):
    degree_u = G.degree[u]
    degree_v = G.degree[v]
    avg_degree = (degree_u + degree_v) / 2

    return {
        "degree_u": degree_u,
        "degree_v": degree_v,
        "avg_degree": avg_degree,
        "dead_end": 1 if min(degree_u, degree_v) <= 1 else 0,
    }


def get_nearest_edge_features(G, lat: float, lon: float) -> dict:
    u, v, k = ox.distance.nearest_edges(G, lon, lat)
    edge_attrs = G.get_edge_data(u, v, k)

    features = edge_to_features(edge_attrs)
    node_context = get_node_context_features(G, u, v)

    features.update(node_context)
    return features


# --------------------------- MAIN FEATURE BUILDING ---------------------------

def build_feature_table_for_city(city_df: pd.DataFrame) -> pd.DataFrame:
    city_name = city_df["city_name"].iloc[0]
    print(f"\nProcessing city: {city_name} | rows: {len(city_df)}")

    G = load_or_download_graph(city_name)

    lons = city_df["lon"].tolist()
    lats = city_df["lat"].tolist()

    # Batch nearest edge lookup (much faster than looping nearest_edges row by row)
    edge_matches = ox.distance.nearest_edges(G, X=lons, Y=lats)

    feature_rows = []

    for row, edge_match in zip(city_df.itertuples(index=False), edge_matches):
        try:
            u, v, k = edge_match
            edge_attrs = G.get_edge_data(u, v, k)

            feats = edge_to_features(edge_attrs)
            node_context = get_node_context_features(G, u, v)
            feats.update(node_context)

        except Exception:
            feats = {
                "edge_length": None,
                "is_tunnel": None,
                "is_lit": None,
                "is_bridge": None,
                "is_oneway": None,
                "has_sidewalk": None,
                "maxspeed": None,
                "lanes": None,
                "highway_primary": None,
                "highway_secondary": None,
                "highway_tertiary": None,
                "highway_residential": None,
                "highway_service": None,
                "highway_footway": None,
                "highway_path": None,
                "footway_sidewalk": None,
                "footway_crossing": None,
                "degree_u": None,
                "degree_v": None,
                "avg_degree": None,
                "dead_end": None,
            }

        feature_rows.append({
            "image_id": row.image_id,
            **feats
        })

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

    full_df = pd.concat(city_dfs, ignore_index=True)
    return full_df


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
    