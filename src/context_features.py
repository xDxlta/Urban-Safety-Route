from pathlib import Path
import re
import pandas as pd
import osmnx as ox
import geopandas as gpd

#This is part 2 of feature engineering, where we added new features, we didnt add during the first run. For shorter waiting times, we didnt delete our downloaded OSM data, but redownload the new features and added them to te rest
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
CITY_CONTEXT_DIR = PROCESSED_DIR / "context_city_files"
CITY_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

def load_feature_table() -> pd.DataFrame:
    file_path = PROCESSED_DIR / "feature_full_multicity.csv"
    return pd.read_csv(file_path)

#we start the same as in feature engineering
def sanitize_city_name(city_name: str) -> str:
    city_name = str(city_name).strip()
    city_name = re.sub(r"[^A-Za-z0-9]+", "_", city_name)
    return city_name.strip("_")
#equivalent to the function get_graph_path from feature engineering
def get_city_context_path(city_name: str) -> Path:
    safe_name = sanitize_city_name(city_name)
    return CITY_CONTEXT_DIR / f"{safe_name}_context.csv"


def load_existing_city_context(city_name: str) -> pd.DataFrame | None:
    path = get_city_context_path(city_name)
    if path.exists():
        print(f"Loading cached context CSV for {city_name}")
        return pd.read_csv(path)
    return None

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

#transforms the dataframe with the coords in a geodataframe
def get_city_points_gdf(city_df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        city_df.copy(),
        geometry=gpd.points_from_xy(city_df["lon"], city_df["lat"]),
        crs="EPSG:4326" #thats the coordinate system used by osm
    )
    return gdf

#We just download some context features that could be helpful like parks or train stations
def get_osm_context_layers(city_name: str):
    place_query = get_place_query(city_name)
    print(f"Downloading OSM context features for {city_name} using query: {place_query}")

    park_tags = {
        "leisure": "park",
        "landuse": "park",
    }

    station_tags = {
        "railway": ["station", "halt", "tram_stop", "subway_entrance"]
    }

    poi_tags = {
        "amenity": True,
        "shop": True,
        "public_transport": True,
    }

    try: #get the parks, stations and pois
        parks = ox.features_from_place(place_query, park_tags)
    except Exception as e: #return empty gdf if there are no parks, stations and pois
        print(f"No parks for {city_name}: {e}")
        parks = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    try:
        stations = ox.features_from_place(place_query, station_tags)
    except Exception as e:
        print(f"No stations for {city_name}: {e}")
        stations = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    try:
        pois = ox.features_from_place(place_query, poi_tags)
    except Exception as e:
        print(f"No POIs for {city_name}: {e}")
        pois = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return parks, stations, pois


def prepare_context_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

#kick out broken geometries
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.is_empty == False].copy()

#in case everything got kicked out now:
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

#only keep geometry not useless data like shopowner or parkname
    gdf = gdf[["geometry"]].copy()
    gdf = gdf.to_crs(epsg=3857) #transform to metric system for distance calculations. Otherwise gps normally uses degree system i think?

    # parks are usually saved as polynoms, stations can be lines or polygons I think. So we take the center of these so we have one point. This means that the distance to a park might be in reality shorter, but this way its easier to calculate
    gdf["geometry"] = gdf.geometry.centroid
    gdf = gdf.reset_index(drop=True)

    return gdf


def add_context_features_for_city(city_df: pd.DataFrame) -> pd.DataFrame:
    city_name = city_df["city_name"].iloc[0]
    print(f"\nProcessing context features for {city_name} | rows: {len(city_df)}")

    # instantly load when context already exist
    existing = load_existing_city_context(city_name)
    if existing is not None:
        return existing

    points_gdf = gpd.GeoDataFrame(
        city_df.copy(),
        geometry=gpd.points_from_xy(city_df["lon"], city_df["lat"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857) #from gps convert to metric system for distance calculations (I used chatGPT for the exact epsg numbers)

    points_gdf = points_gdf.reset_index(drop=True)
    points_gdf["point_id"] = points_gdf.index #changes the index to point_id

    parks, stations, pois = get_osm_context_layers(city_name)

    parks_gdf = prepare_context_gdf(parks)
    stations_gdf = prepare_context_gdf(stations)
    pois_gdf = prepare_context_gdf(pois)

    points_gdf["dist_to_park"] = 9999.0
    points_gdf["dist_to_station"] = 9999.0
    points_gdf["near_park"] = 0
    points_gdf["near_station"] = 0
    points_gdf["poi_count_300m"] = 0

    #The following will be done for parks, stations and pois, but I will only comment the first one
    if not parks_gdf.empty: #only do if there are parks, otherwise let the distance remain at 9999 
        joined_parks = gpd.sjoin_nearest( #for every point look for the nearest park and add distance
            points_gdf[["point_id", "geometry"]],
            parks_gdf[["geometry"]],
            how="left",
            distance_col="dist_to_park"
        )
        joined_parks = joined_parks.sort_values("dist_to_park").drop_duplicates(subset="point_id") #if multiple parks are nearby keep only the closest one and if multiple have the same distance take the first one
        park_dist_map = joined_parks.set_index("point_id")["dist_to_park"]
        points_gdf["dist_to_park"] = points_gdf["point_id"].map(park_dist_map).fillna(9999.0) #adds the distance to the nearest park to the points_gdf and if there is no park nearby fill it with 9999.0
        points_gdf["near_park"] = (points_gdf["dist_to_park"] <= 300).astype(int) #we used 300 meters as threshold for being near a park, if the distance is smaller than 300m we set near_park to 1 otherwise to 0

    if not stations_gdf.empty:
        joined_stations = gpd.sjoin_nearest(
            points_gdf[["point_id", "geometry"]],
            stations_gdf[["geometry"]],
            how="left",
            distance_col="dist_to_station"
        )
        joined_stations = joined_stations.sort_values("dist_to_station").drop_duplicates(subset="point_id")
        station_dist_map = joined_stations.set_index("point_id")["dist_to_station"]
        points_gdf["dist_to_station"] = points_gdf["point_id"].map(station_dist_map).fillna(9999.0)
        points_gdf["near_station"] = (points_gdf["dist_to_station"] <= 300).astype(int)

    if not pois_gdf.empty:
        poi_join = gpd.sjoin_nearest(
            points_gdf[["point_id", "geometry"]],
            pois_gdf[["geometry"]],
            how="left",
            distance_col="poi_dist"
        )
        poi_join = poi_join[poi_join["poi_dist"] <= 300].copy()
        poi_counts = poi_join.groupby("point_id").size()
        points_gdf["poi_count_300m"] = points_gdf["point_id"].map(poi_counts).fillna(0).astype(int)

    result_cols = [
        "image_id",
        "dist_to_park",
        "dist_to_station",
        "near_park",
        "near_station",
        "poi_count_300m",
    ]

    result_df = pd.DataFrame(points_gdf[result_cols])

    # immediatly safe the context features for the city. Downlaoding takes a while and I couldnt have my pc running that long with stable internet so I wanted to safe in between and not loose everything
    out_path = get_city_context_path(city_name)
    result_df.to_csv(out_path, index=False)
    print(f"Saved city context: {out_path}")

    return result_df

#This works the same as in feature engineering 
def build_context_features_all_cities(df: pd.DataFrame) -> pd.DataFrame:
    city_results = []

    for city_name, city_df in df.groupby("city_name"):
        # Skip problem cities again
        if city_name in ["Taipei", "Tokyo", "Hong Kong", "Valparaiso"]:
            print(f"Skipping {city_name} for now")
            continue
        
        try:
            context_df = add_context_features_for_city(city_df.copy())
            city_results.append(context_df)
        except Exception as e:
            print(f"Skipping context features for {city_name} due to error: {e}")

    if not city_results:
        return pd.DataFrame()

    return pd.concat(city_results, ignore_index=True)


def save_context_csv(df: pd.DataFrame, filename: str):
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    feature_df = load_feature_table()
    print("Loaded feature table:", feature_df.shape)

    context_df = build_context_features_all_cities(feature_df)

    print("\nContext feature table shape:", context_df.shape)
    print(context_df.head())

    save_context_csv(context_df, "context_features_multicity.csv")