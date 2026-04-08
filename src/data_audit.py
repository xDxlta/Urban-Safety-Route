from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"

# -------------------------
# LOAD DATA
# -------------------------
feature_path = PROCESSED_DIR / "feature_full_multicity.csv"
context_path = PROCESSED_DIR / "context_features_multicity.csv"

feature_df = pd.read_csv(feature_path)
context_df = pd.read_csv(context_path)

print("Feature df shape:", feature_df.shape)
print("Context df shape:", context_df.shape)

# Merge context onto feature table
df = feature_df.merge(context_df, on="image_id", how="left")

print("Merged df shape:", df.shape)

# -------------------------
# DEFINE FEATURE GROUPS
# -------------------------
basic_feature_cols = [
    "edge_length",
    "is_tunnel",
    "is_lit",
    "is_bridge",
    "is_oneway",
    "has_sidewalk",
    "maxspeed",
    "lanes",
    "highway_primary",
    "highway_secondary",
    "highway_tertiary",
    "highway_residential",
    "highway_service",
    "highway_footway",
    "highway_path",
    "footway_sidewalk",
    "footway_crossing",
    "degree_u",
    "degree_v",
    "avg_degree",
    "dead_end",
]

context_feature_cols = [
    "dist_to_park",
    "dist_to_station",
    "near_park",
    "near_station",
    "poi_count_300m",
]

all_feature_cols = basic_feature_cols + context_feature_cols

# keep only columns that actually exist
basic_feature_cols = [c for c in basic_feature_cols if c in df.columns]
context_feature_cols = [c for c in context_feature_cols if c in df.columns]
all_feature_cols = [c for c in all_feature_cols if c in df.columns]

# -------------------------
# CITY-LEVEL AUDIT
# -------------------------
rows = []

for city_name, city_df in df.groupby("city_name"):
    row = {
        "city_name": city_name,
        "n_rows": len(city_df),
    }

    # basic feature missingness
    if basic_feature_cols:
        row["basic_missing_share"] = city_df[basic_feature_cols].isna().mean().mean()
    else:
        row["basic_missing_share"] = np.nan

    # context feature missingness
    if context_feature_cols:
        row["context_missing_share"] = city_df[context_feature_cols].isna().mean().mean()
    else:
        row["context_missing_share"] = np.nan

    # total missingness
    if all_feature_cols:
        row["all_missing_share"] = city_df[all_feature_cols].isna().mean().mean()
    else:
        row["all_missing_share"] = np.nan

    # suspicious zero shares for important cols
    for col in ["is_lit", "has_sidewalk", "footway_sidewalk", "footway_crossing", "dead_end"]:
        if col in city_df.columns:
            row[f"{col}_zero_share"] = (city_df[col].fillna(0) == 0).mean()

    # context coverage
    if "dist_to_park" in city_df.columns:
        row["dist_to_park_missing_share"] = city_df["dist_to_park"].isna().mean()
        row["dist_to_park_mean"] = city_df["dist_to_park"].mean()

    if "dist_to_station" in city_df.columns:
        row["dist_to_station_missing_share"] = city_df["dist_to_station"].isna().mean()
        row["dist_to_station_mean"] = city_df["dist_to_station"].mean()

    if "poi_count_300m" in city_df.columns:
        row["poi_count_300m_missing_share"] = city_df["poi_count_300m"].isna().mean()
        row["poi_count_300m_mean"] = city_df["poi_count_300m"].mean()

    # target stats
    if "score" in city_df.columns:
        row["score_mean"] = city_df["score"].mean()
        row["score_std"] = city_df["score"].std()

    if "elo_score" in city_df.columns:
        row["elo_score_mean"] = city_df["elo_score"].mean()
        row["elo_score_std"] = city_df["elo_score"].std()

    rows.append(row)

audit_df = pd.DataFrame(rows).sort_values("city_name").reset_index(drop=True)

# -------------------------
# FLAG SUSPICIOUS CITIES
# -------------------------
audit_df["flag_high_basic_missing"] = audit_df["basic_missing_share"] > 0.05
audit_df["flag_high_context_missing"] = audit_df["context_missing_share"] > 0.05
audit_df["flag_high_total_missing"] = audit_df["all_missing_share"] > 0.05

# Optional strong flags
for col in ["is_lit_zero_share", "has_sidewalk_zero_share"]:
    if col in audit_df.columns:
        audit_df[f"flag_{col}"] = audit_df[col] > 0.95

print("\n=== CITY AUDIT TABLE ===")
print(audit_df)

# -------------------------
# SAVE AUDIT
# -------------------------
audit_path = PROCESSED_DIR / "city_data_audit.csv"
audit_df.to_csv(audit_path, index=False)
print(f"\nSaved city audit to: {audit_path}")

# -------------------------
# PRINT MOST SUSPICIOUS CITIES
# -------------------------
print("\n=== CITIES WITH HIGH BASIC MISSING ===")
print(audit_df.loc[audit_df["flag_high_basic_missing"], ["city_name", "n_rows", "basic_missing_share"]])

print("\n=== CITIES WITH HIGH CONTEXT MISSING ===")
print(audit_df.loc[audit_df["flag_high_context_missing"], ["city_name", "n_rows", "context_missing_share"]])

print("\n=== CITIES WITH HIGH TOTAL MISSING ===")
print(audit_df.loc[audit_df["flag_high_total_missing"], ["city_name", "n_rows", "all_missing_share"]])

# -------------------------
# QUICK FILTERED DATASET TEST PREP
# -------------------------
# Cities you already suspect:
manual_bad_cities = [
    "Copenhagen",
    "Santiago",
    "Cape Town",
    "Glasgow",
    "Johannesburg",
    "Hong Kong",
    "Taipei",
    "Tokyo",
    "Valparaiso",
]

filtered_df = df[~df["city_name"].isin(manual_bad_cities)].copy()

filtered_path = PROCESSED_DIR / "feature_context_filtered.csv"
filtered_df.to_csv(filtered_path, index=False)

print(f"\nFiltered dataset shape (without suspected bad cities): {filtered_df.shape}")
print(f"Saved filtered dataset to: {filtered_path}")