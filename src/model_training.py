from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
 
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
 
df = pd.read_csv(PROCESSED_DIR / "feature_context_filtered.csv")
 
# We tested only using european cities closest to Zurich, as well as wrote a code that tells us which cities improve the R2 and which make it worse, but both models performed worse then all cities (excluding the broken ones below)
# Exclude cities with data problems (broke down while downloading or only few kb)
exclude_cities = ["Copenhagen", "Santiago", "Valparaiso", "Hong Kong"
]
df = df[~df["city_name"].isin(exclude_cities)].copy()

print(f"Dataset shape after city filter: {df.shape}")
 
 #log transaform distances, and density. WHY DO WE DO THIS HERE AND NOT IN CONTEXT FEATURES?
df["log_dist_park"]    = np.log1p(df["dist_to_park"])
df["log_dist_station"] = np.log1p(df["dist_to_station"])
df["poi_density_300m"] = df["poi_count_300m"] / (np.pi * (300 ** 2))
 
# Score distribution as plot for visualization reasons, you can find it in the folder
df["elo_score"].hist(bins=50)
plt.title("ELO Score Distribution")
plt.savefig("score_hist.png")
plt.close()

# We tried interactions features (Interaktionsfeatures) to improve the model but they didnt helped a lot
df["lit_and_sidewalk"]     = df["is_lit"] * df["has_sidewalk"]
df["residential_sidewalk"] = df["highway_residential"] * df["has_sidewalk"]
df["footway_lit"]          = df["highway_footway"] * df["is_lit"]
df["road_capacity"]        = df["maxspeed"] * df["lanes"]
df["smooth_and_lit"]       = df["surface_smoothness"] * df["is_lit"]
df["smooth_and_sidewalk"]  = df["surface_smoothness"] * df["has_sidewalk"]
df["busy_road"]            = df["highway_primary"] + df["highway_secondary"] + df["highway_tertiary"]
df["pedestrian_infra"]     = df["highway_footway"] + df["highway_path"] + df["has_sidewalk"]
 
feature_cols = [
    "edge_length",
    "is_tunnel",
    "is_lit",
    "is_bridge",
    "is_oneway",
    "has_sidewalk",
    "maxspeed",
    "lanes",
    "width",
    "surface_smoothness",
    "highway_primary",
    "highway_secondary",
    "highway_tertiary",
    "highway_residential",
    "highway_service",
    "highway_footway",
    "highway_path",
    "degree_u",
    "degree_v",
    "avg_degree",
    "dead_end",
    "log_dist_park",
    "log_dist_station",
    "near_park",
    "near_station",
    "poi_density_300m",
    "lit_and_sidewalk",
    "residential_sidewalk",
    "footway_lit",
    "road_capacity",
    "smooth_and_lit",
    "smooth_and_sidewalk",
    "busy_road",
    "pedestrian_infra",
]
target_col = "elo_score"
 
model_df = df[feature_cols + [target_col, "city_name"]].dropna().copy()
print(f"Model dataset shape: {model_df.shape}")
print(f"Missing values:\n{model_df[feature_cols].isna().sum()}")
 
X = model_df[feature_cols]
y = model_df[target_col]
 
#Train test split. we keep 20% for testing
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

 
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
 
print("\n--- Dummy (mean) ---")
print(f"MAE:  {mean_absolute_error(y_test, dummy_pred):.5f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, dummy_pred)):.5f}")
print(f"R2:   {r2_score(y_test, dummy_pred):.5f}")

#We tried random forest, logistic regression and xgboost but xgboost outperformed the other models (which makes sense) so we only continued with this one
 
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
 
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)
 
xgb_pred = xgb_model.predict(X_test)
 
xgb_mae  = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2   = r2_score(y_test, xgb_pred)
 
print("\n--- XGBoost ---")
print(f"MAE:  {xgb_mae:.5f}")
print(f"RMSE: {xgb_rmse:.5f}")
print(f"R2:   {xgb_r2:.5f}")
 
# Feature importances
xgb_importance = pd.DataFrame({
    "feature":    feature_cols,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)
 
print("\nXGBoost feature importances:")
print(xgb_importance.to_string())
 
# Save feature importances
xgb_importance.to_csv(PROCESSED_DIR / "xgb_feature_importance.csv", index=False)
print(f"\nSaved feature importances to: {PROCESSED_DIR / 'xgb_feature_importance.csv'}")
 
# Save model predictions for analysis
results_df = model_df[["city_name", target_col]].copy().loc[X_test.index]
results_df["predicted"] = xgb_pred
results_df["residual"]  = results_df[target_col] - results_df["predicted"]
results_df.to_csv(PROCESSED_DIR / "xgb_predictions.csv", index=False)
print(f"Saved predictions to: {PROCESSED_DIR / 'xgb_predictions.csv'}")
 
# Residual plot (doesnt look good)
plt.figure(figsize=(8, 5))
plt.scatter(results_df[target_col], results_df["predicted"], alpha=0.3, s=5)
plt.plot([0, 1], [0, 1], "r--", linewidth=1)
plt.xlabel("Actual ELO Score")
plt.ylabel("Predicted ELO Score")
plt.title(f"XGBoost: Actual vs Predicted (R²={xgb_r2:.3f})")
plt.tight_layout()
plt.savefig("xgb_actual_vs_predicted.png", dpi=150)
plt.close()
print("Saved plot: xgb_actual_vs_predicted.png")


#I dont know what brings the R2 down right now, so I check which cities improve it and which cities dont. I assume that these cities have either very different infrastructure (e.g. many tunnels) or very bad data quality 
# Leave one out city analysis
#Fix them on european testcities similar to Zurich 
test_cities_fixed = ["Amsterdam", "Barcelona", "Prague", "Stockholm"]

test_mask  = model_df["city_name"].isin(test_cities_fixed)
train_mask_base = ~test_mask

X_loo_test_fixed = model_df.loc[test_mask, feature_cols]
y_loo_test_fixed = model_df.loc[test_mask, target_col]

print("\nLeave-One-City-Out Analyse (Test: europäische Städte)")
loo_results = []

for city in sorted(model_df["city_name"].unique()):
    if city in test_cities_fixed:
        continue  # never take test cities out
    
    train_mask = train_mask_base & (model_df["city_name"] != city)
    X_loo_train = model_df.loc[train_mask, feature_cols]
    y_loo_train = model_df.loc[train_mask, target_col]

    loo_model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    loo_model.fit(X_loo_train, y_loo_train)
    loo_pred = loo_model.predict(X_loo_test_fixed)
    loo_r2 = r2_score(y_loo_test_fixed, loo_pred)

    loo_results.append({
        "city_left_out": city,
        "n_rows": train_mask.sum(),
        "r2_without_city": loo_r2,
    })
    print(f"  Without {city:<20}: R2 = {loo_r2:.5f}")

baseline_loo = r2_score(
    y_loo_test_fixed,
    XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                 subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                 reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
    ).fit(model_df.loc[train_mask_base, feature_cols],
          model_df.loc[train_mask_base, target_col]
    ).predict(X_loo_test_fixed)
)

loo_df = pd.DataFrame(loo_results)
loo_df["r2_delta"] = loo_df["r2_without_city"] - baseline_loo
loo_df = loo_df.sort_values("r2_delta")

print(f"\nBaseline R2 auf europäischen Teststädten: {baseline_loo:.5f}")
print("\nStädte die das Modell auf europäischen Städten verbessern (positives delta = rauslassen hilft):")
print(loo_df[loo_df["r2_delta"] > 0][["city_left_out", "r2_without_city", "r2_delta"]].to_string())
print("\nStädte die das Modell auf europäischen Städten verschlechtern (negatives delta = behalten hilft):")
print(loo_df[loo_df["r2_delta"] < 0][["city_left_out", "r2_without_city", "r2_delta"]].to_string())

loo_df.to_csv(PROCESSED_DIR / "loo_city_analysis_european_test.csv", index=False)

import joblib

joblib.dump(xgb_model, PROCESSED_DIR / "xgb_safety_model.pkl")
joblib.dump(feature_cols, PROCESSED_DIR / "xgb_feature_cols.pkl")
print("Model saved.")