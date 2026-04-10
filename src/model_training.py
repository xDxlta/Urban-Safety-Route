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
 
# --------------------------- PATHS ---------------------------
 
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
 
# --------------------------- LOAD & FILTER ---------------------------
 
df = pd.read_csv(PROCESSED_DIR / "feature_context_filtered.csv")
 
# Only European cities structurally closest to Zürich
european_cities = [
    "Amsterdam", "Barcelona", "Berlin", "Bratislava", "Bucharest",
    "Copenhagen", "Dublin", "Helsinki", "Kiev", "Lisbon",
    "London", "Madrid", "Milan", "Moscow", "Munich", "Paris",
    "Prague", "Rome", "Stockholm", "Warsaw", "Zagreb",
]
df = df[df["city_name"].isin(european_cities)].copy()
print(f"Dataset shape after city filter: {df.shape}")
 
# --------------------------- FEATURE ENGINEERING ---------------------------
 
df["log_dist_park"]    = np.log1p(df["dist_to_park"])
df["log_dist_station"] = np.log1p(df["dist_to_station"])
df["poi_density_300m"] = df["poi_count_300m"] / (np.pi * (300 ** 2))
 
# Score distribution
df["elo_score"].hist(bins=50)
plt.title("ELO Score Distribution")
plt.savefig("score_hist.png")
plt.close()
 
# --------------------------- FEATURES & TARGET ---------------------------
 
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
]
target_col = "elo_score"
 
model_df = df[feature_cols + [target_col, "city_name"]].dropna().copy()
print(f"Model dataset shape: {model_df.shape}")
print(f"Missing values:\n{model_df[feature_cols].isna().sum()}")
 
X = model_df[feature_cols]
y = model_df[target_col]
 
# --------------------------- TRAIN / TEST SPLIT ---------------------------
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")
 
# --------------------------- DUMMY BASELINE ---------------------------
 
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
 
print("\n--- Dummy (mean) ---")
print(f"MAE:  {mean_absolute_error(y_test, dummy_pred):.5f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, dummy_pred)):.5f}")
print(f"R2:   {r2_score(y_test, dummy_pred):.5f}")
 
# --------------------------- XGBOOST ---------------------------
 
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
 
# Residual plot
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