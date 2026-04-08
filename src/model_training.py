from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"

file_path = PROCESSED_DIR / "feature_context_filtered.csv"
df = pd.read_csv(file_path)
df = df[~df["city_name"].isin([
    "Boston",
    "Helsinki",
    "Stockholm",
    "Singapore",
    "Prague",
    "Bratislava",
    "Zagreb"
])].copy()

print(df.shape)
print(df.head())

# Dann neue Features bauen
df["log_dist_park"] = np.log1p(df["dist_to_park"])
df["log_dist_station"] = np.log1p(df["dist_to_station"])
df["poi_density_300m"] = df["poi_count_300m"] / (np.pi * (300 ** 2))

df["score"].hist(bins=50)
plt.savefig("score_hist.png")
plt.close()

#Use a small set of features first to test the model
feature_cols = [
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
    "log_dist_park",
    "log_dist_station",
    "near_park",
    "near_station",
    "poi_density_300m",
]
target_col = "score"

#Drop rows with missing values in the selected features and target
model_df = df[feature_cols + [target_col, "city_name"]].dropna().copy()

print(model_df.shape)
print(model_df[feature_cols].isna().sum())

#Build X and y
X = model_df[feature_cols]
y = model_df[target_col]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train a simple Random Forest Regressor as a baseline model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

#Feature importance
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(feature_importance)

import numpy as np

dummy_pred = np.full(len(y_test), y_train.mean())

dummy_mae = mean_absolute_error(y_test, dummy_pred)
dummy_rmse = np.sqrt(mean_squared_error(y_test, dummy_pred))
dummy_r2 = r2_score(y_test, dummy_pred)

print("\nDummy model:")
print("MAE:", dummy_mae)
print("RMSE:", dummy_rmse)
print("R2:", dummy_r2)
print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

lin_pred = lin_model.predict(X_test)

lin_mae = mean_absolute_error(y_test, lin_pred)
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))
lin_r2 = r2_score(y_test, lin_pred)

print("\nLinear Regression:")
print("MAE:", lin_mae)
print("RMSE:", lin_rmse)
print("R2:", lin_r2)

from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)



xgb_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nXGBoost feature importances:")
print(xgb_importance)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost:")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R2:", xgb_r2)