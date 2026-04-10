import pandas as pd
df = pd.read_csv("Data/processed/feature_full_multicity.csv")
print(df[["is_lit", "has_sidewalk", "surface_smoothness", "width"]].mean())
print(df[["is_lit", "has_sidewalk", "surface_smoothness", "width"]].describe())