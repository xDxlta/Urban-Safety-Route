from src.data_loading import load_votes

df = load_votes()

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())