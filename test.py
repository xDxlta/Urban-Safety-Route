from src.data_loading import load_votes
import osmnx as ox

G = ox.graph_from_place("Zurich, Switzerland", network_type="walk")
ox.save_graphml(G, "zurich_walk.graphml")
df = load_votes()

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())