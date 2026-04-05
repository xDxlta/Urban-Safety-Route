import osmnx as ox
from pathlib import Path

# Directory for Data
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

# folder
DATA_DIR.mkdir(exist_ok=True)

# load graph for Zurich
G = ox.graph_from_place("Zurich, Switzerland", network_type="walk")

# save graph to file
file_path = DATA_DIR / "zurich_walk.graphml"
ox.save_graphml(G, file_path)

print("Saved in:")
print(file_path)