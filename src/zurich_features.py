from pathlib import Path
import pandas as pd
import sys

# Add the parent directory to the path to import feature_engineering, so we can use the functions we defined there for zurichs feature extraction
sys.path.insert(0, str(Path(__file__).resolve().parent))

#A friend told me about this trick, we can import the functions from our programs, like they are packages
from feature_engineering import (
    load_or_download_graph,
    edge_to_features,
    get_node_context_features,
    NULL_FEATS,
)
import osmnx as ox
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"

def build_zurich_edge_features():
    print("Loading Zürich graph...")
    G = load_or_download_graph("Zurich")
    #load the graph for zurich
    print(f"Zürich: nodes={len(G.nodes)}, edges={len(G.edges)}")
    
    # We exctract the features for each edge. We dont have coordinates we map onto the graph anymore, but we need the features for every edge, so we can predict all the safety scores
    rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        try:
            feats = edge_to_features(data)
            node_ctx = get_node_context_features(G, u, v)
            feats.update(node_ctx)
        except Exception as e:
            feats = NULL_FEATS.copy()
        
        #We match the features with the edge identifiers so we can access them later
        rows.append({
            "u": u,
            "v": v,
            "k": k,
            **feats
        })
        
    
    df = pd.DataFrame(rows)
    print(f"Edge feature table shape: {df.shape}")
    print(f"Missing values:\n{df.isna().sum()}")
    
    out_path = PROCESSED_DIR / "zurich_edge_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df

if __name__ == "__main__":
    build_zurich_edge_features()