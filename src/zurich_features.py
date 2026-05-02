from pathlib import Path
import pandas as pd
import sys

# Füge src zum Path hinzu damit wir feature_engineering importieren können
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    
    print(f"Zürich: nodes={len(G.nodes)}, edges={len(G.edges)}")
    
    # Alle Edges mit Features extrahieren
    rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        try:
            feats = edge_to_features(data)
            node_ctx = get_node_context_features(G, u, v)
            feats.update(node_ctx)
        except Exception as e:
            feats = NULL_FEATS.copy()
        
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