import pandas as pd
df = pd.read_csv("Data/processed/feature_context_filtered.csv")

# Nur Städte die wir aktuell nutzen
exclude = ["Copenhagen", "Santiago", "Valparaiso", "Hong Kong"]
df = df[~df["city_name"].isin(exclude)].copy()

# is_lit und has_sidewalk Zero-Share pro Stadt
audit = df.groupby("city_name")[["is_lit", "has_sidewalk", "surface_smoothness"]].mean()
audit["n_rows"] = df.groupby("city_name").size()
audit = audit.sort_values("is_lit")
print(audit.to_string())

import pandas as pd
df = pd.read_csv("Data/processed/training_base.csv")
print(df["image_id"].head(10).tolist())

import pandas as pd
df = pd.read_csv("Data/archive/votes_clean.csv", nrows=5)
print(df[["left", "right", "place_id_left", "place_id_right"]].to_string())

import pandas as pd

df = pd.read_csv("Data/archive/votes_clean.csv")

left_map = df[["left", "place_id_left"]].rename(
    columns={"left": "image_id", "place_id_left": "place_id"}
)
right_map = df[["right", "place_id_right"]].rename(
    columns={"right": "image_id", "place_id_right": "place_id"}
)

id_map = pd.concat([left_map, right_map]).drop_duplicates(subset="image_id")
print(id_map.shape)
print(id_map.head())

id_map.to_csv("Data/processed/image_id_to_place_id.csv", index=False)

import os
from pathlib import Path

# Passe den Pfad zu deinem Bildordner an
img_dir = Path(r"C:\Users\sidne\Documents\Uni\CS_Projekt\Data\archive\gsv\final_photo_dataset")

# Zähle Bilder pro place_id prefix (erste 24 Zeichen)
files = list(img_dir.glob("*.jpg"))
print(f"Total images: {len(files)}")
print(f"Sample filenames: {[f.name for f in files[:5]]}")

# Extrahiere place_id aus Dateinamen
place_ids = set(f.stem[:24] for f in files)
print(f"Unique place_ids in folder: {len(place_ids)}")

from pathlib import Path
import pandas as pd

img_dir = Path(r"C:\Users\sidne\Documents\Uni\CS_Projekt\Data\archive\gsv\final_photo_dataset")
df = pd.read_csv(r"C:\Users\sidne\Documents\Uni\CS_Projekt\Data\processed\training_base.csv")

# Prüfe ob image_ids direkt als Dateinamen existieren
sample_ids = df["image_id"].head(10).tolist()
for img_id in sample_ids:
    # Versuche verschiedene Suffixe
    for suffix in ["a", "b", "c", ""]:
        path = img_dir / f"{img_id}{suffix}.jpg"
        if path.exists():
            print(f"FOUND: {path.name}")
            break
    else:
        print(f"NOT FOUND: {img_id}")

import os
from pathlib import Path

img_dir = Path(r"C:\Users\sidne\Documents\Uni\CS_Projekt\Data\archive\gsv\final_photo_dataset")
total_size = sum(f.stat().st_size for f in img_dir.glob("*.jpg"))
print(f"Total size: {total_size / 1e9:.2f} GB")
print(f"Average per image: {total_size / 110688 / 1e3:.1f} KB")

