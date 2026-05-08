from pathlib import Path
import pandas as pd

#directory paths + creates folder if it doesn't exist when saving
BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = BASE_DIR / "Data" / "archive"
PROCESSED_DIR = BASE_DIR / "Data" / "processed"

#define valid choices to filter out garbage
VALID_CHOICES = {"left", "right", "equal"}

#load raw votes data from CSV
def load_votes() -> pd.DataFrame:
    file_path = ARCHIVE_DIR / "votes_clean.csv"
    df = pd.read_csv(file_path)
    return df


def clean_votes(df: pd.DataFrame) -> pd.DataFrame:
    
    #Keep only valid vote rows and rows with required IDs/coordinates + filter for "safer" comparisoons, as we dont care about the "more beautfiful" ones 

    df = df[df["choice"].isin(VALID_CHOICES)].copy()
    df = df[df["study_question"] == "safer"].copy()
    df = df.dropna(
        subset=[
            "left",
            "right",
            "lat_left",
            "long_left",
            "lat_right",
            "long_right",
        ]
    )

    return df


# -----------------------    SCORE BUILDING AREA   -----------------------

    #build one perceived-safety score per place_id: score = (wins + 0.5 * equals) / total_comparisons to account for places that are often compared but rarely win, and places that are often compared and often win. 
    #This gives us a score between 0 and 1 for each place_id, where 1 means it always wins, 0 means it never wins, and 0.5 means it wins as often as it loses (including ties
    #NOW DONE ON IMAGE LEVEL (left/right columns), because place id is actually only the city, instead image id is unique identifier for location
    #actually we only use elo score now, see below, because its better, but I dont want to kick out normal score

def build_place_scores(df: pd.DataFrame) -> pd.DataFrame:
    
    #Calculate wins and total comparisons for each place_id
    wins: dict[str, float] = {}
    total: dict[str, int] = {}

    for _, row in df.iterrows():
        left = row["left"]
        right = row["right"]
        choice = row["choice"]

        #pid stands for place id (now image_id)
        #Ensure both place_ids are in the wins and total dictionaries, initializing if necessary
        for pid in (left, right):
            if pid not in wins:
                wins[pid] = 0.0
                total[pid] = 0

        #Increment total comparisons for both place_ids since they were compared in this row
        total[left] += 1
        total[right] += 1

        #Increment wins based on the choice. If left wins, increment lefts wins. If right wins, increment rights wins. If it's a tie, increment both by 0.5.
        if choice == "left":
            wins[left] += 1.0
        elif choice == "right":
            wins[right] += 1.0
        elif choice == "equal":
            wins[left] += 0.5
            wins[right] += 0.5

    #After processing all rows, build a DataFrame with place_id, score, total comparisons, and wins_equivalent (which is the raw wins count without accounting for total comparisons)
    rows = []
    for pid in wins:
        comparisons = total[pid]
        score = wins[pid] / comparisons if comparisons > 0 else None
        rows.append(
            {
                "image_id": pid, # we just changed this from place_id to image_id as mentioned above
                "score": score,
                "comparisons": comparisons,
                "wins_equivalent": wins[pid],
            }
        )

    scores_df = pd.DataFrame(rows)
    return scores_df

#I had this great idea that our rating wasnt accurate enough, so I thought building an elo system for the scores like in chess could improve accuracy. Well, I guess I have too much time. The idea is to consider for unfair matchups, where a win against an unsafe place shouldnt count as much as a win against a safe one. 
def build_elo_scores(df: pd.DataFrame, k_factor: float = 32.0, initial_rating: float = 1500.0) -> pd.DataFrame:
    
    #Calculate Elo ratings for each image_id, wins and total is given, so I use different variables (yes I got confused multiple times)
    ratings: dict[str, float] = {}
    comparisons: dict[str, int] = {}

    for _, row in df.iterrows():
        left = row["left"]
        right = row["right"]
        choice = row["choice"]

        #Initialize ratings and comparison counts if needed
        for img in (left, right):
            if img not in ratings:
                ratings[img] = initial_rating
                comparisons[img] = 0

        rating_left = ratings[left]
        rating_right = ratings[right]

        #Expected scores based on current ratings. CLaude gave me this formular, because I just couldnt get it right
        expected_left = 1 / (1 + 10 ** ((rating_right - rating_left) / 400))
        expected_right = 1 / (1 + 10 ** ((rating_left - rating_right) / 400))

        #Actual result from vote
        if choice == "left":
            actual_left = 1.0
            actual_right = 0.0
        elif choice == "right":
            actual_left = 0.0
            actual_right = 1.0
        elif choice == "equal":
            actual_left = 0.5
            actual_right = 0.5
        else:
            continue

        #Update ratings
        ratings[left] = rating_left + k_factor * (actual_left - expected_left)
        ratings[right] = rating_right + k_factor * (actual_right - expected_right)

        comparisons[left] += 1
        comparisons[right] += 1

    rows = []
    for img in ratings: #Why the hell did we use img as variable name when we used pid before? To seperate further from the other score?
        rows.append(
            {
                "image_id": img,
                "elo_rating": ratings[img],
                "comparisons": comparisons[img],
            }
        )

    elo_df = pd.DataFrame(rows)

    #normalize Elo roughly to 0-1 for easier ML target comparison
    min_rating = elo_df["elo_rating"].min()
    max_rating = elo_df["elo_rating"].max()

    if max_rating > min_rating:
        elo_df["elo_score"] = (elo_df["elo_rating"] - min_rating) / (max_rating - min_rating)
    else:
        elo_df["elo_score"] = 0.5

    return elo_df

#------------------------  COORDINATE AREA -----------------------

#Extract one coordinate pair per place_id from left and right columns. If the same place_id appears multiple times, use the first observed coordinates (in case there are inconsistencies, which we hope are minimal, but you never know)
#NOW DONE ON IMAGE LEVEL
#In the raw data long and lat is switched, so we switch it back. THIS IS INTENTIONAL. I CHECKED IT FOR MULTIPLE CITIES. GUESS HOW LONG IT TOOK ME TO FIND THIS OUT, WHY THE HELL IT DIDNT WORK PROPERLY?

def build_place_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    
    #exctract place id and coords from left and right columns.
    left_places = df[["left", "long_left", "lat_left"]].copy()
    left_places.columns = ["image_id", "lat", "lon"]

    right_places = df[["right", "long_right", "lat_right"]].copy()
    right_places.columns = ["image_id", "lat", "lon"]
    #...concatenate them together, so we dont seperate left and right (because a place can appear in the left row and afterwards in the right row)
    places = pd.concat([left_places, right_places], ignore_index=True)

    #remove exact dupes
    places = places.drop_duplicates()

    #If same place_id appears multiple times with same coords, keep one
    #if same place_id appears with slightly inconsistent coords, keep first for now, because then we have no chance in finding the "true" coords, so we just pray inconsistencies are minimal, but you never know
    places = places.groupby("image_id", as_index=False).first()

    return places



# We also need the city, to map it against its respective city graph on osm, to get the features we later use for our ML model. 
#Extract one city/place name per image_id from left and right columns.
#If the same image appears multiple times, keep the first observed city name.
def build_image_cities(df: pd.DataFrame) -> pd.DataFrame:
    
    left_cities = df[["left", "place_name_left"]].copy()
    left_cities.columns = ["image_id", "city_name"]

    right_cities = df[["right", "place_name_right"]].copy()
    right_cities.columns = ["image_id", "city_name"]

    cities = pd.concat([left_cities, right_cities], ignore_index=True)

    cities = cities.drop_duplicates()
    cities = cities.groupby("image_id", as_index=False).first()

    return cities


# -------------------------------   PREPARING FOR MODEL TRAINING --------------------------------

#Build base training table: place_id | lat | lon | score | comparisons | wins_equivalent
#NOW: image_id | lat | lon | score ...

def build_training_base(df: pd.DataFrame) -> pd.DataFrame:
    scores_df = build_place_scores(df)
    elo_df = build_elo_scores(df)
    coords_df = build_place_coordinates(df)
    cities_df = build_image_cities(df)

    #merge scores and coordinates on place_id to get the final training base with all required columns. We use an inner join to keep only place_ids that have both a score and coordinates, which should be the majority of them, but we might lose some if there are inconsistencies in the data
    training_base = scores_df.merge(coords_df, on="image_id", how="inner")
    training_base = training_base.merge(
        elo_df[["image_id", "elo_rating", "elo_score"]],
        on="image_id",
        how="inner"
    )
    training_base = training_base.merge(
        cities_df,
        on="image_id",
        how="inner"
    )

    # Reorder columns, because I hate chaotic dataframes
    training_base = training_base[
        ["image_id", "city_name", "lat", "lon", "score", "elo_rating", "elo_score", "comparisons", "wins_equivalent"]
    ]

    return training_base

#Im sure we would have big loading times if we had to run the whole process every time, so we save the cleaned and processed training base to a new CSV file, which we can load, without having to redo all the calculations 
def save_processed_csv(df: pd.DataFrame, filename: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# --------------------------- RUNNING THE PROCESS (wohoo!) ---------------------------

#run the whole process and save the final training base. I used Claude to give me the prints and strings (otherwise I wouldnt know which number is which and I need them for error checking (there were multiple errors)), because its faster and tedious to do by hand. 
if __name__ == "__main__":
    votes_df = load_votes()
    print("Raw shape:", votes_df.shape)

    votes_df = clean_votes(votes_df)
    print("Cleaned shape:", votes_df.shape)
    print("\nChoice counts after cleaning:")
    print(votes_df["choice"].value_counts())

    training_base_df = build_training_base(votes_df)
    training_base_df = training_base_df[training_base_df["comparisons"] >= 8].copy()

    print("\nTraining base shape:", training_base_df.shape)
    print("\nTraining base head:")
    print(training_base_df.head())

    print("\nLowest scores:")
    print(training_base_df.sort_values("score").head(10))

    print("\nHighest scores:")
    print(training_base_df.sort_values("score", ascending=False).head(10))

    save_processed_csv(training_base_df, "training_base.csv")

    print("left unique:", votes_df["place_id_left"].nunique())


    print("left unique images:", votes_df["left"].nunique())
    print("right unique images:", votes_df["right"].nunique())
    print("all unique images:", pd.concat([votes_df["left"], votes_df["right"]]).nunique())

    print("\nUnique left sample:")
    print(votes_df["left"].drop_duplicates().head(20).tolist())
    print(votes_df["study_question"].value_counts())
    print(votes_df["study_question"].drop_duplicates().tolist())
    print("\nLowest Elo scores:")
    print(training_base_df.sort_values("elo_score").head(10))

    print("\nHighest Elo scores:")
    print(training_base_df.sort_values("elo_score", ascending=False).head(10))
    print("\nUnique cities:")
    print(training_base_df["city_name"].nunique())

    print("\nCity sample:")
    print(training_base_df["city_name"].drop_duplicates().head(20).tolist())

    print("\nTop city counts:")
    print(training_base_df["city_name"].value_counts().head(20))