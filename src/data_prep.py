import pandas as pd
from datasets import load_dataset
import os
from tqdm import tqdm

def prep_datasets():
    os.makedirs("datasets/movielens", exist_ok=True)
    os.makedirs("datasets/criteo", exist_ok=True)

    print("Downloading MovieLens ratings...")
    try:
        # MovieLens dataset from ashraq/movielens_ratings
        # Note: the dataset might be small enough to load fully
        ds_ml = load_dataset("ashraq/movielens_ratings", split="train")
        df_ml = pd.DataFrame(ds_ml)
        df_ml.to_csv("datasets/movielens/ratings_full.csv", index=False)
        print(f"MovieLens: {len(df_ml)} rows saved.")
    except Exception as e:
        print(f"Error downloading MovieLens: {e}")

    print("Downloading Criteo samples...")
    try:
        # Criteo dataset from reczoo/Criteo_x1
        # Streaming for large datasets
        ds_criteo_stream = load_dataset("reczoo/Criteo_x1", split="train", streaming=True).take(20000)
        criteo_list = []
        for item in tqdm(ds_criteo_stream, total=20000):
            criteo_list.append(item)
        df_criteo = pd.DataFrame(criteo_list)
        df_criteo.to_csv("datasets/criteo/criteo_20k.csv", index=False)
        print(f"Criteo: {len(df_criteo)} rows saved.")
    except Exception as e:
        print(f"Error downloading Criteo: {e}")

if __name__ == "__main__":
    prep_datasets()
