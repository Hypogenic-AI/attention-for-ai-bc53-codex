import sys, os
if os.getcwd() in sys.path: sys.path.remove(os.getcwd())
if "" in sys.path: sys.path.remove("")
from datasets import load_dataset
import pandas as pd

def save_sample(dataset_name, output_dir, split="train", n=100):
    print(f"Loading {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
        sample = list(ds.take(n))
        df = pd.DataFrame(sample)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "sample.csv"), index=False)
        df.head(10).to_json(os.path.join(output_dir, "sample.json"), orient="records", indent=2)
        print(f"Saved sample to {output_dir}/sample.csv")
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")

save_sample("ashraq/movielens_ratings", "datasets/movielens")
save_sample("reczoo/Criteo_x1", "datasets/criteo")
