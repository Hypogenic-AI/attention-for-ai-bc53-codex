# Downloaded Datasets

This directory contains datasets for investigating the attention economy.

## Dataset 1: MovieLens Ratings (Sample)
- **Source**: `ashraq/movielens_ratings` (HuggingFace)
- **Size**: 100 samples in `movielens/sample.csv`
- **Format**: CSV
- **Task**: Recommendation
- **Description**: Classic dataset showing how users allocate "attention" (in the form of ratings) to items.

## Dataset 2: Criteo x1 (Sample)
- **Source**: `reczoo/Criteo_x1` (HuggingFace)
- **Size**: 100 samples in `criteo/sample.csv`
- **Format**: CSV
- **Task**: CTR Prediction
- **Description**: Standard dataset for the attention economy in online advertising.

## Download Instructions

### Using HuggingFace (recommended):
```python
from datasets import load_dataset
# MovieLens
ds_ml = load_dataset("ashraq/movielens_ratings")
# Criteo
ds_criteo = load_dataset("reczoo/Criteo_x1")
```

### Loading the Samples:
```python
import pandas as pd
df_ml = pd.read_csv("datasets/movielens/sample.csv")
df_criteo = pd.read_csv("datasets/criteo/sample.csv")
```
