"""
Combines the three cleaned datasets into one final corpus.
Run this AFTER clean_fakenewsnet_main.py, clean_isot.py, clean_liar.py
have each produced their parquet files.

Output: data/processed/combined_clean.parquet
This is what Person B splits into train/test — BUT NOTE:

LIAR rows carry an 'official_split' value ("train"/"valid"/"test") from
LIAR's own canonical benchmark split — honor it, don't re-shuffle it.
ISOT and FakeNewsNet rows have official_split == "" since neither ships
a canonical split; Person B's source/topic-held-out split logic applies
ONLY to those two. Final split:
    train = LIAR[official_split=="train"] + isot_train_slice + fnn_train_slice
    valid = LIAR[official_split=="valid"] + (carve some from the other two if needed)
    test  = LIAR[official_split=="test"]  + isot_test_slice  + fnn_test_slice
"""
import os
import pandas as pd
from schema import REQUIRED_COLUMNS, validate_dataframe

PARTS = [
    "data/processed/fakenewsnet_main_clean.parquet",
    "data/processed/isot_clean.parquet",
    "data/processed/liar_clean.parquet",
]
OUT_PATH = "data/processed/combined_clean.parquet"


def main():
    frames = []
    for p in PARTS:
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
        else:
            print(f"WARNING: {p} not found — run its cleaning script first, skipping for now")

    if not frames:
        raise SystemExit("No cleaned parts found. Run the individual clean_*.py scripts first.")

    df = pd.concat(frames, ignore_index=True)
    df = df[REQUIRED_COLUMNS]
    validate_dataframe(df)

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} total rows to {OUT_PATH}")
    print("\nBy dataset_source:")
    print(df["dataset_source"].value_counts())
    print("\nBy label:")
    print(df["label"].value_counts())
    print("\nBy dataset_source x label:")
    print(df.groupby("dataset_source")["label"].value_counts())
    print("\nBy official_split (non-empty = has a canonical split to honor):")
    print(df["official_split"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
