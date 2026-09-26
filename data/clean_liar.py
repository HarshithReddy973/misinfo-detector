"""
Maps LIAR (train.tsv / valid.tsv / test.tsv) into the unified schema.

Columns per LIAR's README (no header row in the TSV):
1 id | 2 label | 3 statement | 4 subject | 5 speaker | 6 speaker_job |
7 state | 8 party | 9 barely_true_ct | 10 false_ct | 11 half_true_ct |
12 mostly_true_ct | 13 pants_fire_ct | 14 context

Label collapse — binary training labels + one excluded class:
  true, mostly-true        -> real
  false, pants-fire        -> fake
  half-true, barely-true   -> ambiguous (EXCLUDED from train/valid/test —
                               never trained or evaluated on directly;
                               used only in split_dataset.py's separate
                               calibration_eval.parquet, to check whether
                               the trained model's confidence dips on
                               genuinely ambiguous statements it never saw)
"""
import os
import re
import unicodedata
import pandas as pd
from schema import REQUIRED_COLUMNS, validate_dataframe

COLS = [
    "liar_id", "raw_label", "statement", "subject", "speaker",
    "speaker_job", "state", "party", "barely_true_ct", "false_ct",
    "half_true_ct", "mostly_true_ct", "pants_fire_ct", "context",
]

LABEL_MAP = {
    "true": "real",
    "mostly-true": "real",
    "half-true": "ambiguous",     # excluded from training — see split_dataset.py
    "barely-true": "ambiguous",   # excluded from training — see split_dataset.py
    "false": "fake",
    "pants-fire": "fake",
}

RAW_DIR = "data/raw/liar"
OUT_PATH = "data/processed/liar_clean.parquet"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(filename: str, split_name: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    raw = pd.read_csv(path, sep="\t", header=None, names=COLS)

    df = pd.DataFrame()
    df["id"] = raw["liar_id"].astype(str).apply(lambda x: f"liar_{split_name}_{x}")
    df["raw_text"] = raw["statement"].fillna("")
    df["text"] = df["raw_text"].apply(clean_text)
    df["title"] = ""                                   # LIAR has no headline, only short statements
    df["source"] = raw["state"].fillna("unknown")       # best available proxy — not a true publisher
    df["author"] = raw["speaker"].fillna("unknown")
    df["timestamp"] = ""                                # not available
    df["share_count"] = 0
    df["reply_count"] = 0
    df["label"] = raw["raw_label"].str.lower().str.strip().map(LABEL_MAP)
    df["dataset_source"] = "liar"
    df["official_split"] = split_name   # "train"|"valid"|"test" — LIAR's own canonical split, honor it
    return df


def load_and_map() -> pd.DataFrame:
    parts = []
    for fname, split in [("train.tsv", "train"), ("valid.tsv", "valid"), ("test.tsv", "test")]:
        p = os.path.join(RAW_DIR, fname)
        if os.path.exists(p):
            parts.append(load_split(fname, split))
        else:
            print(f"WARNING: {p} not found, skipping")
    df = pd.concat(parts, ignore_index=True)

    unmapped = df[df["label"].isna()]
    if len(unmapped):
        print(f"WARNING: {len(unmapped)} rows had unrecognized raw labels, dropping")
        df = df.dropna(subset=["label"])

    df = df[REQUIRED_COLUMNS]
    df = df[df["text"].str.len() > 5]
    return df


if __name__ == "__main__":
    df = load_and_map()
    validate_dataframe(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} cleaned rows to {OUT_PATH}")
    print(df["label"].value_counts())
