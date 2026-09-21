"""
Maps ISOT (True.csv / Fake.csv) into the unified schema.
ISOT columns: title, text, subject, date
No source, no author, no engagement data — all default to unknown/0.

NOTE: 'subject' in ISOT is a topic tag (e.g. "politicsNews", "worldnews"),
NOT a publisher/source. Don't map it to 'source' — that would silently
create a fake source-credibility signal that isn't real.
"""
import os
import re
import html
import unicodedata
import pandas as pd
from schema import REQUIRED_COLUMNS, validate_dataframe

TRUE_PATH = "data/raw/isot/True.csv"
FAKE_PATH = "data/raw/isot/Fake.csv"
OUT_PATH = "data/processed/isot_clean.parquet"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_one(path: str, label: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = pd.DataFrame()
    df["raw_text"] = raw["text"].fillna("")
    df["text"] = df["raw_text"].apply(clean_text)
    df["title"] = raw.get("title", "").fillna("")
    df["source"] = "unknown"          # 'subject' is topic, not publisher — don't conflate
    df["author"] = "unknown"
    df["timestamp"] = raw.get("date", "").fillna("")
    df["share_count"] = 0
    df["reply_count"] = 0
    df["label"] = label
    df["dataset_source"] = "isot"
    df["official_split"] = ""   # ISOT has no canonical split — Person B's source/topic split applies
    return df


def load_and_map() -> pd.DataFrame:
    real = load_one(TRUE_PATH, "real")
    fake = load_one(FAKE_PATH, "fake")
    df = pd.concat([real, fake], ignore_index=True)
    df["id"] = [f"isot_{i:05d}" for i in range(len(df))]
    df = df[REQUIRED_COLUMNS]
    df = df[df["text"].str.len() > 20]
    return df


if __name__ == "__main__":
    df = load_and_map()
    validate_dataframe(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} cleaned rows to {OUT_PATH}")
    print(df["label"].value_counts())
