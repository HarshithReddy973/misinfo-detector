"""
Maps the Kaggle mdepak/fakenewsnet mirror into the unified schema.
https://www.kaggle.com/datasets/mdepak/fakenewsnet

Files used (label comes from filename, not a column):
  BuzzFeed_fake_news_content.csv   -> label=fake, source_dataset=buzzfeed
  BuzzFeed_real_news_content.csv   -> label=real, source_dataset=buzzfeed
  PolitiFact_fake_news_content.csv -> label=fake, source_dataset=politifact
  PolitiFact_real_news_content.csv -> label=real, source_dataset=politifact

Columns in each CSV: id, title, text, url, top_img, authors, source,
publish_date, movies, images, canonical_link, meta_data

Deliberately NOT used: *NewsUser.txt, *UserUser.txt, *UserFeature.mat
(social/propagation graph data — out of scope for the core classifier,
optional stretch goal only per the PS).
"""
import os
import re
import ast
import html
import unicodedata
import pandas as pd
from schema import REQUIRED_COLUMNS, validate_dataframe

RAW_DIR = "data/raw/fakenewsnet"
OUT_PATH = "data/processed/fakenewsnet_main_clean.parquet"

FILES = [
    ("BuzzFeed_fake_news_content.csv", "fake", "buzzfeed"),
    ("BuzzFeed_real_news_content.csv", "real", "buzzfeed"),
    ("PolitiFact_fake_news_content.csv", "fake", "politifact"),
    ("PolitiFact_real_news_content.csv", "real", "politifact"),
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_authors(val) -> str:
    """authors column is usually a stringified list like "['John Smith']" """
    if not isinstance(val, str) or not val.strip():
        return "unknown"
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            names = [str(n).strip() for n in parsed if str(n).strip()]
            return ", ".join(names) if names else "unknown"
    except (ValueError, SyntaxError):
        pass
    return val.strip() or "unknown"


def load_one(filename: str, label: str, subset: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    raw = pd.read_csv(path)

    df = pd.DataFrame()
    df["id"] = [f"fnn_{subset}_{label}_{i:04d}" for i in range(len(raw))]
    df["raw_text"] = raw.get("text", "").fillna("")
    df["text"] = df["raw_text"].apply(clean_text)
    df["title"] = raw.get("title", "").fillna("")
    df["source"] = raw.get("source", "unknown").fillna("unknown")
    df["author"] = raw.get("authors", "").apply(parse_authors)
    df["timestamp"] = raw.get("publish_date", "").fillna("")
    df["share_count"] = 0     # not in this mirror — see *UserUser.txt if you add it later
    df["reply_count"] = 0
    df["label"] = label
    df["dataset_source"] = f"fakenewsnet_{subset}"
    df["official_split"] = ""   # FakeNewsNet has no canonical split — Person B's source/topic split applies
    return df


def load_and_map() -> pd.DataFrame:
    parts = []
    for fname, label, subset in FILES:
        p = os.path.join(RAW_DIR, fname)
        if os.path.exists(p):
            parts.append(load_one(fname, label, subset))
        else:
            print(f"WARNING: {p} not found, skipping")
    df = pd.concat(parts, ignore_index=True)
    df = df[REQUIRED_COLUMNS]

    empty_text = (df["text"].str.len() <= 20).sum()
    if empty_text:
        print(f"NOTE: {empty_text} rows have little/no text — check for scraping gaps before dropping")
    df = df[df["text"].str.len() > 20]
    return df


if __name__ == "__main__":
    df = load_and_map()
    validate_dataframe(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} cleaned rows to {OUT_PATH}")
    print(df.groupby("dataset_source")["label"].value_counts())
