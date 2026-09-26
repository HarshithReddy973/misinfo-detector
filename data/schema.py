"""
Unified schema for all datasets (FakeNewsNet, LIAR, etc.)
Every dataset gets mapped into this shape before anything downstream touches it.

SHARE THIS FILE WITH THE TEAM FIRST — everyone builds against this.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd


REQUIRED_COLUMNS = [
    "id",            # str — unique row id, e.g. "fnn_0001"
    "text",          # str — cleaned article/post body
    "raw_text",      # str — UNCLEANED original text (keep stylistic noise: caps, punctuation)
    "title",         # str — headline/title, "" if none
    "source",        # str — publisher/domain name, "unknown" if missing
    "author",        # str — author name, "unknown" if missing
    "timestamp",     # str (ISO 8601) or "" if missing
    "label",         # str — "real" | "fake" | "ambiguous"  (ambiguous = LIAR's half-true/
                      #       barely-true rows, EXCLUDED from train/valid/test — see split_dataset.py)
    "share_count",   # int — 0 if not available
    "reply_count",   # int — 0 if not available
    "dataset_source",# str — "fakenewsnet" | "liar" | ... (so you can trace provenance)
    "official_split",# str — "train"|"valid"|"test" if the source dataset ships a canonical
                      #       split (LIAR does), else "" meaning "not pre-split, split it yourself"
]


@dataclass
class UnifiedRecord:
    id: str
    text: str
    raw_text: str
    title: str
    source: str
    author: str
    timestamp: str
    label: str          # "real" | "fake" | "ambiguous" (ambiguous excluded from training)
    share_count: int
    reply_count: int
    dataset_source: str
    official_split: str = ""   # "train"|"valid"|"test" if the source has a canonical split, else ""

    def __post_init__(self):
        if self.label not in ("real", "fake", "ambiguous"):
            raise ValueError(f"label must be 'real', 'fake', or 'ambiguous', got {self.label!r}")

    def to_dict(self):
        return asdict(self)


def validate_dataframe(df: pd.DataFrame) -> None:
    """Call this at the end of every cleaning script before saving. Fails loud, not silent."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    bad_labels = df[~df["label"].isin(["real", "fake", "ambiguous"])]
    if len(bad_labels):
        raise ValueError(f"{len(bad_labels)} rows have invalid labels: {bad_labels['label'].unique()}")

    empty_text = df[df["text"].str.strip() == ""]
    if len(empty_text):
        print(f"WARNING: {len(empty_text)} rows have empty cleaned text — check your cleaning logic")

    print(f"✓ Schema validated: {len(df)} rows, columns OK, labels OK")


if __name__ == "__main__":
    # quick smoke test
    r = UnifiedRecord(
        id="test_0001", text="sample cleaned text", raw_text="Sample RAW text!!!",
        title="Test headline", source="example.com", author="unknown",
        timestamp="2026-09-21T00:00:00", label="real",
        share_count=0, reply_count=0, dataset_source="test",
    )
    df = pd.DataFrame([r.to_dict()])
    validate_dataframe(df)
