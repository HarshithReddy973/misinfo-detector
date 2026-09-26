"""
Splits combined_clean.parquet into train/valid/test + a separate
calibration_eval set.

Step 1 -- separate by label:
  - real / fake rows  -> the trainable pool, goes through the split below
  - ambiguous rows    -> NEVER trained or evaluated on directly. Written to
                          data/processed/calibration_eval.parquet instead.
                          Use it once, after training, to check whether the
                          model's confidence naturally dips on these
                          genuinely ambiguous statements it never saw --
                          that's the calibration story for your report.

Step 2 -- split the trainable (real/fake) pool:
  - LIAR rows (official_split != "") -> use LIAR's own canonical split as-is.
  - ISOT + FakeNewsNet rows (official_split == "") -> no canonical split
    exists. Neither has a usable real 'source' field for a source-held-out
    split (ISOT's is all "unknown"; FakeNewsNet has only ~420 rows total,
    too few to hold out by source without breaking class balance). So:
    dedupe exact-duplicate text first (ISOT has known repeated articles --
    a duplicate landing in both train and test is leakage that inflates
    your score), then do a stratified random split by label.

Output:
  data/processed/train.parquet             (real/fake only)
  data/processed/valid.parquet             (real/fake only)
  data/processed/test.parquet              (real/fake only)
  data/processed/calibration_eval.parquet  (ambiguous only, held out entirely)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = "data/processed/combined_clean.parquet"
SEED = 42
TEST_FRAC = 0.15   # of the non-LIAR trainable rows
VALID_FRAC = 0.15  # of the non-LIAR trainable rows remaining after test is carved out


def split_non_liar(df: pd.DataFrame):
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} exact-duplicate text rows before splitting (leakage prevention)")

    train_val, test = train_test_split(
        df, test_size=TEST_FRAC, stratify=df["label"], random_state=SEED
    )
    train, valid = train_test_split(
        train_val, test_size=VALID_FRAC, stratify=train_val["label"], random_state=SEED
    )
    return train, valid, test


def main():
    df = pd.read_parquet(IN_PATH)

    # --- Step 1: pull ambiguous rows out entirely, before any splitting ---
    ambiguous = df[df["label"] == "ambiguous"]
    trainable = df[df["label"].isin(["real", "fake"])]

    print(f"Total rows: {len(df)}  |  trainable (real/fake): {len(trainable)}  |  ambiguous (excluded): {len(ambiguous)}")

    ambiguous.to_parquet("data/processed/calibration_eval.parquet", index=False)
    print(f"\ncalibration_eval: {len(ambiguous)} rows (all LIAR half-true/barely-true -- never trained or tested on)")
    print(ambiguous["dataset_source"].value_counts())

    # --- Step 2: split the trainable pool ---
    liar = trainable[trainable["official_split"] != ""]
    other = trainable[trainable["official_split"] == ""]

    liar_train = liar[liar["official_split"] == "train"]
    liar_valid = liar[liar["official_split"] == "valid"]
    liar_test = liar[liar["official_split"] == "test"]

    other_train, other_valid, other_test = split_non_liar(other)

    train = pd.concat([liar_train, other_train], ignore_index=True)
    valid = pd.concat([liar_valid, other_valid], ignore_index=True)
    test = pd.concat([liar_test, other_test], ignore_index=True)

    train.to_parquet("data/processed/train.parquet", index=False)
    valid.to_parquet("data/processed/valid.parquet", index=False)
    test.to_parquet("data/processed/test.parquet", index=False)

    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        print(f"\n{name}: {len(split)} rows")
        print(split["label"].value_counts())

    # sanity checks
    leak_train_test = set(train["text"]) & set(test["text"])
    if leak_train_test:
        print(f"\nWARNING: {len(leak_train_test)} identical texts appear in BOTH train and test!")
    else:
        print("\n[OK] No exact-text overlap between train and test")

    assert "ambiguous" not in set(train["label"]) | set(valid["label"]) | set(test["label"])
    print("[OK] Confirmed: no 'ambiguous' rows leaked into train/valid/test")


if __name__ == "__main__":
    main()
