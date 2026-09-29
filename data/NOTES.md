# Data Notes — Misinformation Detector

This file documents the main data sources, cleaning, standardization, splitting, and feature-generation steps used by **Misinformation Detector**.

For the application and model details, see the root `README.md`.

---

## 1. Datasets

Three datasets are used:

* **FakeNewsNet** — news articles with available source and social/engagement metadata.
* **LIAR** — short political statements with six truthfulness labels.
* **ISOT Fake News Dataset** — news articles directly labeled as fake or real.w

The original datasets have different formats and metadata, so each is cleaned separately before being converted to a common schema.

---

## 2. Label Processing

The system uses a binary classification setup:

```text
fake
real
```

### LIAR

Original labels:

```text
pants-fire
false
barely-true
half-true
mostly-true
true
```

Mapping used:

```text
pants-fire  → fake
false       → fake

mostly-true → real
true        → real

barely-true → removed
half-true   → removed
```

half-true , barely-true is excluded because the current model is designed for binary real/fake classification.

### ISOT

```text
Fake → fake
Real → real
```

### FakeNewsNet

The available fake/real labels are standardized to:

```text
fake
real
```

---

## 3. Common Schema

All datasets are converted into a common structure before merging.

Important columns include:

| Column           | Purpose                                   |
| ---------------- | ----------------------------------------- |
| `id`             | Internal identifier                       |
| `item_id`        | Original dataset identifier               |
| `text`           | Cleaned text                              |
| `raw_text`       | Original text used for feature generation |
| `title`          | Title, when available                     |
| `source`         | Publisher/source, when available          |
| `author`         | Author, when available                    |
| `timestamp`      | Timestamp, when available                 |
| `label`          | `real` or `fake`                          |
| `share_count`    | Available engagement information          |
| `reply_count`    | Available engagement information          |
| `dataset_source` | Original dataset                          |
| `official_split` | Original split, when available            |

Missing information is left unavailable rather than artificially generated.

---

## 4. Cleaning Process

Each dataset has its own cleaning script because the raw formats differ.

General flow:

```text
Raw Dataset
    ↓
Dataset-specific cleaning
    ↓
Label conversion
    ↓
Column standardization
    ↓
Common schema
    ↓
Cleaned Parquet
```

Cleaning includes:

* Reading the original files
* Extracting relevant fields
* Removing unusable/invalid records
* Standardizing labels
* Standardizing column types
* Preserving available metadata
* Assigning `dataset_source`

---

## 5. Text Fields

Two text fields are retained:

### `raw_text`

Original text preserved for feature generation.

This is important because capitalization, punctuation and writing style are part of the linguistic signal.

### `text`

Cleaned text retained in the unified dataset.

**Important:** The current feature-generation pipeline uses `raw_text`, not the cleaned `text` field, for the main linguistic features.

For semantic embeddings, a lightly normalized version of `raw_text` is used with URLs removed and whitespace normalized.

---

## 6. Merging

After individual cleaning, the datasets are merged into a unified dataset:

```text
FakeNewsNet ──┐
LIAR ─────────┼──→ combined_clean.parquet
ISOT ─────────┘
```

`dataset_source` is retained for traceability.

---

## 7. Dataset Splitting

The unified data is divided into:

```text
train.parquet
valid.parquet
test.parquet
```

* **Train:** model training and training-dependent statistics.
* **Validation:** development and model evaluation.
* **Test:** final evaluation.

Ambiguous LIAR records excluded from the binary dataset are kept separately for the calibration/evaluation workflow where applicable.

---

## 8. Credibility Features

Source and author information is converted into four features:

```text
source_fake_ratio
source_known
author_fake_ratio
author_known
```

The fake ratios represent the historical proportion of fake-labeled training records associated with a source or author.

### Leakage Control

Credibility lookups are fitted **using training data only**.

```text
Training data
      ↓
Source/author statistics
      ↓
credibility_lookups.json
      ↓
Applied to validation/test/inference
```

Validation and test labels are never used to create these lookup statistics.

### Unknown Source/Author

If a source or author is not present in the training lookup:

```text
fake_ratio = training global fake ratio
known = 0
```

This allows missing or unseen metadata to be handled without inventing historical information.

---

## 9. Feature Generation

After splitting, features are generated independently for each split.

### Feature composition

```text
17 Linguistic Features
        +
384 Semantic Embedding Features
        +
4 Credibility Features
        =
405 Model Features
```

### Linguistic Features

The 17 features include:

```text
word_count
char_count
sentence_count
avg_sentence_len
avg_word_len
vocab_diversity
punct_density
caps_density
exclam_count
question_count
url_count
hashtag_count
mention_count
hedging_count
sentiment_compound
subjectivity
readability
```

Sentiment, subjectivity, and readability calculations use the available text-processing libraries. For long articles, these calculations are limited to the first 2,000 characters for efficiency.

### Semantic Features

Semantic representation is generated using:

```text
all-MiniLM-L6-v2
```

Output:

```text
384-dimensional embedding
```

### Credibility Features

```text
source_fake_ratio
source_known
author_fake_ratio
author_known
```

---

## 10. Feature Files and Artifacts

Main generated files:

```text
train_features.parquet
valid_features.parquet
test_features.parquet
credibility_lookups.json
feature_manifest.json
```

### `feature_manifest.json`

Stores the feature order and block structure used by the model and inference pipeline.

### `credibility_lookups.json`

Stores the training-derived source and author fake-ratio dictionaries.

---

## 11. Data Flow

```text
Raw datasets
     ↓
Dataset-specific cleaning
     ↓
Label standardization
     ↓
Common schema
     ↓
Merge
     ↓
Train / Validation / Test
     ↓
Feature Generation
     ↓
405 Features
     ↓
Model Training / Inference
```

---

## 12. Important Data Considerations

* `raw_text` is used for the main linguistic features.
* `dataset_source` is **not used as a model feature**, since it could allow the model to learn dataset identity instead of misinformation patterns.
* `label` is the target and is not included as a model feature.
* Credibility statistics use training data only to avoid leakage.
* Unknown sources/authors use the training global fake ratio with `*_known = 0`.
* `title` is retained in the unified data but is not currently used by the feature-generation module.
* `timestamp` is not currently used as a model feature.
* `share_count` and `reply_count` are not currently used as model features; in the current processed data they do not provide useful variation.
* Missing metadata is not artificially filled with fabricated information.

---

## 13. Main Data Outputs

```text
data/
├── raw/
│   ├── FakeNewsNet/
│   ├── ISOT-dataset/
│   └── LIAR-dataset/
│
├── processed/
│   ├── combined_clean.parquet
│   ├── train.parquet
│   ├── valid.parquet
│   ├── test.parquet
│   └── ...
│
├── credibility_lookups.json
└── feature_manifest.json
```

The exact file names or locations may change as the project evolves.

---

## 14. Dataset References

* **FakeNewsNet:**
  https://www.kaggle.com/datasets/mdepak/fakenewsnet

* **LIAR:**
  https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset

* **ISOT Fake News Dataset:**

       https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset

For the complete application documentation, see:

```text
../README.md
```
