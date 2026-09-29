# Data Notes — Misinformation Detector

This document describes the datasets, preprocessing, feature structure, data splitting, credibility features, and data artifacts used by the Misinformation Detector.

---

# 1. Datasets

The project combines multiple misinformation-related datasets.

## FakeNewsNet

FakeNewsNet provides fake and real news articles together with source and author information where available.

## LIAR

LIAR contains short political statements with multiple truthfulness labels.

The current preprocessing maps:

```text
pants-fire → fake
false → fake

mostly-true → real
true → real
```

The following labels are excluded:

```text
barely-true
half-true
```

## ISOT

The ISOT Fake News Dataset contains articles labeled as fake or real.

---

# 2. Label Processing

The project uses a binary target:

```text
Real
Fake
```

Dataset-specific labels are converted into this common representation so that the datasets can be combined into a common dataset.

---

# 3. Common Data Schema

After dataset-specific preprocessing, the datasets use a common structure.

Important fields include:

```text
text
title
source
author
label
```

Additional dataset-specific fields may exist in the intermediate datasets.

---

# 4. Data Processing

The datasets follow the general processing flow:

```text
Raw Datasets
    ↓
Dataset-specific Cleaning
    ↓
Column Standardization
    ↓
Text Cleaning
    ↓
Label Standardization
    ↓
Common Schema
    ↓
Dataset Merge
    ↓
Train / Validation / Test Split
```

The cleaned and processed datasets are available in the Google Drive folder linked below.

---

# 5. Raw Text and Model Text

Where applicable:

```text
raw_text
```

stores the original article text, while:

```text
text
```

is the standardized text used by the model pipeline.

---

# 6. Train / Validation / Test Data

The merged dataset is divided into:

```text
Training
Validation
Test
```

The complete split datasets are available in the Google Drive `processed/` folder.

---

# 7. Credibility Features

The model uses source and author history as additional features.

The four credibility features are:

```text
source_fake_ratio
source_known
author_fake_ratio
author_known
```

### Source Fake Ratio

Historical proportion of fake examples associated with a source in the training data.

### Source Known

Indicates whether the source exists in the training-derived lookup.

### Author Fake Ratio

Historical proportion of fake examples associated with an author in the training data.

### Author Known

Indicates whether the author exists in the training-derived lookup.

---

# 8. Leakage Control

Credibility statistics are based on training data so that validation and test examples do not directly influence the credibility lookup.

Unknown sources and authors use training-set fallback values.

---

# 9. Feature Structure

The final model uses:

```text
17 linguistic features
+ 384 semantic features
+ 4 credibility features
--------------------------------
= 405 total features
```

### Linguistic Features

```text
word count
character count
sentence count
average sentence length
average word length
vocabulary diversity
punctuation density
capitalization density
exclamation count
question count
URL count
hashtag count
mention count
hedging count
sentiment
subjectivity
readability
```

### Semantic Features

Semantic features use:

```text
all-MiniLM-L6-v2
```

which produces 384-dimensional embeddings.

### Credibility Features

```text
source_fake_ratio
source_known
author_fake_ratio
author_known
```

---

# 10. Data and Feature Files

The complete processed data is available through the public Google Drive folder:

**[Complete - DATASETS](https://drive.google.com/drive/folders/1XyAvQ3e9YdKo9HzdqpEp5YydTBwMR4b9?usp=sharing)**

The Google Drive `processed/` folder contains:

```text
processed/
├── features/
│   ├── train_features.parquet
│   ├── valid_features.parquet
│   └── test_features.parquet
│
├── calibration_eval.parquet
├── combined_clean.parquet
├── fakenewsnet_main_clean.parquet
├── isot_clean.parquet
├── liar_clean.parquet
├── train.parquet
├── valid.parquet
└── test.parquet
```

The `features/` directory contains the complete feature files.

The complete files are provided for:

* Reference
* Reproducibility
* Further experimentation
* Inspecting processed data and generated features

They are **not required to run the application**.

---

# 11. GitHub Application Feature File

The GitHub repository contains a trimmed version of the training feature file:

```text
data/processed/train_features.parquet
```

This file contains **100 rows** and is used as the SHAP background dataset.

The complete version is available in Google Drive at:

```text
processed/features/train_features.parquet
```

Therefore:

```text
GitHub
└── data/processed/train_features.parquet
    └── 100 rows → application SHAP background

Google Drive
└── processed/features/train_features.parquet
    └── Complete training feature dataset
```

The complete training feature file is not required for the application.

---

# 12. SHAP Background Data

The application uses the 100-row GitHub feature file:

```text
data/processed/train_features.parquet
```

as its SHAP background dataset.

The SHAP flow is:

```text
100-row train_features.parquet
          ↓
   SHAP background data
          ↓
     SHAP Explainer
          ↓
   Explain prediction
          ↓
Linguistic / Semantic / Credibility
```

If the 100-row file is missing:

```text
Prediction       → works
Probability      → works
Confidence       → works
Review priority  → works
SHAP             → skipped
```

---

# 13. Complete Data Flow

```text
Raw Datasets
     ↓
Dataset Cleaning
     ↓
Label Standardization
     ↓
Common Schema
     ↓
Dataset Merge
     ↓
Train / Validation / Test Split
     ↓
Feature Data
     ↓
Model Training
     ↓
Calibrated Model
     ↓
Application Inference
     ↓
Prediction + Confidence + SHAP
```

The complete intermediate and feature files are available in Google Drive for reference.

---

# 14. Important Data Considerations

### Raw Datasets

Complete raw datasets are available through the public Google Drive folder.

### Processed Datasets

Complete processed datasets and feature files are also available through Google Drive.

They are not required to run the already-trained application.

### Application Feature File

The only processed feature file required by the application is:

```text
data/processed/train_features.parquet
```

The GitHub version contains 100 rows for SHAP background data.

### Feature Ordering

The model expects the same feature ordering used during training.

The `feature_manifest.json` file describes the feature structure used by the model.

---

# 15. Dataset References

The project uses:

* FakeNewsNet
* LIAR
* ISOT Fake News Dataset

---

# 16. Related Application Files

```text
model/inference.py
app/app.py
api/main.py
data/credibility_lookups.json
data/feature_manifest.json
models/calibrated_model.joblib
```
