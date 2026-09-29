# Misinformation Detector

**Misinformation Detector** is an NLP-based Trust & Safety system that identifies potentially misleading or fake news content and provides **calibrated confidence, source/author signals, and SHAP-based explanations** to support human review.

The system is designed as a **decision-support tool, not an automated fact-checker**.

---

## Features

### Single Article Analysis

Analyze an individual article or post using:

* Article text — required
* Source — optional
* Author — optional

Returns:

* Real / Fake prediction
* Confidence score
* Probability of Real
* Probability of Fake
* Review priority
* Source and author credibility information
* SHAP-based explanation

### Batch Analysis

Upload a CSV file for analyzing multiple articles.

Required column:

```text
text
```

Optional columns:

```text
title,source,author
```

Example:

```csv
text,title,source,author
"Article text here","Example Article","example.com","John Doe"
"Another article","Second Article","",""
```

### Human Review Queue

Reviewers can:

* Confirm a prediction
* Dismiss a prediction
* Relabel an item

Reviewer actions are stored separately from the original model prediction.

---

# System Architecture

```text
User
  ↓
Single Article / CSV Batch
  ↓
Streamlit UI
  ↓ HTTP
FastAPI Backend
  ↓
Inference Module
  ↓
Trained Model
  ↓
Prediction / Confidence / SHAP
  ↓
Final Result
  ↓
SQLite Database
```

### Main Components

| Component                      | Purpose                        |
| ------------------------------ | ------------------------------ |
| Streamlit                      | Frontend and user interface    |
| FastAPI                        | Backend API                    |
| `model/inference.py`           | Centralized inference pipeline |
| Calibrated Logistic Regression | Primary classification model   |
| `all-MiniLM-L6-v2`             | Semantic text embeddings       |
| SHAP                           | Model explanation              |
| SQLite                         | Prediction and review storage  |

---

# Machine Learning

The model uses **405 features**:

```text
17 linguistic features
+ 384 semantic embedding features
+ 4 credibility features
--------------------------------
= 405 features
```

### Linguistic Features

The 17 linguistic features include:

* Word count
* Character count
* Sentence count
* Average sentence length
* Average word length
* Vocabulary diversity
* Punctuation density
* Capitalization density
* Exclamation count
* Question count
* URL count
* Hashtag count
* Mention count
* Hedging count
* Sentiment
* Subjectivity
* Readability

### Semantic Features

Semantic representations are generated using:

```text
all-MiniLM-L6-v2
```

This produces a **384-dimensional embedding** for the article text.

### Credibility Features

```text
source_fake_ratio
source_known
author_fake_ratio
author_known
```

Unknown sources and authors use training-set fallback values.

---

# Prediction and Confidence

The model performs binary classification:

```text
Real / Fake
```

The class with the highest predicted probability becomes the raw prediction.

For the application UI:

```text
Confidence < 0.60 → Uncertain
```

**Uncertain is a post-prediction display category, not a third trained class.**

---

# Explainability

The system uses **SHAP** to explain predictions.

Explanations are grouped into:

* Linguistic
* Semantic
* Credibility

## SHAP Background File

The application requires:

```text
data/processed/train_features.parquet
```

The version included in this repository contains **100 training-feature rows** and is used as the SHAP background dataset.

After cloning, this file must be present at exactly:

```text
data/processed/train_features.parquet
```

If it is missing:

* Predictions still work
* Probabilities and confidence still work
* Review priority still works
* SHAP explanations are skipped
* The **"Why this prediction?"** section will not render

The original complete training feature file is much larger and is available separately through Google Drive.

---

# Complete Datasets and Processed Files

The complete datasets and processed files used during development are provided through a public Google Drive folder.

**Google Drive:** [Complete - DATASETS](https://drive.google.com/drive/folders/1XyAvQ3e9YdKo9HzdqpEp5YydTBwMR4b9?usp=sharing)

The Google Drive folder contains the complete data used during development, including raw datasets and processed files.

The processed folder is organized as:

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

The complete processed files are provided for:

* Reference
* Reproducibility
* Further experimentation
* Inspecting the processed datasets and generated features

They are **not required to run the application**.

### GitHub vs Google Drive

The important distinction is:

```text
GitHub
└── data/processed/train_features.parquet
    └── 100 rows → used by SHAP

Google Drive
└── processed/features/train_features.parquet
    └── Complete original training feature dataset
```

The GitHub version is intentionally trimmed to 100 rows for the application's SHAP background.

The complete feature files are available only through the Google Drive dataset folder.

---

# Inference Pipeline

All predictions are handled through:

```text
model/inference.py
```

The pipeline:

1. Receives article text, source, and author
2. Generates the required model features
3. Runs the trained model
4. Calculates class probabilities
5. Calculates confidence
6. Determines review priority
7. Generates SHAP explanations using the 100-row background file

The same inference pipeline is used for single-article and batch analysis.

---

# Installation

## 1. Clone the Repository

```bash
git clone "URL of this repo"
cd misinfo-detector
```

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Download NLTK Resources

```bash
python3 -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

# Required Application Files

The following files are required to run the application:

```text
models/calibrated_model.joblib
data/credibility_lookups.json
data/feature_manifest.json
data/processed/train_features.parquet
```

The last file is the **100-row SHAP background dataset** included in the repository.

The complete raw and processed datasets are not required for normal application usage.

---

# Data Setup

No dataset download is required to run the existing application.

The trained model and required application artifacts are already provided in the repository.

For reference or further experimentation, the complete datasets are available through the public Google Drive folder documented above.

---

# Running the Application

The application uses:

* FastAPI backend
* Streamlit frontend

## 1. Start the Backend

```bash
cd api
python -m uvicorn main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/
```

Expected:

```json
{"status":"ok"}
```

## 2. Start the Frontend

From the project root:

```bash
streamlit run app/app.py
```

Frontend:

```text
http://localhost:8501
```

---

# Application Workflow

### Single Article

```text
Enter Article
      ↓
Generate Features
      ↓
Model Prediction
      ↓
Probability + Confidence
      ↓
Review Priority
      ↓
SHAP Explanation
      ↓
Store Prediction
      ↓
Human Review
```

### Batch Analysis

```text
Upload CSV
    ↓
Validate Columns
    ↓
Run Inference for Each Article
    ↓
Store Results
    ↓
Display Batch Results
```

---

# API Endpoints

| Endpoint                          | Purpose                    |
| --------------------------------- | -------------------------- |
| `GET /`                           | Health check               |
| `POST /predict`                   | Predict a single article   |
| `POST /predict/batch/csv`         | Analyze CSV batch          |
| `GET /queue`                      | Retrieve review queue      |
| `GET /batches`                    | Retrieve batch information |
| `POST /predictions/{id}/feedback` | Submit reviewer feedback   |

---

# Database

The application uses:

```text
data/app.db
```

It stores:

* Predictions
* Confidence
* Class probabilities
* Review status
* Batch information
* Reviewer feedback

Reviewer feedback is stored separately so that the original model prediction is not overwritten.

---

# Project Structure

```text
misinfo-detector/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │
│   ├── processed/
│   │   └── train_features.parquet   # 100-row SHAP background
│   │
│   ├── credibility_lookups.json
│   ├── feature_manifest.json
│   ├── app.db
│   └── NOTES.md
│
├── model/
│   └── inference.py
│
├── models/
│   └── calibrated_model.joblib
│
├── notebooks/
├── requirements.txt
└── README.md
```

---

# Model Performance

## Validation Set

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 92.70% |
| Precision | 91.25% |
| Recall    | 92.82% |
| F1        | 92.03% |
| ROC-AUC   | 96.97% |

## Test Set

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 93.08% |
| Precision | 92.12% |
| Recall    | 92.47% |
| F1        | 92.29% |
| ROC-AUC   | 97.23% |

An XGBoost comparison model was also evaluated during development.

---

# Limitations

* The model learns patterns from its training data and does not independently verify claims.
* Predictions should not be treated as definitive fact-checking results.
* Source and author credibility signals depend on available metadata.
* Unknown sources and authors have limited historical information.
* Performance may vary across domains, topics, writing styles, and newer information.
* Human review remains important for high-impact decisions.

---

# Future Improvements

Potential improvements include:

* More recent and diverse datasets
* Improved source and author verification
* Propagation and social-context features
* Multilingual support
* Transformer-based classification models
* External evidence retrieval
* Reviewer analytics
* Model monitoring and drift detection

---

# Documentation

* `README.md` — application setup, architecture, usage, API, and model information
* `data/NOTES.md` — dataset sources, preprocessing, feature structure, and data artifacts
