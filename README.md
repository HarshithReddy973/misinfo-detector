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

The system returns:

* Real / Fake prediction
* Confidence score
* Probability of Real
* Probability of Fake
* Review priority
* Source and author credibility information
* SHAP-based explanation when feature data is available

### Batch Analysis

Upload a CSV file to analyze multiple articles at once.

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

The system scores every row and provides aggregate results along with CSV export.

### Human Review Queue

The Review Queue allows reviewers to inspect analyzed content.

Reviewers can:

* Confirm a prediction
* Dismiss a prediction
* Relabel an item

Reviewer actions are stored separately in an append-only feedback table.

The original model prediction is never overwritten.

---

# System Architecture

```text
                    User
                     │
          ┌──────────┴──────────┐
          │                     │
   Single Article          CSV Batch
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
              Streamlit UI
                     │
                  HTTP
                     │
                     ▼
                FastAPI
                     │
                     ▼
             Inference Module
                     │
                     ▼
              Trained Model
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Prediction  Confidence   SHAP
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
                Final Result
                     │
                     ▼
                 SQLite DB
```

### Main Components

| Component      | Technology                     | Purpose                           |
| -------------- | ------------------------------ | --------------------------------- |
| Frontend       | Streamlit                      | User interface                    |
| Backend        | FastAPI                        | API and application logic         |
| Inference      | Python                         | Centralized prediction pipeline   |
| Model          | Calibrated Logistic Regression | Classification                    |
| Embeddings     | `all-MiniLM-L6-v2`             | Semantic representation           |
| Explainability | SHAP                           | Prediction explanations           |
| Database       | SQLite                         | Predictions and reviewer feedback |

---

# Machine Learning

The primary model is a **Calibrated Logistic Regression** classifier.

The model uses three feature groups:

```text
17 Linguistic Features
        +
384 Semantic Features
        +
4 Credibility Features
        =
405 Model Features
```

## Linguistic Features

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

These features capture writing style and surface-level characteristics.

## Semantic Features

The system uses:

```text
all-MiniLM-L6-v2
```

to generate a **384-dimensional semantic embedding** for the input text.

## Credibility Features

The model also uses:

```text
source_fake_ratio
source_known
author_fake_ratio
author_known
```

These provide historical source/author signals when such information is available.

Credibility information is treated as additional evidence rather than proof that an article is true or false.

---

# Prediction and Confidence

The model is trained on two classes:

```text
Real
Fake
```

The class with the higher probability becomes the raw prediction.

A confidence threshold is then applied:

```text
Confidence < 0.60
        │
        ▼
   Uncertain


Confidence ≥ 0.60
        │
        ▼
   Real / Fake
```

`Uncertain` is therefore a **post-prediction display category**, not a third trained class.

---

# Explainability

Misinformation Detector uses **SHAP** to explain model predictions.

The explanation provides information about signals contributing to the prediction, including:

```text
Linguistic Signals
Semantic Signals
Credibility Signals
```

The system can also identify influential individual linguistic features.

SHAP explanations describe the behavior of the trained model; they do not independently verify the factual claims in an article.

### SHAP Data Requirement

SHAP explanations require:

```text
data/processed/train_features.parquet
```

If this file is unavailable, **predictions still work**, but the explanation section is not generated.

---

# Inference Pipeline

The main prediction logic is centralized in:

```text
model/inference.py
```

The inference module:

1. Receives article text and optional metadata.
2. Generates the required features.
3. Loads the trained model.
4. Produces Real/Fake probabilities.
5. Determines the prediction.
6. Calculates confidence.
7. Determines review priority.
8. Generates the SHAP explanation when the required feature data is available.

Both single-article and batch analysis use the same inference pipeline.

---

# Data Pipeline

The project combines multiple misinformation datasets and converts them into a common schema.

The general pipeline is:

```text
Raw Datasets
      ↓
Dataset-specific Cleaning
      ↓
Label Standardization
      ↓
Common Schema
      ↓
Merge
      ↓
Train / Validation / Test Split
      ↓
Feature Generation
      ↓
Model Training
```

The detailed data documentation is available in:

```text
data/NOTES.md
```

It covers:

* Dataset sources
* Label processing
* Cleaning
* Common schema
* Merging
* Dataset splitting
* Credibility statistics
* Feature generation
* Data outputs

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/HarshithReddy973/misinfo-detector.git
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

## 4. Download Required NLTK Data

Run this once after installation:

```bash
python3 -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

# Data Setup

Processed data is not required to be regenerated if you only want to run the application.

The trained model and supporting artifacts are already included in the repository.

If you need to regenerate the processed datasets, place the raw datasets in:

```text
data/raw/
```

Expected locations:

```text
data/raw/
├── isot/
├── liar/
└── fakenewsnet/
```

Then run the data-processing scripts from the repository root:

```bash
python data/clean_isot.py
python data/clean_liar.py
python data/clean_fakenewsnet_main.py
python data/merge_all.py
python data/split_dataset.py
```

This generates the processed train/validation/test data and calibration data.

The processed data directories are gitignored.

For the complete data-processing details, see:

```text
data/NOTES.md
```

---

# Required Model Files

The application uses the following trained artifacts:

```text
models/
└── calibrated_model.joblib

data/
├── credibility_lookups.json
└── feature_manifest.json
```

### `calibrated_model.joblib`

The trained calibrated Logistic Regression model.

### `credibility_lookups.json`

Training-derived source and author credibility statistics.

### `feature_manifest.json`

The feature structure and ordering expected by the model.

These artifacts allow the application to perform inference without retraining the model.

---

# Running the Application

The application consists of two services:

```text
FastAPI Backend
       +
Streamlit Frontend
```

Run them in **two separate terminals**.

## Terminal 1 — Backend

From the repository root:

```bash
cd api
python -m uvicorn main:app --reload --port 8000
```

The backend should start on:

```text
http://localhost:8000
```

You can verify the API with:

```bash
curl http://localhost:8000/
```

A successful response should contain a status such as:

```json
{"status":"ok"}
```

## Terminal 2 — Frontend

From the repository root:

```bash
streamlit run app/app.py
```

The Streamlit application normally runs at:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# Application Workflow

## Single Article

```text
User enters article text
        ↓
Optional source / author
        ↓
Analyze
        ↓
Streamlit
        ↓
FastAPI
        ↓
Inference Module
        ↓
Feature Generation
        ↓
Trained Model
        ↓
Prediction + Confidence
        ↓
SHAP Explanation
        ↓
Result
        ↓
SQLite
```

## Batch Analysis

```text
User uploads CSV
        ↓
CSV validation
        ↓
FastAPI batch endpoint
        ↓
Same inference pipeline
        ↓
Prediction for every row
        ↓
Aggregate results
        ↓
CSV export
        ↓
SQLite storage
```

## Review Workflow

```text
Prediction
     ↓
Review Queue
     ↓
Reviewer Action
     ↓
Confirm / Dismiss / Relabel
     ↓
Append-only Feedback Log
```

The original prediction remains unchanged.

---

# API Endpoints

The FastAPI backend provides endpoints for:

| Endpoint                     | Purpose                  |
| ---------------------------- | ------------------------ |
| `/`                          | API health/status        |
| `/predict`                   | Analyze a single article |
| `/predict/batch/csv`         | Analyze a CSV batch      |
| `/queue`                     | Access the review queue  |
| `/batches`                   | Access batch information |
| `/predictions/{id}/feedback` | Submit reviewer feedback |

The exact request and response schemas are defined in the API implementation.

---

# Database

The application uses SQLite:

```text
data/app.db
```

The database is generated automatically when required.

It stores:

* Predictions
* Confidence and probability information
* Review status
* Batch information
* Reviewer feedback

Reviewer feedback is stored separately from the original prediction.

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
│   ├── processed/
│   ├── clean_isot.py
│   ├── clean_liar.py
│   ├── clean_fakenewsnet_main.py
│   ├── merge_all.py
│   ├── split_dataset.py
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
│
├── requirements.txt
└── README.md
```

---

# Model Performance

The final Calibrated Logistic Regression model achieved the following results on the prepared validation and test sets:

| Dataset Split | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------- | -------: | --------: | -----: | -------: | ------: |
| Validation    |   92.70% |    91.25% | 92.82% |   92.03% |  96.97% |
| Test          |   93.08% |    92.12% | 92.47% |   92.29% |  97.23% |

An XGBoost model was also evaluated as a comparison model.

These results represent performance on the project's prepared datasets and should not be interpreted as guaranteed performance on unseen real-world news.

---

# Limitations

* The model learns patterns from its training data and may not generalize to every domain or topic.
* A prediction does not establish the factual truth of an article.
* Source and author signals depend on available historical metadata.
* Unknown sources and authors provide limited credibility information.
* Dataset characteristics can affect measured performance.
* Human review remains important, especially for uncertain or high-impact content.

---

# Future Improvements

Potential improvements include:

* Adding more diverse and recent datasets
* Improving source and author verification
* Incorporating propagation and social-context signals
* Improving multilingual support
* Exploring transformer-based classification models
* Adding external evidence retrieval
* Improving reviewer analytics
* Adding model monitoring and drift detection

---

# Documentation

### Main Documentation

This `README.md` covers:

* System overview
* Features
* Architecture
* Machine-learning pipeline
* Installation
* Data setup
* Application setup
* API
* Review workflow
* Model performance
* Limitations

### Data Documentation

Detailed information about the datasets and data-processing pipeline is available in:

```text
data/NOTES.md
```

### Architecture Diagram

Add the final system architecture diagram in the **System Architecture** section above.
