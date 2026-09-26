# Misinformation Detector — Intra-IIT Hackathon 2026

NLP/Trust & Safety system that flags likely misinformation in news
articles/posts, with calibrated confidence and SHAP-based explanations,
for human reviewers — not an automated fact-checker.

See `TEAM_SUMMARY.md` for the full project history and design decisions.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First time only — download NLTK data (needs internet):
```bash
python3 -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('punkt_tab')"
```
> macOS + python.org install: if this fails with an SSL error, run
> `open "/Applications/Python 3.x/Install Certificates.command"` first.

## Data setup (only needed if regenerating the processed data)
`data/raw/` and `data/processed/` are gitignored — regenerate locally:
1. Download raw datasets: ISOT → `data/raw/isot/`, LIAR → `data/raw/liar/`,
   FakeNewsNet (mdepak/fakenewsnet on Kaggle) → `data/raw/fakenewsnet/`
2. From repo root:
   ```bash
   python data/clean_isot.py
   python data/clean_liar.py
   python data/clean_fakenewsnet_main.py
   python data/merge_all.py
   python data/split_dataset.py
   ```
This produces `train/valid/test.parquet` + `calibration_eval.parquet`.
The trained model, credibility lookups, and feature manifest are already
committed under `models/` and `data/` — you don't need to retrain to run
the app. **Exception:** SHAP explanations need `data/processed/train_features.parquet`
(from `build_features.py`) — without it, predictions still work, just
without the explanation section.

## Run the app (two terminals, both from repo root with venv active)

**Terminal 1 — backend:**
```bash
cd api
python -m uvicorn main:app --reload --port 8000
```
Wait for `Application startup complete.` Check it worked:
`curl http://localhost:8000/` → should return `{"status":"ok", ...}`.

**Terminal 2 — frontend:**
```bash
streamlit run app/app.py
```
Opens a browser tab (usually `localhost:8501`). Paste an article, optionally
a source/author, click Analyze.

## Repo structure
```
data/    # schema, cleaning/split scripts, credibility lookups, feature manifest
model/   # inference.py — loads the trained model once, real prediction logic
api/     # FastAPI backend — main.py (endpoints), schemas.py (request/response contracts)
app/     # Streamlit frontend — thin client, calls the API over HTTP
models/  # calibrated_model.joblib
notebooks/
```

## Status
- [x] Data pipeline (4 datasets → schema → clean → merge → split)
- [x] Model trained + calibrated, SHAP explainability
- [x] API wired to real model
- [x] Frontend (submission form, results, credibility, explanation)
- [ ] Batch inference + CSV export
- [ ] Human review / decision tracker (append-only feedback log)
- [ ] Evaluation write-up, midterm report, demo video
