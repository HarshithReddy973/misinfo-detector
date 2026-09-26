# Misinformation Detector — Intra-IIT Hackathon 2026

## Setup (run once)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API (dummy model — Day 1)
```bash
cd api
uvicorn main:app --reload --port 8000
```
Test it:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Breaking: scientists confirm shocking discovery!!!"}'
```
Docs auto-generated at http://localhost:8000/docs

## Repo structure
```
data/        # schema.py (shared contract), cleaning scripts, raw/ + processed/ data
api/         # FastAPI app — main.py (endpoints), schemas.py (request/response contracts)
model/       # training scripts, saved model artifacts
app/         # dashboard (Streamlit/React)
notebooks/   # EDA, split strategy, experiments
```

## Team roles (Day 1)
- Person A: `data/schema.py` (shared contract) + FakeNewsNet cleaning
- Person B: LIAR cleaning + train/test split strategy (see `data/NOTES.md` once A pushes it)
- Person C: this repo skeleton + `api/` dummy endpoint

## Data setup (do this before running anything that needs data)
`data/raw/` and `data/processed/` are gitignored on purpose — raw
datasets aren't ours to redistribute, and generated files bloat the repo.
Every teammate regenerates them locally:

1. Download the raw datasets yourself:
   - ISOT: `True.csv`, `Fake.csv` → `data/raw/isot/`
   - LIAR: `train.tsv`, `valid.tsv`, `test.tsv` → `data/raw/liar/`
   - FakeNewsNet (Kaggle mdepak/fakenewsnet): the 4 `*_news_content.csv`
     files → `data/raw/fakenewsnet/`
2. From the repo root, run:
   ```bash
   python data/clean_isot.py
   python data/clean_liar.py
   python data/clean_fakenewsnet_main.py
   python data/merge_all.py
   ```
3. This produces `data/processed/combined_clean.parquet` — the file the
   model track trains on and Person B splits into train/test.

## Status
- [x] Day 1: schema locked, dummy API running, FakeNewsNet cleaned
- [ ] Day 2: baseline model, dashboard shell
- [ ] Day 3: real model + SHAP explainability, HITL actions
- [ ] Day 4: full integration, evaluation, batch analysis
- [ ] Day 5: docs, demo video, polish
