# Project Summary — Misinformation Detector (Intra-IIT Hackathon 2026)

For anyone picking this up mid-build — here's what's been done, in order,
and why each decision was made.

## 1. Datasets collected
Three sources, each with a different shape and reliability:
- **ISOT** (`True.csv`/`Fake.csv`) — full articles, ~44k rows, but no real
  `source`/`author`/`timestamp` fields (only a topic tag, deliberately not
  used as `source`).
- **LIAR** (`train/valid/test.tsv`) — ~12.8k short political statements,
  6-way graded labels, ships its **own official train/valid/test split**.
- **FakeNewsNet (main)** — the small Shu et al. BuzzFeed+PolitiFact set
  (~420 rows total). Genuinely has real `source`/`author`/`publish_date`
  fields, but too small to carry training volume — used for metadata
  quality, not volume.
- **FakeNewsNet-raw** (gossipcop/politifact ID+URL lists) — **deliberately
  excluded**. No article text, only URLs and tweet IDs; scraping them
  wasn't worth the time cost in a 5-day build.

## 2. Unified schema + per-dataset cleaning
`data/schema.py` defines one shape every dataset gets mapped into
(`text`, `raw_text`, `title`, `source`, `author`, `timestamp`, `label`,
`share_count`, `reply_count`, `dataset_source`, `official_split`).
`clean_isot.py`, `clean_liar.py`, `clean_fakenewsnet_main.py` each map
their raw files into this shape, filling in `"unknown"`/`""`/`0` where a
field doesn't exist for that dataset (e.g. ISOT has no real `source`).

**Where labels actually come from** (non-obvious, worth knowing if
debugging a labeling bug): ISOT and FakeNewsNet get their label from
*which file/filename* a row came from, not a column inside the file.
LIAR's label comes from an actual column, run through a mapping.

## 3. Label design — binary train, one excluded class
LIAR's 6-way scale collapses to:
- `true`, `mostly-true` → `real`
- `false`, `pants-fire` → `fake`
- `half-true`, `barely-true` → `ambiguous`

**`ambiguous` rows are never trained or tested on.** They're pulled out
entirely at split time into a separate `calibration_eval.parquet` — used
once, after training, to check whether the model's confidence naturally
dips on statements that are genuinely hard to call, without ever having
seen them. "Uncertain" as a user-facing label is *not* a trained class —
it's derived at inference time from low model confidence (< 60%).

## 4. Merge + split
- `merge_all.py` combines all three cleaned datasets into
  `combined_clean.parquet` — the full, unfiltered record of everything.
- `split_dataset.py` then: pulls out `ambiguous` rows →
  `calibration_eval.parquet`; for the remaining real/fake pool, **honors
  LIAR's own official split** (never re-shuffled) and does a **stratified
  random split** for ISOT/FakeNewsNet (no usable source field to split by,
  so exact-duplicate text is dropped first to prevent leakage, then a
  seeded stratified split). Output: `train.parquet`, `valid.parquet`,
  `test.parquet`.

## 5. Feature engineering (`build_features.py`, not detailed here — see
that script directly) produced `train_features.parquet` etc., feeding:
- 17 linguistic features (word/char/sentence counts, sentiment, subjectivity,
  readability, punctuation/caps density, hedging words, etc.)
- 384-dim sentence embeddings (`all-MiniLM-L6-v2`)
- 4 credibility features (`source_fake_ratio`, `source_known`,
  `author_fake_ratio`, `author_known`) — looked up from
  `credibility_lookups.json`, built from train-split label ratios only.

`feature_manifest.json` records the canonical feature order and block
groupings (`linguistic` / `semantic_embedding` / `credibility`) — the API
loads this rather than hardcoding feature lists, so it can't silently
drift out of sync with the model.

**Known caveat:** the `source` credibility lookup mixes real publisher
domains (from FakeNewsNet) with US state names (LIAR's best available
proxy, since LIAR has no true publisher field) — worth mentioning in the
report rather than presenting it as one clean signal.

## 6. Model training (teammate's work)
A `CalibratedClassifierCV` (5-fold) wrapping a `scaler → classifier`
pipeline, trained on the binary real/fake pool. Outputs calibrated
probabilities, not raw scores.

## 7. Explainability
SHAP (`LinearExplainer`), averaged across the 5 calibration folds, against
a 100-row background sample from `train_features.parquet`. Explanation is
presented at two levels: **block-level signals** (linguistic / semantic /
credibility — which category of evidence dominated) and **top-5
individual linguistic feature contributions**. The "main signal" is
whichever block has the largest total |contribution|.

## 8. API (`api/main.py`, `model/inference.py`)
FastAPI backend. `model/inference.py` loads the model, embedder,
credibility lookups, and SHAP background **once at startup**, not per
request. `/predict` returns: `label` (`real`/`fake`/`uncertain` — uncertain
derived from confidence < 60%, not a trained class), `confidence`,
`review_priority` (`uncertain`/`moderate`/`high`), both class
probabilities, the SHAP explanation, and source/author credibility with
explicit `known` flags (so the frontend can show "this source wasn't in
our training data" instead of presenting a fallback number as if it were
specific).

## 9. Frontend (`app/app.py`)
Streamlit — a thin client with **zero model logic**. It only collects
input, calls the API over HTTP, and renders the response: prediction
banner, probability bar, credibility section (with unknown-source/author
caveats), the SHAP explanation with the main-signal summary sentence, and
signal-strength comparison. FastAPI and Streamlit are two independent
processes that only communicate over HTTP — this is why the frontend
could be built against a dummy hardcoded API response on Day 1, before any
model existed, and swapped to the real thing later without changing the
frontend code at all.

## Not done yet (next up)
- **Batch inference** — submit a dataset, get aggregate stats + CSV export
  (PS requirement, MVP-listed).
- **Human-in-the-loop review/decision tracker** — confirm/dismiss/relabel
  a flagged item, logged to a structured feedback store, **never
  overwriting the original prediction or label** (explicit PS
  requirement).
- **Evaluation write-up** — accuracy/precision/recall/F1/AUC on the real
  test set, plus a concrete false-positive/false-negative error analysis
  (judging-weighted, don't skip it).
- Documentation (README ✓ in this same drop), demo video, midterm report.
