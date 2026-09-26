"""
Real model inference, refactored from predict_raw.py (the teammate's CLI
script) into an importable module the API loads once at startup.

Nothing about the feature engineering or SHAP logic changed from
predict_raw.py — this file just:
  1. Removes the CLI input()/print() parts.
  2. Loads the model/embedder/lookups/feature manifest ONCE at import
     time (module-level), not on every single prediction — SHAP needs a
     100-row background sample loaded from train_features.parquet, and
     re-reading that from disk on every API call would be slow enough to
     hurt a live demo.
  3. Uses feature_manifest.json for feature ordering/blocks instead of
     the hardcoded lists in predict_raw.py, so this stays correct
     automatically if the model track ever changes the feature set.
  4. Returns a plain dict instead of printing to console.

PATHS — adjust these constants if your actual file locations differ:
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import nltk

for _resource, _path in [
    ("vader_lexicon", "sentiment/vader_lexicon"),
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        print(f"[inference] downloading NLTK resource '{_resource}'...")
        ok = nltk.download(_resource, quiet=False)
        if not ok:
            raise RuntimeError(
                f"[inference] FAILED to download NLTK resource '{_resource}'. "
                f"Check your internet connection, then retry manually with: "
                f"python3 -c \"import nltk; nltk.download('{_resource}')\""
            )

# Resolved relative to THIS FILE, not the current working directory —
# so it works whether you run uvicorn from api/, from repo root, or
# anywhere else. Adjust these four paths if your files live elsewhere.
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "calibrated_model.joblib"
LOOKUP_PATH = ROOT / "data" / "credibility_lookups.json"
MANIFEST_PATH = ROOT / "data" / "feature_manifest.json"
TRAIN_FEATURES_PATH = ROOT / "data" / "processed" / "train_features.parquet"

CONFIDENCE_UNCERTAIN_CUTOFF = 0.60   # below this -> label shown as "uncertain", matches predict_raw.py's threshold
CONFIDENCE_MODERATE_CUTOFF = 0.80    # below this -> "moderate", used for review-priority framing


# ---- Load everything ONCE at import time ----
print(f"[inference] loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print("[inference] loading sentence-transformer embedder (all-MiniLM-L6-v2)")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

sia = SentimentIntensityAnalyzer()

with open(LOOKUP_PATH, "r") as f:
    lookups = json.load(f)
source_lookup = lookups["source"]
author_lookup = lookups["author"]

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)
FEATURE_COLS = manifest["feature_order"]
BLOCKS = manifest["blocks"]   # {"linguistic": [...], "semantic_embedding": [...], "credibility": [...]}

# global_fake_ratio: fallback for unknown source/author. predict_raw.py hardcoded
# this from the training run; recompute it from the lookup values themselves so
# it can't silently drift out of sync with whatever lookups you actually loaded.
_all_known_ratios = list(source_lookup.values()) + list(author_lookup.values())
GLOBAL_FAKE_RATIO = float(np.mean(_all_known_ratios)) if _all_known_ratios else 0.5

print(f"[inference] loaded {len(source_lookup)} sources, {len(author_lookup)} authors, "
      f"{len(FEATURE_COLS)} features, global_fake_ratio={GLOBAL_FAKE_RATIO:.4f}")

# SHAP background sample — loaded once, reused for every prediction's SHAP call
try:
    _train_features = pd.read_parquet(TRAIN_FEATURES_PATH)
    _shap_background = _train_features[FEATURE_COLS].sample(n=min(100, len(_train_features)), random_state=42)
    print(f"[inference] SHAP background: {len(_shap_background)} rows from {TRAIN_FEATURES_PATH}")
except FileNotFoundError:
    _shap_background = None
    print(f"[inference] WARNING: {TRAIN_FEATURES_PATH} not found — SHAP explanations will be disabled. "
          f"Predictions will still work, just without the explanation field.")


def get_features(text: str, source: str, author: str):
    """
    IMPORTANT: `text` here must be the RAW, uncleaned text — feature_manifest.json's
    notes say linguistic features were built from raw_text, not the cleaned `text`
    column. Only the embedding step gets a URL-stripped copy internally. Don't pass
    already-cleaned text in here or you'll be off-distribution from training.
    """
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = max(1, len(TextBlob(text).sentences))
    avg_sentence_len = word_count / sentence_count
    avg_word_len = char_count / max(1, word_count)

    words = text.lower().split()
    vocab_diversity = len(set(words)) / max(1, word_count)

    punct_count = sum(c in ".,!?;:" for c in text)
    punct_density = punct_count / max(1, char_count)

    caps_count = sum(c.isupper() for c in text)
    caps_density = caps_count / max(1, char_count)

    exclam_count = text.count("!")
    question_count = text.count("?")
    url_count = text.lower().count("http")
    hashtag_count = text.count("#")
    mention_count = text.count("@")

    hedging_words = ["may", "might", "could", "possibly", "perhaps", "allegedly", "reportedly"]
    hedging_count = sum(text.lower().count(w) for w in hedging_words)

    sentiment_compound = sia.polarity_scores(text)["compound"]
    subjectivity = TextBlob(text).sentiment.subjectivity
    readability = textstat.flesch_reading_ease(text)

    linguistic = [
        word_count, char_count, sentence_count, avg_sentence_len, avg_word_len,
        vocab_diversity, punct_density, caps_density, exclam_count, question_count,
        url_count, hashtag_count, mention_count, hedging_count,
        sentiment_compound, subjectivity, readability,
    ]

    clean_text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    embedding = embedder.encode([clean_text])[0].tolist()

    source_key = (source or "").lower().strip()
    author_key = (author or "").lower().strip()

    if source_key in source_lookup:
        source_fake_ratio = source_lookup[source_key]
        source_known = 1
    else:
        source_fake_ratio = GLOBAL_FAKE_RATIO
        source_known = 0

    if author_key in author_lookup:
        author_fake_ratio = author_lookup[author_key]
        author_known = 1
    else:
        author_fake_ratio = GLOBAL_FAKE_RATIO
        author_known = 0

    credibility = [source_fake_ratio, source_known, author_fake_ratio, author_known]
    features = linguistic + embedding + credibility

    return features, source_fake_ratio, source_known, author_fake_ratio, author_known


def get_shap_explanation(features):
    """Returns block-level signals + top-5 individual linguistic SHAP contributions.
    Returns None if the SHAP background couldn't be loaded (missing train_features.parquet)."""
    if _shap_background is None:
        return None

    X = pd.DataFrame([features], columns=FEATURE_COLS)

    all_shap_values = []
    for calibrated in model.calibrated_classifiers_:
        pipeline = calibrated.estimator
        scaler = pipeline.named_steps["scaler"]
        classifier = pipeline.named_steps["classifier"]

        background_scaled = scaler.transform(_shap_background)
        X_scaled = scaler.transform(X)

        explainer = shap.LinearExplainer(classifier, background_scaled)
        values = explainer.shap_values(X_scaled)[0]
        all_shap_values.append(values)

    shap_values = sum(all_shap_values) / len(all_shap_values)
    shap_dict = dict(zip(FEATURE_COLS, shap_values))

    def block_signal(block_name):
        vals = [shap_dict[f] for f in BLOCKS[block_name]]
        return float(sum(vals)), float(sum(abs(v) for v in vals))

    linguistic_signal, linguistic_strength = block_signal("linguistic")
    semantic_signal, semantic_strength = block_signal("semantic_embedding")
    credibility_signal, credibility_strength = block_signal("credibility")

    top_linguistic = sorted(
        ((f, float(shap_dict[f])) for f in BLOCKS["linguistic"]),
        key=lambda x: abs(x[1]), reverse=True,
    )[:5]

    return {
        "linguistic_signal": linguistic_signal,
        "semantic_signal": semantic_signal,
        "credibility_signal": credibility_signal,
        "linguistic_strength": linguistic_strength,
        "semantic_strength": semantic_strength,
        "credibility_strength": credibility_strength,
        "top_linguistic": top_linguistic,
    }


def predict(text: str, source: str = "", author: str = "") -> dict:
    """Main entry point the API calls. Returns everything the frontend needs."""
    features, source_fake_ratio, source_known, author_fake_ratio, author_known = get_features(text, source, author)

    X = pd.DataFrame([features], columns=FEATURE_COLS)
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    prob_fake = float(probabilities[classes.index("fake")])
    prob_real = float(probabilities[classes.index("real")])

    raw_prediction = "real" if prob_real >= prob_fake else "fake"
    confidence = max(prob_real, prob_fake)

    # This is the confidence-bucketing plan from earlier: "uncertain" is NOT a
    # trained class, it's derived here from a low-confidence band.
    if confidence < CONFIDENCE_UNCERTAIN_CUTOFF:
        display_label = "uncertain"
    else:
        display_label = raw_prediction

    if confidence < CONFIDENCE_UNCERTAIN_CUTOFF:
        review_priority = "uncertain"
    elif confidence < CONFIDENCE_MODERATE_CUTOFF:
        review_priority = "moderate"
    else:
        review_priority = "high"

    shap_result = get_shap_explanation(features)

    return {
        "raw_prediction": raw_prediction,     # "real" | "fake" — what the model actually voted, before uncertain-bucketing
        "label": display_label,               # "real" | "fake" | "uncertain" — what to SHOW the user
        "confidence": round(confidence, 4),
        "review_priority": review_priority,   # "uncertain" | "moderate" | "high" — for the reviewer queue's sort order
        "prob_real": round(prob_real, 4),
        "prob_fake": round(prob_fake, 4),
        "source_fake_ratio": round(source_fake_ratio, 4),
        "source_known": bool(source_known),
        "author_fake_ratio": round(author_fake_ratio, 4),
        "author_known": bool(author_known),
        "shap": shap_result,   # None if train_features.parquet wasn't found
    }
