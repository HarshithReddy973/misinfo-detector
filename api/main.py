"""
Prediction API -- real model (model/inference.py) + persistence (db.py) for
the reviewer queue and human-in-the-loop feedback.

Run with:  uvicorn main:app --reload --port 8000   (from inside api/)
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from schemas import (
    PredictRequest, PredictResponse, ExplanationFeature, SignalStrengths,
    FeedbackRequest, BatchSummary,
)

from model import inference
import db

db.init_db()

app = FastAPI(title="Misinformation Detector API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "shap_enabled": inference._shap_background is not None,
    }


def _direction(value: float) -> str:
    if value > 0:
        return "real"
    elif value < 0:
        return "fake"
    return "neutral"


def _build_explanation(shap_result) -> list[ExplanationFeature]:
    if shap_result is None:
        return []
    rows = [
        ExplanationFeature(feature="linguistic_signal", contribution=shap_result["linguistic_signal"]),
        ExplanationFeature(feature="semantic_signal", contribution=shap_result["semantic_signal"]),
        ExplanationFeature(feature="credibility_signal", contribution=shap_result["credibility_signal"]),
    ]
    for feat_name, value in shap_result["top_linguistic"]:
        if abs(value) < 1e-6:
            continue
        rows.append(ExplanationFeature(feature=feat_name, contribution=value))
    return rows


def _run_prediction(text: str, source: str, author: str) -> tuple[PredictResponse, dict]:
    """Shared by /predict and batch processing. Returns (response_model, raw_result_dict)."""
    try:
        result = inference.predict(text, source or "", author or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    shap_result = result["shap"]
    signal_strengths = None
    main_signal = None
    main_signal_direction = None

    if shap_result is not None:
        signal_strengths = SignalStrengths(
            linguistic=shap_result["linguistic_strength"],
            semantic=shap_result["semantic_strength"],
            credibility=shap_result["credibility_strength"],
        )
        strengths = {
            "linguistic": shap_result["linguistic_strength"],
            "semantic": shap_result["semantic_strength"],
            "credibility": shap_result["credibility_strength"],
        }
        main_signal = max(strengths, key=strengths.get)
        main_signal_value = {
            "linguistic": shap_result["linguistic_signal"],
            "semantic": shap_result["semantic_signal"],
            "credibility": shap_result["credibility_signal"],
        }[main_signal]
        main_signal_direction = _direction(main_signal_value)

    response = PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        review_priority=result["review_priority"],
        prob_real=result["prob_real"],
        prob_fake=result["prob_fake"],
        explanation=_build_explanation(shap_result),
        signal_strengths=signal_strengths,
        main_signal=main_signal,
        main_signal_direction=main_signal_direction,
        source_credibility=round(1 - result["source_fake_ratio"], 4),
        source_known=result["source_known"],
        author_known=result["author_known"],
    )
    return response, result


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    response, result = _run_prediction(req.text, req.source, req.author)
    explanation_list = [e.model_dump() for e in response.explanation]
    pred_id = db.save_prediction(req.text, req.title, req.source, req.author, result, explanation_list)
    response.prediction_id = pred_id
    return response


# --- Batch analysis ---

@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(requests: list[PredictRequest]) -> list[PredictResponse]:
    """JSON-list batch endpoint -- convenient for testing/scripting. For the
    PS's file-upload batch requirement, use /predict/batch/csv instead."""
    batch_id = db.create_batch(filename=None)
    results = []
    for req in requests:
        response, result = _run_prediction(req.text, req.source, req.author)
        explanation_list = [e.model_dump() for e in response.explanation]
        pred_id = db.save_prediction(req.text, req.title, req.source, req.author, result, explanation_list, batch_id=batch_id)
        response.prediction_id = pred_id
        results.append(response)
    return results


@app.post("/predict/batch/csv", response_model=BatchSummary)
async def predict_batch_csv(file: UploadFile = File(...)):
    """Upload a CSV with at least a 'text' column (optional 'source', 'author', 'title').
    Runs every row through the model and stores results under one batch_id."""
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't parse CSV: {e}")

    if "text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'text' column")

    batch_id = db.create_batch(file.filename)

    for _, row in df.iterrows():
        text = str(row.get("text", "") or "")
        if not text.strip():
            continue
        source = str(row.get("source", "") or "")
        author = str(row.get("author", "") or "")
        title = str(row.get("title", "") or "")
        response, result = _run_prediction(text, source, author)
        explanation_list = [e.model_dump() for e in response.explanation]
        db.save_prediction(text, title, source, author, result, explanation_list, batch_id=batch_id)

    return BatchSummary(**db.get_batch_summary(batch_id))


@app.get("/batches/{batch_id}/export.csv")
def export_batch_csv(batch_id: str):
    rows = db.get_predictions(batch_id=batch_id, limit=100000)
    if not rows:
        raise HTTPException(status_code=404, detail="Batch not found or empty")
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.csv"},
    )


# --- Reviewer queue / human-in-the-loop ---

@app.get("/batches")
def list_batches():
    """Batch groups + a pseudo-group for ungrouped single-analysis items, each
    with counts -- used by the Review Queue's group list before drilling in."""
    return db.get_batches()


@app.get("/queue")
def get_queue(
    review_status: str = Query(default=None),
    label: str = Query(default=None),
    source: str = Query(default=None),
    batch_id: str = Query(default=None),
    no_batch: bool = Query(default=False),
    sort_by: str = Query(default="confidence"),
    sort_dir: str = Query(default="asc"),
    limit: int = Query(default=200),
):
    return db.get_predictions(
        review_status=review_status, label=label, source=source, batch_id=batch_id, no_batch=no_batch,
        sort_by=sort_by, sort_dir=sort_dir, limit=limit,
    )


@app.get("/predictions/{prediction_id}")
def get_prediction_detail(prediction_id: str):
    pred = db.get_prediction(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Not found")
    pred["feedback_history"] = db.get_feedback_history(prediction_id)
    return pred


@app.post("/predictions/{prediction_id}/feedback")
def submit_feedback(prediction_id: str, req: FeedbackRequest):
    if not db.get_prediction(prediction_id):
        raise HTTPException(status_code=404, detail="Prediction not found")
    try:
        feedback_id = db.add_feedback(prediction_id, req.action, req.corrected_label, req.note)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"feedback_id": feedback_id, "status": "ok"}
