"""
Prediction API -- wired to the REAL trained model (model/inference.py).

Run with:  uvicorn main:app --reload --port 8000   (from inside api/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictRequest, PredictResponse, ExplanationFeature, SignalStrengths

from model import inference

app = FastAPI(title="Misinformation Detector API", version="0.3.0")

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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = inference.predict(req.text, req.source or "", req.author or "")
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
        # same "strongest wins" logic predict_raw.py used -- whichever block
        # has the largest |contribution| is the one driving the prediction
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

    return PredictResponse(
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


@app.post("/predict/batch")
def predict_batch(requests: list[PredictRequest]) -> list[PredictResponse]:
    return [predict(r) for r in requests]
