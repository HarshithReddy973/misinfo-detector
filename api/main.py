"""
Dummy prediction API — Day 1 deliverable.
Returns a hardcoded but REALISTIC response shape so the dashboard (Person B/
product track) can build against it starting tomorrow, before the real model
exists. Swap the body of predict() for the real model call later — the
contract (PredictRequest -> PredictResponse) should not need to change.

Run with:  uvicorn main:app --reload --port 8000
Test with: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"Breaking: scientists confirm shocking discovery!!!"}'
"""
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictRequest, PredictResponse, ExplanationFeature

app = FastAPI(title="Misinformation Detector API", version="0.1.0")

# allow the dashboard (Streamlit/React, different port) to call this locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "note": "dummy model — real model not wired in yet"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    DUMMY LOGIC — replace this function body once the model track has a
    trained model. The response shape must stay the same so the frontend
    doesn't need changes.
    """
    # fake-but-plausible confidence so the dashboard has something to render
    confidence = round(random.uniform(0.55, 0.95), 2)
    label = random.choice(["real", "fake", "uncertain"])

    dummy_explanation = [
        ExplanationFeature(feature="subjectivity", contribution=0.21),
        ExplanationFeature(feature="source_credibility", contribution=-0.14),
        ExplanationFeature(feature="excessive_punctuation", contribution=0.09),
    ]

    return PredictResponse(
        label=label,
        confidence=confidence,
        explanation=dummy_explanation,
        source_credibility=round(random.uniform(0.3, 0.9), 2),
    )


@app.post("/predict/batch")
def predict_batch(requests: list[PredictRequest]) -> list[PredictResponse]:
    """Stub for the batch-analysis requirement — same dummy logic, looped."""
    return [predict(r) for r in requests]
