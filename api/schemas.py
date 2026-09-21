"""
API request/response contracts. Keep these aligned with data/schema.py —
the 'label' and feature names should match what the model track produces.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Article or post body")
    title: Optional[str] = ""
    source: Optional[str] = "unknown"
    author: Optional[str] = "unknown"


class ExplanationFeature(BaseModel):
    feature: str            # e.g. "subjectivity", "source_credibility"
    contribution: float     # signed SHAP-style value, + pushes toward "fake"


class PredictResponse(BaseModel):
    label: str                       # "real" | "fake" | "uncertain"
    confidence: float                # calibrated, 0-1
    explanation: List[ExplanationFeature] = []
    source_credibility: Optional[float] = None
