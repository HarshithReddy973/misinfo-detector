"""
API request/response contracts. Keep these aligned with data/schema.py and
model/inference.py's output.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="RAW article/post text -- do NOT pre-clean it, "
                                                        "the model's linguistic features expect raw_text")
    title: Optional[str] = ""
    source: Optional[str] = ""
    author: Optional[str] = ""


class ExplanationFeature(BaseModel):
    feature: str            # e.g. "linguistic_signal", "exclam_count", "semantic_signal"
    contribution: float     # signed SHAP-style value, + pushes toward "real", - pushes toward "fake"


class SignalStrengths(BaseModel):
    linguistic: float
    semantic: float
    credibility: float


class PredictResponse(BaseModel):
    prediction_id: Optional[str] = None    # DB id -- needed to submit reviewer feedback on this item later
    label: str                       # "real" | "fake" | "uncertain" (uncertain = low-confidence band, not a trained class)
    confidence: float                # calibrated, 0-1 -- confidence in whichever of real/fake the model voted for
    review_priority: str             # "uncertain" | "moderate" | "high" -- for sorting the reviewer queue
    prob_real: float
    prob_fake: float
    explanation: List[ExplanationFeature] = []
    signal_strengths: Optional[SignalStrengths] = None   # |contribution| per block -- which block dominates
    main_signal: Optional[str] = None                      # "linguistic" | "semantic" | "credibility" -- the dominant block
    main_signal_direction: Optional[str] = None            # "real" | "fake" | "neutral" -- which way the dominant block pushed
    source_credibility: Optional[float] = None    # 1 - source_fake_ratio (higher = more credible)
    source_known: Optional[bool] = None            # False = source wasn't in the training credibility DB
    author_known: Optional[bool] = None


class FeedbackRequest(BaseModel):
    action: str                          # "confirm" | "dismiss" | "relabel"
    corrected_label: Optional[str] = None   # required when action == "relabel"
    note: Optional[str] = None


class BatchItemResult(BaseModel):
    row_index: int
    text: str
    prediction: PredictResponse
    prediction_id: str


class BatchSummary(BaseModel):
    batch_id: str
    total_items: int
    label_counts: dict
