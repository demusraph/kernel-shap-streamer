from typing import List, Dict
from pydantic import BaseModel, Field

class FeatureImportance(BaseModel):
    feature_name: str
    shap_value: float
    contribution_pct: float

class ExplanationRequest(BaseModel):
    features: Dict[str, float]
    background_data: List[Dict[str, float]] = Field(..., min_length=1)
    nsamples: int = Field(default=100, ge=10, le=1000)

class ExplanationResponse(BaseModel):
    base_value: float
    attributions: List[FeatureImportance]
    latency_ms: float
    converged: bool