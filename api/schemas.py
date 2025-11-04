from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str
    return_probabilities: bool = False


class BatchItem(BaseModel):
    text: str
    return_probabilities: bool = False


class BatchPredictionRequest(BaseModel):
    items: List[BatchItem]


class IntentScore(BaseModel):
    label: str
    score: float


class IntentPrediction(BaseModel):
    intent: str
    confidence: float
    intents: List[IntentScore] = Field(default_factory=list)
    all_probabilities: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status: str
    device: str


class ModelInfo(BaseModel):
    model_name: str
    intents: Dict[str, int]
    max_length: int
    device: str
