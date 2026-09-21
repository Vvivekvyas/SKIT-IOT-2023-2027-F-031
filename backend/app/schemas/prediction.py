from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    prediction_id: str
    classification: str
    confidence: float

    model_config = {"from_attributes": True}


class AlertSummary(BaseModel):
    id: str
    classification: str
    confidence: float
    timestamp: datetime

    model_config = {"from_attributes": True}


class AlertListResponse(BaseModel):
    alerts: list[AlertSummary]
    next_cursor: str | None = None
