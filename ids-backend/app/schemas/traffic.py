"""
Pydantic schemas — the contract between the API and the outside world.
Keep these separate from the ORM models (app/models) so the DB shape
can change without breaking the API, and vice versa.
"""
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TrafficFeaturesIn(BaseModel):
    """
    What the client sends to /predict.
    This is intentionally loose (dict of feature_name -> value) because the
    exact feature set is finalized in the Dataset Loading Pipeline task —
    inference.py maps this dict onto the model's expected feature order.
    """
    source_ip: str | None = None
    destination_ip: str | None = None
    protocol: str | None = None
    features: dict[str, float]


class PredictionOut(BaseModel):
    predicted_label: str
    confidence: float
    model_used: str
    is_attack: bool


class TrafficRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_ip: str | None
    destination_ip: str | None
    protocol: str | None
    predicted_label: str
    confidence: float
    model_used: str
    created_at: datetime


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    traffic_record_id: int
    severity: str
    message: str
    created_at: datetime


class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
