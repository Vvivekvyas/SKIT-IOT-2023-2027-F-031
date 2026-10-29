from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

MAX_FEATURES = 500
MAX_FEATURE_NAME_LENGTH = 100
MAX_FEATURE_TEXT_LENGTH = 200


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Flat name -> value only. Nested objects/lists are rejected.
    features: dict[str, float | int | bool | str | None]

    @field_validator("features")
    @classmethod
    def check_features(cls, features):
        if not features:
            raise ValueError("features must not be empty")
        if len(features) > MAX_FEATURES:
            raise ValueError(f"features can have at most {MAX_FEATURES} entries")
        for name, value in features.items():
            if not name.strip() or len(name) > MAX_FEATURE_NAME_LENGTH:
                raise ValueError(
                    f"feature names must be 1-{MAX_FEATURE_NAME_LENGTH} characters"
                )
            if isinstance(value, str) and len(value) > MAX_FEATURE_TEXT_LENGTH:
                raise ValueError(
                    f"text values must be at most {MAX_FEATURE_TEXT_LENGTH} characters"
                )
        return features


class PredictResponse(BaseModel):
    prediction_id: str
    classification: str
    confidence: float
    anomaly_score: float | None = None
    is_alert: bool
    model_version: str | None = None

    model_config = {"from_attributes": True}


class ModelInfoResponse(BaseModel):
    ready: bool
    model_version: str | None = None
    message: str | None = None
    class_names: list[str] = []
    feature_names: list[str] = []


class AlertSummary(BaseModel):
    id: str
    classification: str
    confidence: float
    timestamp: datetime

    model_config = {"from_attributes": True}


class AlertListResponse(BaseModel):
    alerts: list[AlertSummary]
    next_cursor: str | None = None
