"""
Turns a raw feature dict into a classification result.
"""
import numpy as np

from app.core.config import settings
from app.ml.model_registry import get_feature_columns, get_model
from app.schemas.traffic import PredictionOut

ATTACK_LABELS = {"DoS", "Probe", "R2L", "U2R"}


def align_features(raw_features: dict[str, float], feature_columns: list[str]) -> np.ndarray:
    """Reorders/pads incoming features to match the exact column order used at training time."""
    row = [raw_features.get(col, 0.0) for col in feature_columns]
    return np.array(row, dtype=float).reshape(1, -1)


def predict(raw_features: dict[str, float]) -> PredictionOut:
    model = get_model()
    feature_columns = get_feature_columns()
    X = align_features(raw_features, feature_columns)

    predicted_label = str(model.predict(X)[0])

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(X)))

    model_name = getattr(settings, "ACTIVE_MODEL_PATH", "unknown")

    return PredictionOut(
        predicted_label=predicted_label,
        confidence=round(confidence, 4),
        model_used=model_name,
        is_attack=predicted_label in ATTACK_LABELS,
    )
