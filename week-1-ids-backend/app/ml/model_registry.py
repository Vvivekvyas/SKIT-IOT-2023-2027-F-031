"""
Model registry: a single point of truth for "which trained model is live".

Why this exists: your proposal trains 5 candidates (RF, SVM, Decision Tree,
XGBoost, hybrid FT-Transformer+Autoencoder) and picks the best one (Module 2).
Instead of hardcoding a model class into the API, the API always loads
whatever file `settings.ACTIVE_MODEL_PATH` points to. Swapping models after
comparison = drop a new .pkl in app/ml/artifacts/ and update .env. No
redeploy of API logic needed.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings


class ModelNotLoadedError(RuntimeError):
    pass


@lru_cache
def get_model() -> Any:
    path = Path(settings.ACTIVE_MODEL_PATH)
    if not path.exists():
        raise ModelNotLoadedError(
            f"No trained model found at {path}. "
            "Train a model (scripts/train_models.py) and save it there first."
        )
    return joblib.load(path)


@lru_cache
def get_feature_columns() -> list[str]:
    """
    The exact ordered list of feature names the model was trained on.
    Produced by the Dataset Loading Pipeline task and saved as JSON so
    inference can align incoming feature dicts to the right column order.
    """
    path = Path(settings.MODEL_FEATURE_LIST_PATH)
    if not path.exists():
        raise ModelNotLoadedError(
            f"No feature column manifest found at {path}. "
            "This should be produced during preprocessing/feature selection."
        )
    with open(path) as f:
        return json.load(f)
