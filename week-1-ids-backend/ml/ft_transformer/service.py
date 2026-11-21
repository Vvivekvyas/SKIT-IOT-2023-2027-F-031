import hmac
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .data_prep import FTDataPrep
from .model import load_model

MAX_RECORDS = 256
MAX_STRING_LEN = 128


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1, max_length=MAX_RECORDS)


class PredictResponse(BaseModel):
    task: str
    n_records: int
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    classes: Optional[List[str]] = None


def validate_records(records, prep):
    expected = set(prep.feature_cols)
    errors = []
    for i, record in enumerate(records):
        keys = set(record.keys())
        missing = expected - keys
        extra = keys - expected
        if missing:
            errors.append(f"record {i}: missing {sorted(missing)}")
        if extra:
            errors.append(f"record {i}: unexpected {sorted(extra)}")
        if missing or extra:
            continue
        for c in prep.numeric_cols:
            v = record[c]
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errors.append(f"record {i}: '{c}' must be a number")
            elif not math.isfinite(v):
                errors.append(f"record {i}: '{c}' must be finite")
        for c in prep.categorical_cols:
            v = record[c]
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                errors.append(f"record {i}: '{c}' must be a scalar")
            elif len(str(v)) > MAX_STRING_LEN:
                errors.append(f"record {i}: '{c}' is too long")
        if len(errors) >= 20:
            break
    return errors


def create_app(artifact_dir, api_key=None):
    key = api_key or os.environ.get("FT_API_KEY")
    if not key:
        raise RuntimeError("FT_API_KEY is not set")
    artifact_dir = Path(artifact_dir)
    prep = FTDataPrep.load(artifact_dir)
    model = load_model(artifact_dir / "model.pt")
    app = FastAPI(title="FT-Transformer Model Service", docs_url=None, redoc_url=None, openapi_url=None)

    def require_key(x_api_key: Optional[str] = Header(default=None)):
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), key.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/schema", dependencies=[Depends(require_key)])
    def schema():
        return {
            "task": prep.task,
            "numeric_features": prep.numeric_cols,
            "categorical_features": prep.categorical_cols,
            "classes": prep.classes,
            "max_records": MAX_RECORDS,
        }

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
    def predict(body: PredictRequest):
        errors = validate_records(body.records, prep)
        if errors:
            raise HTTPException(status_code=422, detail=errors)
        df = pd.DataFrame(body.records, columns=prep.feature_cols)
        x_num, x_cat = prep.transform_features(df)
        with torch.no_grad():
            out = model(x_num, x_cat)
        if prep.task == "classification":
            probs = torch.softmax(out, dim=1)
            labels = [prep.classes[i] for i in probs.argmax(dim=1).tolist()]
            return PredictResponse(
                task=prep.task,
                n_records=len(labels),
                predictions=labels,
                probabilities=probs.tolist(),
                classes=prep.classes,
            )
        values = out.squeeze(-1).tolist()
        return PredictResponse(task=prep.task, n_records=len(values), predictions=values)

    return app


def app_factory():
    root = Path(__file__).resolve().parents[2]
    artifact_dir = os.environ.get("FT_ARTIFACT_DIR", str(root / "ml" / "artifacts" / "ft_transformer"))
    return create_app(artifact_dir)
