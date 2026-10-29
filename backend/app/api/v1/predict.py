"""
Prediction endpoints.

POST /predict         classify one network flow (analyst, admin), rate limited
GET  /predict/model   which model is loaded and which features it expects (analyst, admin)

The request body is the flat feature dictionary validated in schemas/prediction.py.
Only the result (label, confidence, anomaly score, model version) is saved.
The submitted feature values are never logged or stored.
"""
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import CurrentUser, require_role
from app.core.limiter import limiter
from app.db.database import get_db
from app.models.prediction import Prediction
from app.schemas.prediction import ModelInfoResponse, PredictRequest, PredictResponse
from app.services.predictor import FeatureError, ModelUnavailable, is_normal, prediction_service

logger = logging.getLogger("ids.api")

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
@limiter.limit(settings.PREDICT_RATE_LIMIT)
def predict_flow(
    request: Request,  # required by slowapi
    payload: PredictRequest,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    if not prediction_service.ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The prediction model is not available yet",
        )
    try:
        result = prediction_service.predict(payload.features)
    except FeatureError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    except ModelUnavailable:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The prediction model is not available yet",
        )

    row = Prediction(
        id=str(uuid.uuid4()),
        classification=result["prediction"],
        confidence=result["confidence"],
        anomaly_score=result["anomaly_score"],
        model_version=result["model_version"],
    )
    db.add(row)
    db.commit()

    logger.info(
        "prediction id=%s user=%s label=%s model=%s",
        row.id, user.username, row.classification, row.model_version,
    )
    return PredictResponse(
        prediction_id=row.id,
        classification=row.classification,
        confidence=row.confidence,
        anomaly_score=row.anomaly_score,
        is_alert=not is_normal(row.classification),
        model_version=row.model_version,
    )


@router.get("/model", response_model=ModelInfoResponse)
def model_info(user: CurrentUser = Depends(require_role("analyst"))):
    ready = prediction_service.ready
    return ModelInfoResponse(
        ready=ready,
        model_version=prediction_service.model_version if ready else None,
        message=None if ready else prediction_service.unavailable_reason,
        class_names=prediction_service.class_names if ready else [],
        feature_names=prediction_service.feature_names if ready else [],
    )
