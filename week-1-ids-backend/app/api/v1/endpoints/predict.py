from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.ml.inference import predict as run_inference
from app.models.traffic import Alert, TrafficRecord
from app.schemas.traffic import PredictionOut, TrafficFeaturesIn

router = APIRouter()


@router.post("/predict", response_model=PredictionOut)
def classify_traffic(payload: TrafficFeaturesIn, db: Session = Depends(get_db)) -> PredictionOut:
    """
    Classifies a single network-traffic sample as Normal or an attack type,
    persists it, and raises an Alert row if it's malicious.
    This is the endpoint the dashboard and any live feed hits.
    """
    result = run_inference(payload.features)

    record = TrafficRecord(
        source_ip=payload.source_ip,
        destination_ip=payload.destination_ip,
        protocol=payload.protocol,
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        model_used=result.model_used,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    if result.is_attack:
        alert = Alert(
            traffic_record_id=record.id,
            severity="high",
            message=f"Detected {result.predicted_label} traffic from {payload.source_ip or 'unknown'}",
        )
        db.add(alert)
        db.commit()

    return result
