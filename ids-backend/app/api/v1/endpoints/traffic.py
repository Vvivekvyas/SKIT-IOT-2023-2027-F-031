from fastapi import APIRouter, Depends
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.traffic import Alert, TrafficRecord
from app.schemas.traffic import AlertOut, TrafficRecordOut

router = APIRouter()


@router.get("/records", response_model=list[TrafficRecordOut])
def list_traffic_records(
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[TrafficRecord]:
    """Backs the dashboard's 'table view: traffic records with predicted label'."""
    return (
        db.query(TrafficRecord)
        .order_by(desc(TrafficRecord.created_at))
        .limit(limit)
        .all()
    )


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(limit: int = 50, db: Session = Depends(get_db)) -> list[Alert]:
    """Backs the dashboard's 'alert section highlighting detected intrusions'."""
    return db.query(Alert).order_by(desc(Alert.created_at)).limit(limit).all()
