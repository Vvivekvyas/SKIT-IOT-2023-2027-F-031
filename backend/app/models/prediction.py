"""
Prediction table — stores each classification result. Doubles as the source
for the /alerts endpoints (an "alert" is simply a non-Normal prediction).
Matches POST /predict, GET /predict/{id}, GET /alerts from api-specification.md.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    classification: Mapped[str] = mapped_column(String, nullable=False)
    # "Normal" | "DoS" | "Probe" | "R2L" | "U2R"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
