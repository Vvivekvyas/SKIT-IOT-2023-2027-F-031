"""
Core ORM models.

TrafficRecord  -> one row per classified network flow (table view in dashboard)
Alert          -> raised automatically whenever a TrafficRecord is classified as an attack
User           -> admin login (Designs slide)
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class TrafficRecord(Base):
    __tablename__ = "traffic_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Raw/derived features used for the prediction are stored as JSON-ish text
    # for now; Module 2 (feature selection) will firm up the exact schema.
    source_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    destination_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    protocol: Mapped[str | None] = mapped_column(String(16), nullable=True)

    predicted_label: Mapped[str] = mapped_column(String(32))   # "Normal" | "DoS" | "Probe" | "R2L" | "U2R"
    confidence: Mapped[float] = mapped_column(Float)
    model_used: Mapped[str] = mapped_column(String(64))        # e.g. "RandomForest", "FT-Transformer+AE"

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    alerts: Mapped[list["Alert"]] = relationship(back_populates="traffic_record")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    traffic_record_id: Mapped[int] = mapped_column(ForeignKey("traffic_records.id"))
    severity: Mapped[str] = mapped_column(String(16), default="high")  # low | medium | high
    message: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    traffic_record: Mapped["TrafficRecord"] = relationship(back_populates="alerts")
