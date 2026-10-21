"""
Dataset table — tracks uploaded traffic datasets, their processing status
and the result of the data-pipeline validation.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="processing")
    # "processing" | "processed" | "invalid" | "failed"
    rows: Mapped[int] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Added in Week 8 (data pipeline integration)
    dataset_type: Mapped[str | None] = mapped_column(String, nullable=True)
    # "CICIDS2017" | "CIC-IoT2023" | None (not recognised / not checked)
    is_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # None = not validated
    validation_report: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON text
    message: Mapped[str | None] = mapped_column(String, nullable=True)  # short note for the user
