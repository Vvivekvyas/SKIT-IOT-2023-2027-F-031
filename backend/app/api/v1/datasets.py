"""
Dataset ingestion endpoints.

POST /datasets/upload accepts a CSV, validates it, stores it on disk, and
kicks off a background task that counts rows and flips status to
"processed" (or "failed" on error) — simulating the handoff to the ML
preprocessing pipeline, which is Nancy's side of the system.

GET /datasets/{id}/status lets the caller poll for that result.
"""
import csv
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import CurrentUser, require_role
from app.db.database import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetStatusResponse, DatasetUploadResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])

ALLOWED_CONTENT_TYPES = {"text/csv", "application/vnd.ms-excel", "application/octet-stream"}
MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _process_dataset(dataset_id: str, file_path: str, db_url: str) -> None:
    """
    Runs in the background after the upload response is already sent.
    Counts rows and updates the dataset's status. Uses its own DB session
    since background tasks run outside the request's session lifecycle.
    """
    from app.db.database import SessionLocal  # local import avoids circulars

    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            return
        try:
            with open(file_path, newline="", encoding="utf-8", errors="replace") as f:
                row_count = sum(1 for _ in csv.reader(f)) - 1  # minus header
            dataset.rows = max(row_count, 0)
            dataset.status = "processed"
        except Exception:
            dataset.status = "failed"
        db.commit()
    finally:
        db.close()


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES and not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted",
        )

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{dataset_id}.csv")

    size = 0
    with open(file_path, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                os.remove(file_path)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit",
                )
            out.write(chunk)

    dataset = Dataset(id=dataset_id, name=file.filename, status="processing")
    db.add(dataset)
    db.commit()

    background_tasks.add_task(_process_dataset, dataset_id, file_path, settings.DATABASE_URL)

    return DatasetUploadResponse(dataset_id=dataset_id, status="processing")


@router.get("/{dataset_id}/status", response_model=DatasetStatusResponse)
def get_dataset_status(
    dataset_id: str,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return DatasetStatusResponse(
        dataset_id=dataset.id,
        status=dataset.status,
        rows=dataset.rows,
        uploaded_at=dataset.uploaded_at,
    )
