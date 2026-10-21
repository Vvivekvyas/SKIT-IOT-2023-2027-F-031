"""
Dataset endpoints.

POST   /datasets/upload              upload a CSV (analyst, admin)
GET    /datasets                     list datasets, newest first (analyst, admin)
GET    /datasets/{id}                dataset details (analyst, admin)
GET    /datasets/{id}/status         processing status (analyst, admin)
GET    /datasets/{id}/validation     data-pipeline validation report (analyst, admin)
GET    /datasets/{id}/preview        first rows of the CSV (analyst, admin)
POST   /datasets/{id}/process        re-run processing (analyst, admin)
DELETE /datasets/{id}                delete dataset + file (admin only)

Upload checks: .csv extension, allowed content type, not empty, not binary,
header line present, size limit. A bad upload never leaves a file behind.

Processing runs in the background: it counts the rows, then runs the ML
team's validator (src/data) through app/services/data_pipeline.py.
Status values:
  processing  still running
  processed   finished (is_valid true, or null if validation was skipped)
  invalid     finished, but the file failed validation or is not a known dataset
  failed      the file could not be read as a CSV
"""
import csv
import json
import logging
import os
import re
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Path,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import CurrentUser, require_role
from app.db.database import SessionLocal, get_db
from app.models.dataset import Dataset
from app.schemas.dataset import (
    DatasetDetailResponse,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetStatusResponse,
    DatasetUploadResponse,
    DatasetValidationResponse,
)
from app.services.data_pipeline import KNOWN_DATASET_TYPES, PipelineUnavailable, run_validation

logger = logging.getLogger("ids.api")

router = APIRouter(prefix="/datasets", tags=["datasets"])

ALLOWED_CONTENT_TYPES = {"text/csv", "application/vnd.ms-excel", "application/octet-stream"}
MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Dataset IDs are UUIDs we generate. Anything else is rejected up front.
DatasetId = Annotated[
    str,
    Path(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="Dataset ID (UUID)",
    ),
]


# ---------- helpers ----------

def _file_path(dataset_id: str) -> str:
    return os.path.join(settings.UPLOAD_DIR, f"{dataset_id}.csv")


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _clean_filename(raw: str | None) -> str:
    """Keep only the file name: no folders, no control characters, max 255 chars."""
    name = os.path.basename((raw or "").replace("\\", "/"))
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    return name[:255]


def _validate_csv_file(path: str) -> None:
    """Basic sanity checks on a saved upload. Raises 400 with a clear reason."""
    if os.path.getsize(path) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty")
    with open(path, "rb") as f:
        head = f.read(8192)
    if b"\x00" in head:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "File does not look like a text CSV (binary content found)",
        )
    lines = head.decode("utf-8", errors="replace").splitlines()
    if not lines or not lines[0].strip():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "The first line of the CSV (column names) is empty",
        )


def _get_dataset_or_404(db: Session, dataset_id: str) -> Dataset:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return dataset


def _to_detail(dataset: Dataset) -> DatasetDetailResponse:
    path = _file_path(dataset.id)
    size = os.path.getsize(path) if os.path.exists(path) else None
    return DatasetDetailResponse(
        dataset_id=dataset.id,
        name=dataset.name,
        status=dataset.status,
        rows=dataset.rows,
        size_bytes=size,
        uploaded_at=dataset.uploaded_at,
        dataset_type=dataset.dataset_type,
        is_valid=dataset.is_valid,
        message=dataset.message,
    )


def _process_dataset(dataset_id: str) -> None:
    """
    Runs in the background after the response is already sent.
    Counts rows, runs the data-pipeline validation, and saves the outcome.
    Uses its own DB session since background tasks run outside the request's
    session lifecycle.
    """
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            return
        try:
            path = _file_path(dataset_id)
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                total_rows = max(sum(1 for _ in csv.reader(f)) - 1, 0)  # minus header
            dataset.rows = total_rows

            try:
                result = run_validation(path, dataset.dataset_type, total_rows)
            except PipelineUnavailable as exc:
                # Validation can't run here, but the upload itself is fine.
                logger.warning("Validation skipped for %s: %s", dataset_id, exc)
                dataset.status = "processed"
                dataset.is_valid = None
                dataset.message = f"Validation skipped: {exc}."
            else:
                dataset.dataset_type = result["dataset_type"]
                dataset.is_valid = result["valid"]
                dataset.message = result["message"]
                dataset.validation_report = (
                    json.dumps(result["report"], default=str) if result["report"] else None
                )
                dataset.status = "processed" if result["valid"] else "invalid"
        except Exception:
            logger.exception("Processing failed for dataset %s", dataset_id)
            dataset.status = "failed"
            dataset.is_valid = None
            dataset.message = "Could not read the file as a CSV."
        db.commit()
    finally:
        db.close()


# ---------- endpoints ----------

@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    dataset_type: Annotated[
        str | None,
        Form(description="CICIDS2017 or CIC-IoT2023. Leave empty to detect it from the columns."),
    ] = None,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    dataset_type = (dataset_type or "").strip() or None
    if dataset_type is not None and dataset_type not in KNOWN_DATASET_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"dataset_type must be one of: {', '.join(KNOWN_DATASET_TYPES)}",
        )

    filename = _clean_filename(file.filename)
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .csv files are accepted")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported content type: {file.content_type}",
        )

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    dataset_id = str(uuid.uuid4())
    file_path = _file_path(dataset_id)

    try:
        size = 0
        with open(file_path, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit",
                    )
                out.write(chunk)

        _validate_csv_file(file_path)

        db.add(
            Dataset(id=dataset_id, name=filename, status="processing", dataset_type=dataset_type)
        )
        db.commit()
    except Exception:
        # Any failure: don't leave a half-saved file behind.
        _remove_quietly(file_path)
        raise

    background_tasks.add_task(_process_dataset, dataset_id)

    return DatasetUploadResponse(dataset_id=dataset_id, status="processing")


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    total = db.query(Dataset).count()
    items = (
        db.query(Dataset)
        .order_by(Dataset.uploaded_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return DatasetListResponse(
        datasets=[_to_detail(d) for d in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset(
    dataset_id: DatasetId,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    return _to_detail(_get_dataset_or_404(db, dataset_id))


@router.get("/{dataset_id}/status", response_model=DatasetStatusResponse)
def get_dataset_status(
    dataset_id: DatasetId,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    dataset = _get_dataset_or_404(db, dataset_id)
    return DatasetStatusResponse(
        dataset_id=dataset.id,
        status=dataset.status,
        rows=dataset.rows,
        uploaded_at=dataset.uploaded_at,
        is_valid=dataset.is_valid,
        message=dataset.message,
    )


@router.get("/{dataset_id}/validation", response_model=DatasetValidationResponse)
def get_dataset_validation(
    dataset_id: DatasetId,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    dataset = _get_dataset_or_404(db, dataset_id)
    if dataset.status == "processing":
        raise HTTPException(status.HTTP_409_CONFLICT, "Validation is still running")
    if not dataset.validation_report:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            dataset.message or "No validation report for this dataset",
        )
    return DatasetValidationResponse(
        dataset_id=dataset.id,
        dataset_type=dataset.dataset_type,
        status=dataset.status,
        is_valid=dataset.is_valid,
        report=json.loads(dataset.validation_report),
    )


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview_dataset(
    dataset_id: DatasetId,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    dataset = _get_dataset_or_404(db, dataset_id)
    path = _file_path(dataset.id)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset file is missing on the server"
        )
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        columns = next(reader, [])
        rows = [row for _, row in zip(range(limit), reader)]
    return DatasetPreviewResponse(dataset_id=dataset.id, columns=columns, rows=rows)


@router.post(
    "/{dataset_id}/process",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def reprocess_dataset(
    dataset_id: DatasetId,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("analyst")),
):
    dataset = _get_dataset_or_404(db, dataset_id)
    if dataset.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Dataset is already being processed"
        )
    if not os.path.exists(_file_path(dataset.id)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset file is missing on the server"
        )
    dataset.status = "processing"
    dataset.rows = None
    dataset.is_valid = None
    dataset.validation_report = None
    dataset.message = None
    db.commit()
    background_tasks.add_task(_process_dataset, dataset.id)
    return DatasetUploadResponse(dataset_id=dataset.id, status="processing")


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: DatasetId,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("admin")),
):
    dataset = _get_dataset_or_404(db, dataset_id)
    _remove_quietly(_file_path(dataset.id))
    db.delete(dataset)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
