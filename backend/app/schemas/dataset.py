from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    status: str


class DatasetStatusResponse(BaseModel):
    dataset_id: str
    status: str
    rows: int | None
    uploaded_at: datetime
    is_valid: bool | None = None
    message: str | None = None

    model_config = {"from_attributes": True}


class DatasetDetailResponse(BaseModel):
    dataset_id: str
    name: str
    status: str
    rows: int | None
    size_bytes: int | None
    uploaded_at: datetime
    dataset_type: str | None = None
    is_valid: bool | None = None
    message: str | None = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetDetailResponse]
    total: int
    limit: int
    offset: int


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[list[str]]


class DatasetValidationResponse(BaseModel):
    dataset_id: str
    dataset_type: str | None
    status: str
    is_valid: bool | None
    report: dict[str, Any]
