from datetime import datetime

from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    status: str


class DatasetStatusResponse(BaseModel):
    dataset_id: str
    status: str
    rows: int | None
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class DatasetDetailResponse(BaseModel):
    dataset_id: str
    name: str
    status: str
    rows: int | None
    size_bytes: int | None
    uploaded_at: datetime


class DatasetListResponse(BaseModel):
    datasets: list[DatasetDetailResponse]
    total: int
    limit: int
    offset: int


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[list[str]]
