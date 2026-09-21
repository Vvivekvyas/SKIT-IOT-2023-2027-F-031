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
