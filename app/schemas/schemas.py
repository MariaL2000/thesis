from pydantic import BaseModel
from typing import Optional, Dict, Any

class FileValidationResponse(BaseModel):
    filename: str
    format: str
    status: str
    path: str

class PipelineUploadResponse(BaseModel):
    message: str
    pipeline_status: Dict[str, Any]

    class Config:
        from_attributes = True
