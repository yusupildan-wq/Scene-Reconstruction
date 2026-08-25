import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import JobStatus


class ProjectCreate(BaseModel):
    name: str


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    status: JobStatus
    stage_detail: str | None
    error_message: str | None
    frame_count: int | None
    selected_frame_count: int | None
    output_storage_key: str | None
    camera_storage_key: str | None
    progress_percent: int
    execution_mode: str
    scene_url: str | None = None
    cameras_url: str | None = None
    metrics: dict | None
    created_at: datetime
    updated_at: datetime


class ProviderCapabilityOut(BaseModel):
    available: bool
    detail: str
    vram_gb: float | None = None


class ComputeCapabilitiesOut(BaseModel):
    local_nvidia: ProviderCapabilityOut
    runpod: ProviderCapabilityOut
