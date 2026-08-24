import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    EXTRACTING_FRAMES = "extracting_frames"
    DISPATCHED = "dispatched"
    RUNNING_SFM = "running_sfm"
    TRAINING = "training"
    COMPLETE = "complete"
    FAILED = "failed"
    PREPARING_FRAMES = "preparing_frames"
    VGGT_GEOMETRY = "vggt_geometry"
    GAUSSIAN_OPTIMIZATION = "gaussian_optimization"
    FINALIZING = "finalizing"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    jobs: Mapped[list["Job"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"))
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, values_callable=lambda enum_cls: [member.value for member in enum_cls]),
        default=JobStatus.PENDING,
    )
    stage_detail: Mapped[str | None] = mapped_column(String(255), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)

    input_storage_key: Mapped[str] = mapped_column(String(1024))
    frame_count: Mapped[int | None] = mapped_column(default=None)
    selected_frame_count: Mapped[int | None] = mapped_column(default=None)

    output_storage_key: Mapped[str | None] = mapped_column(String(1024), default=None)
    camera_storage_key: Mapped[str | None] = mapped_column(String(1024), default=None)
    runpod_job_id: Mapped[str | None] = mapped_column(String(255), default=None)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    stage_artifacts: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Holds evaluation results once complete: held-out PSNR/SSIM/LPIPS, training
    # wall-clock time, GPU memory, etc. Populated by the worker, never fabricated
    # by the backend.
    metrics: Mapped[dict | None] = mapped_column(JSONB, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    project: Mapped["Project"] = relationship(back_populates="jobs")
