import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Job, JobStatus, Project
from app.orchestrator import run_pipeline
from app.schemas import JobOut
from app.storage import get_storage

router = APIRouter(tags=["jobs"])


@router.post("/projects/{project_id}/jobs", response_model=JobOut, status_code=201)
async def create_job(
    project_id: uuid.UUID,
    video: UploadFile,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    job = Job(project_id=project_id, status=JobStatus.PENDING, input_storage_key="")
    session.add(job)
    await session.flush()  # assigns job.id without committing yet

    input_key = f"projects/{project_id}/jobs/{job.id}/input.mp4"
    video_bytes = await video.read()
    get_storage().save(input_key, video_bytes)

    job.input_storage_key = input_key
    await session.commit()
    await session.refresh(job)

    background_tasks.add_task(run_pipeline, job.id)
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/projects/{project_id}/jobs", response_model=list[JobOut])
async def list_jobs(project_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc())
    )
    return result.scalars().all()
