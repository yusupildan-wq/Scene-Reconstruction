import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Job, JobStatus, Project
from app.orchestrator import run_pipeline
from app.schemas import JobOut
from app.storage import LocalStorage, get_storage

router = APIRouter(tags=["jobs"])


def job_out(job: Job) -> JobOut:
    data = JobOut.model_validate(job)
    if job.status == JobStatus.COMPLETE and job.output_storage_key:
        data.scene_url = f"/jobs/{job.id}/scene.ply"
        if job.camera_storage_key:
            data.cameras_url = f"/jobs/{job.id}/scene_cameras.json"
    return data


@router.post("/projects/{project_id}/jobs", response_model=JobOut, status_code=201)
async def create_job(project_id: uuid.UUID, video: UploadFile, background_tasks: BackgroundTasks,
                     session: AsyncSession = Depends(get_session)):
    if await session.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if not video.filename or not (video.content_type or "").startswith("video/"):
        raise HTTPException(status_code=415, detail="Please upload a video file")
    job = Job(project_id=project_id, status=JobStatus.PENDING, input_storage_key="",
              progress_percent=0, stage_detail="Upload received", stage_artifacts={})
    session.add(job)
    await session.flush()
    key = f"projects/{project_id}/jobs/{job.id}/input.mp4"
    video.file.seek(0)
    get_storage().save_fileobj(key, video.file)
    job.input_storage_key = key
    await session.commit()
    await session.refresh(job)
    background_tasks.add_task(run_pipeline, job.id)
    return job_out(job)


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_out(job)


@router.get("/projects/{project_id}/jobs", response_model=list[JobOut])
async def list_jobs(project_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc()))
    return [job_out(job) for job in result.scalars().all()]


@router.post("/jobs/{job_id}/retry", response_model=JobOut)
async def retry_job(job_id: uuid.UUID, background_tasks: BackgroundTasks,
                    session: AsyncSession = Depends(get_session)):
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.FAILED:
        raise HTTPException(status_code=409, detail="Only failed jobs can be retried")
    job.status, job.error_message, job.stage_detail = JobStatus.PENDING, None, "Retrying from saved artifacts"
    job.runpod_job_id = None
    await session.commit()
    background_tasks.add_task(run_pipeline, job.id)
    return job_out(job)


async def _artifact(job_id: uuid.UUID, cameras: bool, session: AsyncSession):
    job = await session.get(Job, job_id)
    key = (job.camera_storage_key if cameras else job.output_storage_key) if job else None
    if not key or job.status != JobStatus.COMPLETE:
        raise HTTPException(status_code=404, detail="Artifact is not ready")
    storage = get_storage()
    if isinstance(storage, LocalStorage):
        path = storage._path(key)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact is missing")
        return FileResponse(path, media_type="application/json" if cameras else "application/octet-stream")
    return StreamingResponse(
        storage.iter_bytes(key),
        media_type="application/json" if cameras else "application/octet-stream",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/jobs/{job_id}/scene.ply")
async def scene(job_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    return await _artifact(job_id, False, session)


@router.get("/jobs/{job_id}/scene_cameras.json")
async def cameras(job_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    return await _artifact(job_id, True, session)
