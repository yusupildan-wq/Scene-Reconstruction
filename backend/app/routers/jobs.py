import asyncio
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Job, JobStatus, Project
from app.orchestrator import run_pipeline
from app.schemas import ComputeCapabilitiesOut, JobOut
from app.storage import get_storage
from app.executors import LocalNvidiaExecutor, RunPodExecutor

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
                     execution_mode: str = Form("runpod"),
                     session: AsyncSession = Depends(get_session)):
    if await session.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if not video.filename or not (video.content_type or "").startswith("video/"):
        raise HTTPException(status_code=415, detail="Please upload a video file")
    if execution_mode not in {"local_nvidia", "runpod"}:
        raise HTTPException(status_code=422, detail="execution_mode must be local_nvidia or runpod")
    provider = LocalNvidiaExecutor() if execution_mode == "local_nvidia" else RunPodExecutor()
    capability = await asyncio.to_thread(provider.validate)
    if not capability.available:
        raise HTTPException(status_code=422, detail=capability.detail)
    job = Job(project_id=project_id, status=JobStatus.PENDING, input_storage_key="",
              execution_mode=execution_mode, progress_percent=0, stage_detail="Upload received", stage_artifacts={})
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


@router.get("/compute/capabilities", response_model=ComputeCapabilitiesOut)
async def compute_capabilities():
    local, runpod = await asyncio.gather(
        asyncio.to_thread(LocalNvidiaExecutor().validate),
        asyncio.to_thread(RunPodExecutor().validate),
    )
    return {"local_nvidia": local, "runpod": runpod}


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
    # progress_percent only ever moves forward within a run (see orchestrator._set's
    # max()), so a failed attempt's high-water mark must be cleared here or it leaks
    # into the retry's UI as a stale percentage that doesn't match the resumed stage.
    job.progress_percent = 0
    await session.commit()
    await session.refresh(job)
    background_tasks.add_task(run_pipeline, job.id)
    return job_out(job)


async def _artifact(job_id: uuid.UUID, cameras: bool, session: AsyncSession):
    job = await session.get(Job, job_id)
    key = (job.camera_storage_key if cameras else job.output_storage_key) if job else None
    if not key or job.status != JobStatus.COMPLETE:
        raise HTTPException(status_code=404, detail="Artifact is not ready")
    path = get_storage().path(key)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact is missing")
    return FileResponse(path, media_type="application/json" if cameras else "application/octet-stream")


@router.get("/jobs/{job_id}/scene.ply")
async def scene(job_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    return await _artifact(job_id, False, session)


@router.get("/jobs/{job_id}/scene_cameras.json")
async def cameras(job_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    return await _artifact(job_id, True, session)
