"""Resumable CPU/GPU reconstruction orchestration."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from app.config import settings
from app.db import SessionLocal
from app.dispatch import get_gpu_job, submit_gpu_job
from app.models import Job, JobStatus
from app.pipeline import extract_frames
from app.storage import S3Storage, get_storage
from sqlalchemy import select

logger = logging.getLogger(__name__)
SCRATCH_DIR = Path(settings.storage_local_path).parent / "scratch"
WORKER_URL_TTL_SECONDS = 24 * 60 * 60


def _runpod_progress(result: dict) -> dict:
    """Normalize RunPod's progress field across SDK/API response variants."""
    value = result.get("progress")
    if isinstance(value, list):
        value = value[-1] if value else None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _validate_viewer_artifacts(job: Job, storage) -> None:
    """Reject incomplete GPU output before a user enters a broken viewer."""
    if not job.output_storage_key or not storage.exists(job.output_storage_key):
        raise RuntimeError("Reconstruction completed without scene.ply")
    if not job.camera_storage_key or not storage.exists(job.camera_storage_key):
        raise RuntimeError("Reconstruction completed without camera metadata")
    try:
        metadata = json.loads(storage.read(job.camera_storage_key))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("Reconstruction produced invalid camera metadata") from error
    if metadata.get("coordinate_convention") not in {"opencv", "opengl"}:
        raise RuntimeError("Camera metadata is missing its coordinate convention")
    frames = metadata.get("frames")
    matrices = metadata.get("camera_to_world_matrices")
    if not (isinstance(frames, list) and frames) and not (isinstance(matrices, list) and matrices):
        raise RuntimeError("Camera metadata contains no training poses")


async def _set(job, session, status: JobStatus, progress: int, detail: str) -> None:
    job.status = status
    job.progress_percent = max(job.progress_percent or 0, min(100, progress))
    job.stage_detail = detail
    job.error_message = None
    await session.commit()


async def _prepare_frames(job, session, storage) -> list[str]:
    artifacts = dict(job.stage_artifacts or {})
    frame_keys = artifacts.get("frame_keys", [])
    if frame_keys and all(storage.exists(key) for key in frame_keys):
        await _set(job, session, JobStatus.PREPARING_FRAMES, 25, "Reusing prepared frames")
        return frame_keys
    await _set(job, session, JobStatus.PREPARING_FRAMES, 10, "Selecting sharp, useful frames")
    scratch = SCRATCH_DIR / str(job.id)
    scratch.mkdir(parents=True, exist_ok=True)
    video_path = scratch / "input.mp4"
    video_path.write_bytes(storage.read(job.input_storage_key))
    result = await asyncio.to_thread(extract_frames, video_path, scratch / "frames")
    if result.selected_frame_count < 2:
        raise RuntimeError(
            "The video did not contain enough sharp frames. Record a slower pass with the room well lit."
        )
    frame_keys = []
    for frame_path in result.frame_paths:
        key = f"projects/{job.project_id}/jobs/{job.id}/frames/{frame_path.name}"
        storage.save(key, frame_path.read_bytes())
        frame_keys.append(key)
    job.frame_count, job.selected_frame_count = result.total_frames_seen, result.selected_frame_count
    artifacts["frame_keys"] = frame_keys
    job.stage_artifacts = artifacts
    await session.commit()
    return frame_keys


async def _run_local_demo(job, session, storage) -> None:
    """No-cost full-flow adapter using the existing V3 result, never a GPU."""
    await _set(job, session, JobStatus.VGGT_GEOMETRY, 42, "Estimating cameras and room geometry")
    await asyncio.sleep(0.8)
    await _set(job, session, JobStatus.GAUSSIAN_OPTIMIZATION, 68, "Optimizing Gaussian appearance")
    await asyncio.sleep(0.8)
    source_ply = Path(settings.local_demo_scene_ply).resolve()
    source_cameras = Path(settings.local_demo_cameras_json).resolve()
    if not source_ply.is_file():
        raise RuntimeError(
            f"Local demo artifact not found at {source_ply}. Set LOCAL_DEMO_SCENE_PLY, "
            "or configure GPU_BACKEND=runpod for real reconstruction."
        )
    await _set(job, session, JobStatus.FINALIZING, 92, "Packaging viewer artifacts")
    storage.save(job.output_storage_key, source_ply.read_bytes())
    if source_cameras.is_file():
        storage.save(job.camera_storage_key, source_cameras.read_bytes())


async def _run_runpod(job, session, storage: S3Storage, frame_keys: list[str]) -> None:
    if not isinstance(storage, S3Storage):
        raise RuntimeError("GPU_BACKEND=runpod requires STORAGE_BACKEND=s3")
    if storage.exists(job.output_storage_key) and storage.exists(job.camera_storage_key):
        await _set(job, session, JobStatus.FINALIZING, 96, "Reusing completed viewer artifacts")
        return
    artifacts = dict(job.stage_artifacts or {})
    geometry_key = artifacts.get("geometry_key") or (
        f"projects/{job.project_id}/jobs/{job.id}/output/vggt_geometry.tar.gz"
    )
    artifacts["geometry_key"] = geometry_key
    job.stage_artifacts = artifacts
    await session.commit()
    if not job.runpod_job_id:
        await _set(job, session, JobStatus.VGGT_GEOMETRY, 32, "Queued on GPU")
        payload = {
            "job_id": str(job.id),
            "frame_urls": [storage.url_for(key, WORKER_URL_TTL_SECONDS) for key in frame_keys],
            "scene_upload_url": storage.upload_url_for(job.output_storage_key),
            "cameras_upload_url": storage.upload_url_for(job.camera_storage_key),
            "geometry_upload_url": storage.upload_url_for(geometry_key),
            "quality_profile": settings.reconstruction_quality_profile,
        }
        if storage.exists(geometry_key):
            payload["geometry_download_url"] = storage.url_for(geometry_key, WORKER_URL_TTL_SECONDS)
        job.runpod_job_id = await asyncio.to_thread(submit_gpu_job, payload)
        await session.commit()
    while True:
        result = await asyncio.to_thread(get_gpu_job, job.runpod_job_id)
        state = result.get("status")
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        update = _runpod_progress(result) or output
        stage = update.get("stage")
        detail = update.get("detail")
        progress = int(update.get("progress", 0) or 0)
        if stage == "gaussian_optimization":
            await _set(job, session, JobStatus.GAUSSIAN_OPTIMIZATION, progress or 65, detail or "Optimizing Gaussian appearance")
        elif stage == "finalizing":
            await _set(job, session, JobStatus.FINALIZING, progress or 92, detail or "Packaging viewer artifacts")
        elif state in {"IN_QUEUE", "IN_PROGRESS"}:
            await _set(job, session, JobStatus.VGGT_GEOMETRY, progress or 38, detail or "Estimating cameras and room geometry")
        if state == "COMPLETED":
            if output.get("error"):
                raise RuntimeError(str(output["error"]))
            if isinstance(output.get("metrics"), dict):
                job.metrics = output["metrics"]
                await session.commit()
            return
        if state in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise RuntimeError(result.get("error") or f"GPU job ended with {state}")
        await asyncio.sleep(settings.runpod_poll_seconds)


async def run_pipeline(job_id: uuid.UUID) -> None:
    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if job is None:
            return
        try:
            storage = get_storage()
            base = f"projects/{job.project_id}/jobs/{job.id}/output"
            job.output_storage_key = job.output_storage_key or f"{base}/scene.ply"
            job.camera_storage_key = job.camera_storage_key or f"{base}/scene_cameras.json"
            frame_keys = await _prepare_frames(job, session, storage)
            if settings.gpu_backend == "local":
                await _run_local_demo(job, session, storage)
            elif settings.gpu_backend == "runpod":
                await _run_runpod(job, session, storage, frame_keys)
            else:
                raise RuntimeError(f"Unknown GPU_BACKEND={settings.gpu_backend!r}")
            _validate_viewer_artifacts(job, storage)
            await _set(job, session, JobStatus.COMPLETE, 100, "Ready to explore")
        except Exception as exc:
            logger.exception("Pipeline failed for job %s", job_id)
            job.status = JobStatus.FAILED
            job.error_message = f"{type(exc).__name__}: {exc}"
            await session.commit()


async def recover_active_jobs() -> None:
    """Resume jobs after an API restart; every stage is artifact-aware."""
    active = [status for status in JobStatus if status not in {JobStatus.COMPLETE, JobStatus.FAILED}]
    async with SessionLocal() as session:
        result = await session.execute(select(Job.id).where(Job.status.in_(active)))
        for job_id in result.scalars():
            asyncio.create_task(run_pipeline(job_id))
