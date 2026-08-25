"""Resumable orchestration with provider-independent local artifacts."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tarfile
import uuid
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.executors import ExecutionRequest, LocalNvidiaExecutor, RunPodExecutor
from app.models import Job, JobStatus
from app.pipeline import extract_frames
from app.storage import get_storage

logger = logging.getLogger(__name__)
SCRATCH_DIR = Path(settings.storage_local_path).resolve().parent / "scratch"


def _validate_viewer_artifacts(job: Job, storage) -> None:
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
    if not metadata.get("frames") and not metadata.get("camera_to_world_matrices"):
        raise RuntimeError("Camera metadata contains no training poses")


async def _set(job: Job, session, status: JobStatus, progress: int, detail: str) -> None:
    job.status = status
    job.progress_percent = max(job.progress_percent or 0, min(100, progress))
    job.stage_detail = detail
    job.error_message = None
    await session.commit()


async def _prepare_frames(job: Job, session, storage) -> list[str]:
    artifacts = dict(job.stage_artifacts or {})
    frame_keys = artifacts.get("frame_keys", [])
    if frame_keys and all(storage.exists(key) for key in frame_keys):
        await _set(job, session, JobStatus.PREPARING_FRAMES, 25, "Reusing prepared frames")
        return frame_keys
    await _set(job, session, JobStatus.PREPARING_FRAMES, 10, "Selecting sharp, useful frames")
    scratch = SCRATCH_DIR / str(job.id) / "preprocessing"
    scratch.mkdir(parents=True, exist_ok=True)
    video_path = scratch / "input.mp4"
    video_path.write_bytes(storage.read(job.input_storage_key))
    result = await asyncio.to_thread(extract_frames, video_path, scratch / "frames")
    if result.selected_frame_count < 2:
        raise RuntimeError("The video did not contain enough sharp frames. Record a slower pass with the room well lit.")
    frame_keys = []
    for frame_path in result.frame_paths:
        key = f"projects/{job.project_id}/jobs/{job.id}/frames/{frame_path.name}"
        with frame_path.open("rb") as source:
            storage.save_fileobj(key, source)
        frame_keys.append(key)
    job.frame_count, job.selected_frame_count = result.total_frames_seen, result.selected_frame_count
    artifacts["frame_keys"] = frame_keys
    job.stage_artifacts = artifacts
    await session.commit()
    return frame_keys


def _restore_geometry(storage, key: str | None, scene_dir: Path) -> None:
    if not key or not storage.exists(key):
        return
    archive_path = scene_dir.parent / "saved_geometry.tar.gz"
    archive_path.write_bytes(storage.read(key))
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(scene_dir)


def _workspace(job: Job, storage, frame_keys: list[str]) -> tuple[Path, Path]:
    root = SCRATCH_DIR / str(job.id) / "reconstruction"
    scene_dir, result_dir = root / "scene", root / "result"
    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for index, key in enumerate(frame_keys):
        (images_dir / f"frame_{index:06d}.jpg").write_bytes(storage.read(key))
    _restore_geometry(storage, (job.stage_artifacts or {}).get("geometry_key"), scene_dir)
    return scene_dir, result_dir


async def _execute(job: Job, session, storage, frame_keys: list[str]):
    if storage.exists(job.output_storage_key) and storage.exists(job.camera_storage_key):
        await _set(job, session, JobStatus.FINALIZING, 98, "Reusing completed local artifacts")
        return
    scene_dir, result_dir = await asyncio.to_thread(_workspace, job, storage, frame_keys)
    loop = asyncio.get_running_loop()

    def report(stage: str, progress: int, detail: str) -> None:
        status = {
            "vggt_geometry": JobStatus.VGGT_GEOMETRY,
            "gaussian_optimization": JobStatus.GAUSSIAN_OPTIMIZATION,
            "finalizing": JobStatus.FINALIZING,
        }.get(stage, JobStatus.VGGT_GEOMETRY)
        asyncio.run_coroutine_threadsafe(_set(job, session, status, progress, detail), loop).result()

    def remember_pod(pod_id: str) -> None:
        async def save() -> None:
            job.runpod_job_id = pod_id
            await session.commit()
        asyncio.run_coroutine_threadsafe(save(), loop).result()

    executor = LocalNvidiaExecutor() if job.execution_mode == "local_nvidia" else RunPodExecutor(remember_pod)
    if job.execution_mode == "runpod" and job.runpod_job_id:
        try:
            await asyncio.to_thread(executor.terminate_existing, job.runpod_job_id)
        except Exception:
            logger.warning("Could not clean up previous pod %s", job.runpod_job_id, exc_info=True)
        job.runpod_job_id = None
        await session.commit()
    try:
        result = await asyncio.to_thread(executor.execute, ExecutionRequest(
            str(job.id), scene_dir, result_dir, settings.reconstruction_quality_profile,
        ), report)
    except Exception:
        geometry = result_dir / "vggt_geometry.tar.gz"
        if not geometry.is_file() and (scene_dir / "sparse").is_dir():
            result_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(geometry, "w:gz") as archive:
                archive.add(scene_dir / "sparse", arcname="sparse")
        if geometry.is_file():
            geometry_key = f"projects/{job.project_id}/jobs/{job.id}/output/vggt_geometry.tar.gz"
            with geometry.open("rb") as source:
                storage.save_fileobj(geometry_key, source)
            artifacts = dict(job.stage_artifacts or {})
            artifacts["geometry_key"] = geometry_key
            job.stage_artifacts = artifacts
            await session.commit()
        raise
    for path, key in ((result.scene_ply, job.output_storage_key), (result.cameras_json, job.camera_storage_key)):
        with path.open("rb") as source:
            storage.save_fileobj(key, source)
    geometry_key = f"projects/{job.project_id}/jobs/{job.id}/output/vggt_geometry.tar.gz"
    with result.geometry_archive.open("rb") as source:
        storage.save_fileobj(geometry_key, source)
    artifacts = dict(job.stage_artifacts or {})
    artifacts["geometry_key"] = geometry_key
    job.stage_artifacts = artifacts
    job.metrics = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    job.runpod_job_id = result.provider_job_id or job.runpod_job_id
    await session.commit()


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
            await _execute(job, session, storage, frame_keys)
            _validate_viewer_artifacts(job, storage)
            await _set(job, session, JobStatus.COMPLETE, 100, "Ready to explore")
        except Exception as exc:
            logger.exception("Pipeline failed for job %s", job_id)
            job.status = JobStatus.FAILED
            job.error_message = f"{type(exc).__name__}: {exc}"
            await session.commit()


async def recover_active_jobs() -> None:
    active = [status for status in JobStatus if status not in {JobStatus.COMPLETE, JobStatus.FAILED}]
    async with SessionLocal() as session:
        result = await session.execute(select(Job.id).where(Job.status.in_(active)))
        for job_id in result.scalars():
            asyncio.create_task(run_pipeline(job_id))
