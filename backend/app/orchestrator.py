"""Runs a job's pipeline stages in the background after the upload request returns.

This is the async job pattern the project is built around: the HTTP request that
uploads a video must not block for however long reconstruction takes (minutes to
tens of minutes), so it does the minimum work to persist the input and create a Job
row, then returns immediately. Everything after that happens out-of-band and is
observed by the frontend polling GET /jobs/{id}, not by the original request.

In V0, FastAPI's BackgroundTasks stands in for what would become a real queue+worker
(RunPod's serverless queue, once dispatch.py is implemented) -- it runs in the same
process after the response is sent, which is fine for the CPU-only stage but is not
how the GPU stage will work once it's real.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from app.config import settings
from app.db import SessionLocal
from app.dispatch import DispatchNotConfigured, dispatch_to_gpu_worker
from app.models import Job, JobStatus
from app.pipeline import extract_frames
from app.storage import get_storage

logger = logging.getLogger(__name__)

SCRATCH_DIR = Path(settings.storage_local_path).parent / "scratch"


async def run_pipeline(job_id: uuid.UUID) -> None:
    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if job is None:
            logger.error("run_pipeline: job %s not found", job_id)
            return

        try:
            job.status = JobStatus.EXTRACTING_FRAMES
            await session.commit()

            storage = get_storage()
            video_bytes = storage.read(job.input_storage_key)

            job_scratch = SCRATCH_DIR / str(job_id)
            job_scratch.mkdir(parents=True, exist_ok=True)
            video_path = job_scratch / "input.mp4"
            video_path.write_bytes(video_bytes)

            result = extract_frames(video_path, job_scratch / "frames")
            job.frame_count = result.total_frames_seen
            job.selected_frame_count = result.selected_frame_count
            await session.commit()

            job.status = JobStatus.DISPATCHED
            await session.commit()

            dispatch_to_gpu_worker(job_id=str(job.id), video_url=job.input_storage_key)

        except (DispatchNotConfigured, NotImplementedError) as exc:
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            await session.commit()
        except Exception as exc:  # noqa: BLE001 -- surface any pipeline failure honestly
            logger.exception("Pipeline failed for job %s", job_id)
            job.status = JobStatus.FAILED
            job.error_message = f"{type(exc).__name__}: {exc}"
            await session.commit()
