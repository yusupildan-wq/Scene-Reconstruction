"""Dispatch a job's GPU stage (COLMAP + Gaussian Splatting training) to RunPod.

Submits the job and returns immediately with RunPod's own job id -- it does not
wait for the GPU work to finish (that can take minutes), matching the async job
pattern used everywhere else in this backend. Polling RunPod for completion and
updating our Job row once it's done is not implemented yet -- that's the next gap
after this, tracked separately from "can we submit a job at all."
"""

from __future__ import annotations

import requests

from app.config import settings

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


class DispatchNotConfigured(RuntimeError):
    pass


def dispatch_to_gpu_worker(job_id: str, frame_urls: list[str], output_url: str) -> str:
    """Submit the GPU stage to RunPod's serverless endpoint. Returns RunPod's own
    job id (distinct from our Job.id), which we store so a later polling step can
    look up whether it's finished."""
    if not (settings.runpod_api_key and settings.runpod_endpoint_id):
        raise DispatchNotConfigured(
            "RunPod is not configured (RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID missing). "
            "The GPU worker has not been built or deployed yet -- see worker/README.md."
        )

    response = requests.post(
        f"{RUNPOD_API_BASE}/{settings.runpod_endpoint_id}/run",
        headers={"Authorization": f"Bearer {settings.runpod_api_key}"},
        json={"input": {"job_id": job_id, "frame_urls": frame_urls, "output_url": output_url}},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["id"]
