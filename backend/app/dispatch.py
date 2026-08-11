"""Dispatch a job's GPU stage (COLMAP + Gaussian Splatting training) to RunPod.

Not implemented yet -- this requires a RunPod account, an API key, and a built
worker image (see worker/), none of which exist yet. This raises rather than
faking a result: a job that can't actually run should be marked FAILED with an
honest error message, never silently marked COMPLETE.
"""

from __future__ import annotations

from app.config import settings


class DispatchNotConfigured(RuntimeError):
    pass


def dispatch_to_gpu_worker(job_id: str, video_url: str) -> str:
    """Submit the GPU stage to RunPod's serverless endpoint and return RunPod's job id.

    TODO once RunPod is set up:
      1. POST to https://api.runpod.ai/v2/{endpoint_id}/run with the video_url and
         job_id in the payload, Authorization: Bearer {runpod_api_key}.
      2. Store the returned RunPod job id on our Job row (runpod_job_id).
      3. Poll https://api.runpod.ai/v2/{endpoint_id}/status/{runpod_job_id} (or use a
         webhook, if RunPod delivers one) to update our Job.status as the worker
         progresses through RUNNING_SFM -> TRAINING -> COMPLETE/FAILED.
    """
    if not (settings.runpod_api_key and settings.runpod_endpoint_id):
        raise DispatchNotConfigured(
            "RunPod is not configured (RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID missing). "
            "The GPU worker has not been built or deployed yet -- see worker/README.md."
        )
    raise NotImplementedError("RunPod dispatch is not implemented yet.")
