"""Thin RunPod adapter. Application state remains in our database/storage."""
from __future__ import annotations

import requests

from app.config import settings

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


class DispatchNotConfigured(RuntimeError):
    pass


def submit_gpu_job(payload: dict) -> str:
    if not (settings.runpod_api_key and settings.runpod_endpoint_id):
        raise DispatchNotConfigured(
            "RunPod is not configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID, "
            "or use GPU_BACKEND=local for the no-cost product demo."
        )
    response = requests.post(
        f"{RUNPOD_API_BASE}/{settings.runpod_endpoint_id}/run",
        headers={"Authorization": f"Bearer {settings.runpod_api_key}"},
        json={"input": payload}, timeout=30,
    )
    response.raise_for_status()
    return response.json()["id"]


def get_gpu_job(runpod_job_id: str) -> dict:
    response = requests.get(
        f"{RUNPOD_API_BASE}/{settings.runpod_endpoint_id}/status/{runpod_job_id}",
        headers={"Authorization": f"Bearer {settings.runpod_api_key}"}, timeout=30,
    )
    response.raise_for_status()
    return response.json()
