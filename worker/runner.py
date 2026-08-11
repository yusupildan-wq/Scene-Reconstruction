"""Skeleton of the GPU worker's job entrypoint. Not implemented -- see README.md.

Every stage below raises NotImplementedError deliberately: nothing here should be
allowed to silently "succeed" without actually doing the work, since a fake result
would be indistinguishable from a real reconstruction to anyone looking at the Job
row.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JobPayload:
    job_id: str
    video_url: str  # presigned URL the worker can download the input video from


def run_colmap_sfm(job: JobPayload) -> None:
    """Feature extraction -> matching -> sparse SfM (poses + sparse point cloud)."""
    raise NotImplementedError


def train_gaussian_splatting(job: JobPayload) -> None:
    """Per-scene optimization: differentiable rasterization + photometric loss
    against the extracted frames, using the SfM point cloud as initialization."""
    raise NotImplementedError


def evaluate_and_upload(job: JobPayload) -> dict:
    """Compute held-out-view PSNR/SSIM/LPIPS, upload the trained scene + metrics."""
    raise NotImplementedError


def handler(event: dict) -> dict:
    """RunPod serverless entrypoint: receives `event["input"]`, returns a result dict."""
    job = JobPayload(job_id=event["input"]["job_id"], video_url=event["input"]["video_url"])
    run_colmap_sfm(job)
    train_gaussian_splatting(job)
    return evaluate_and_upload(job)


if __name__ == "__main__":
    import runpod  # type: ignore[import-not-found]

    runpod.serverless.start({"handler": handler})
