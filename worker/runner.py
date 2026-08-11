"""GPU worker: real COLMAP SfM + Gaussian Splatting training pipeline.

Entry point is handler(), called once per job by RunPod's serverless runtime.
Frame images and the output destination arrive as presigned URLs in the job
payload -- the worker never touches storage credentials directly, only plain
HTTP GET/PUT against temporary links (see backend/app/storage.py's presigned URL
design).

STATUS: run_colmap_sfm is proven correct -- it's the same pycolmap calls
validated in experiments/sfm/run_sfm.py (16/16 cameras registered, ~1.2% pose
error vs. ground truth on a synthetic scene). train_gaussian_splatting is
UNVERIFIED: this machine has no CUDA GPU and gsplat requires one, so it has never
actually been run. It's written against gsplat's real documented API and checked
against pycolmap's real object shapes where possible, but treat it as a first
draft to validate once real GPU access exists (RunPod), not proven-correct code.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import requests
import torch
from gsplat import rasterization
from PIL import Image
from scipy.spatial import cKDTree


@dataclass
class JobPayload:
    job_id: str
    frame_urls: list[str]  # presigned GET URLs, one per extracted+filtered frame
    output_url: str  # presigned PUT URL for the trained scene artifact


@dataclass
class SfmResult:
    reconstruction: pycolmap.Reconstruction
    images_dir: Path


def _download_frames(frame_urls: list[str], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(frame_urls):
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        (dest_dir / f"frame_{i:04d}.jpg").write_bytes(response.content)


def run_colmap_sfm(job: JobPayload, workdir: Path) -> SfmResult:
    """Feature extraction -> matching -> sparse SfM (poses + sparse point cloud)."""
    images_dir = workdir / "images"
    _download_frames(job.frame_urls, images_dir)

    database_path = workdir / "database.db"
    sparse_dir = workdir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    pycolmap.extract_features(str(database_path), str(images_dir))
    pycolmap.match_exhaustive(str(database_path))
    reconstructions = pycolmap.incremental_mapping(str(database_path), str(images_dir), str(sparse_dir))

    if not reconstructions:
        raise RuntimeError("COLMAP failed to register any cameras for this video")

    best_id = max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
    return SfmResult(reconstruction=reconstructions[best_id], images_dir=images_dir)


def _init_gaussians_from_sfm(recon: pycolmap.Reconstruction, device: torch.device):
    """Seed Gaussians from COLMAP's sparse point cloud: position/color come
    straight from SfM's triangulated points, scale from each point's distance to
    its nearest neighbor, rotation starts as identity, opacity starts partly
    transparent -- standard 3DGS initialization, refined by training below."""
    points = recon.points3D
    xyz = np.array([p.xyz for p in points.values()], dtype=np.float32)
    rgb = np.array([p.color for p in points.values()], dtype=np.float32) / 255.0

    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=2)
    nearest_neighbor_dist = np.clip(dists[:, 1], 1e-4, None)

    n = xyz.shape[0]
    means = torch.tensor(xyz, device=device).requires_grad_(True)
    log_scales = np.log(nearest_neighbor_dist)[:, None].repeat(3, axis=1)
    scales = torch.tensor(log_scales, dtype=torch.float32, device=device).requires_grad_(True)
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    quats = quats.clone().requires_grad_(True)
    opacities = torch.full((n,), -2.0, device=device).requires_grad_(True)  # sigmoid(-2) ~= 0.12
    colors = torch.tensor(rgb, device=device).requires_grad_(True)

    return means, quats, scales, opacities, colors


def _colmap_camera_to_K(camera: pycolmap.Camera) -> np.ndarray:
    # Ignoring lens distortion params here (e.g. SIMPLE_RADIAL's k) for the first
    # version -- a real simplification, not an oversight; revisit if training
    # quality on real (non-synthetic) footage suffers from it.
    fx, fy = camera.focal_length_x, camera.focal_length_y
    cx, cy = camera.principal_point_x, camera.principal_point_y
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _colmap_pose_to_viewmat(image: pycolmap.Image) -> np.ndarray:
    # cam_from_world().matrix() returns the 3x4 [R|t] world-to-camera transform;
    # gsplat wants a full 4x4 matrix, so pad with the [0,0,0,1] homogeneous row.
    RT = image.cam_from_world().matrix()
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :4] = RT
    return viewmat


def train_gaussian_splatting(sfm: SfmResult, num_iterations: int = 3000) -> dict:
    """Per-scene optimization: differentiable rasterization + photometric loss,
    refining the SfM-initialized Gaussians against the real extracted frames.

    UNVERIFIED -- see module docstring. Written against gsplat's documented API,
    never actually executed (no local CUDA GPU to run it on).
    """
    device = torch.device("cuda")
    recon = sfm.reconstruction

    means, quats, scales, opacities, colors = _init_gaussians_from_sfm(recon, device)
    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": 1.6e-4},
            {"params": [quats], "lr": 1e-3},
            {"params": [scales], "lr": 5e-3},
            {"params": [opacities], "lr": 5e-2},
            {"params": [colors], "lr": 2.5e-3},
        ]
    )

    cameras = []
    for image in recon.images.values():
        if not image.has_pose:
            continue
        img_path = sfm.images_dir / image.name
        pixels_np = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        pixels = torch.tensor(pixels_np, device=device)
        viewmat = torch.tensor(_colmap_pose_to_viewmat(image), device=device)
        K = torch.tensor(_colmap_camera_to_K(recon.camera(image.camera_id)), device=device)
        cameras.append({"pixels": pixels, "viewmat": viewmat, "K": K, "h": pixels.shape[0], "w": pixels.shape[1]})

    if not cameras:
        raise RuntimeError("No registered cameras with poses to train against")

    for step in range(num_iterations):
        cam = cameras[step % len(cameras)]
        render, _alpha, _meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            cam["viewmat"].unsqueeze(0),
            cam["K"].unsqueeze(0),
            cam["w"],
            cam["h"],
            sh_degree=None,  # plain RGB per Gaussian for the first version, no
            # view-dependent color yet -- spherical harmonics is a real quality
            # improvement to add once the basic loop is confirmed working
        )
        loss = torch.abs(render[0] - cam["pixels"]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "means": means.detach().cpu().numpy(),
        "quats": quats.detach().cpu().numpy(),
        "scales": scales.detach().cpu().numpy(),
        "opacities": opacities.detach().cpu().numpy(),
        "colors": colors.detach().cpu().numpy(),
    }


def evaluate_and_upload(job: JobPayload, gaussians: dict) -> dict:
    """Save the trained scene, upload it via the presigned PUT URL, return metrics."""
    buffer = io.BytesIO()
    np.savez(buffer, **gaussians)
    buffer.seek(0)
    response = requests.put(job.output_url, data=buffer.getvalue(), timeout=120)
    response.raise_for_status()

    return {"num_gaussians": int(gaussians["means"].shape[0])}


def handler(event: dict) -> dict:
    """RunPod serverless entrypoint: receives `event["input"]`, returns a result dict."""
    payload = event["input"]
    job = JobPayload(
        job_id=payload["job_id"],
        frame_urls=payload["frame_urls"],
        output_url=payload["output_url"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        sfm = run_colmap_sfm(job, Path(tmp))
        gaussians = train_gaussian_splatting(sfm)
        return evaluate_and_upload(job, gaussians)


if __name__ == "__main__":
    import runpod  # type: ignore[import-not-found]

    runpod.serverless.start({"handler": handler})
