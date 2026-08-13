"""GPU worker: real COLMAP SfM + Gaussian Splatting training pipeline.

Entry point is handler(), called once per job by RunPod's serverless runtime.
Frame images and the output destination arrive as presigned URLs in the job
payload -- the worker never touches storage credentials directly, only plain
HTTP GET/PUT against temporary links (see backend/app/storage.py's presigned URL
design).

train_gaussian_splatting is deliberately decoupled from COLMAP: it only depends
on ReconstructedScene (plain arrays), not on pycolmap types. That's what lets a
different reconstruction backend -- e.g. a feed-forward model like DUSt3R --
feed the same trainer, by producing a ReconstructedScene instead of running its
own separate training code. See colmap_reconstruction_to_scene for how COLMAP's
output gets converted into that shape.

STATUS: run_colmap_sfm and train_gaussian_splatting have both been run
successfully on real GPU hardware (Colab) against real phone video -- COLMAP
registering real cameras, Gaussian Splatting training producing a real,
inspectable result, adaptive density control genuinely growing Gaussian count.
Known real limitation at this point: still far short of the density real
room-scale splat scenes need for photorealism (thousands of Gaussians vs. the
hundreds of thousands typical results use).
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
import torch.nn.functional as F
from gsplat import rasterization
from PIL import Image
from scipy.spatial import cKDTree
from gsplat.strategy import DefaultStrategy


@dataclass
class JobPayload:
    job_id: str
    frame_urls: list[str]  # presigned GET URLs, one per extracted+filtered frame
    output_url: str  # presigned PUT URL for the trained scene artifact


@dataclass
class ReconstructedScene:
    """Backend-agnostic reconstruction output: plain arrays, no pycolmap types.

    train_gaussian_splatting only depends on this, not on COLMAP or any specific
    SfM implementation -- so DUSt3R (or anything else) can feed the same trainer
    just by producing one of these, instead of needing its own training code.
    """

    points_xyz: np.ndarray  # (N, 3) float32
    points_rgb: np.ndarray  # (N, 3) float32, in [0, 1]
    camera_viewmats: list[np.ndarray]  # each (4, 4) float32, world-to-camera
    camera_Ks: list[np.ndarray]  # each (3, 3) float32, intrinsics
    camera_images: list[np.ndarray]  # each (H, W, 3) float32, in [0, 1]


def colmap_reconstruction_to_scene(recon: pycolmap.Reconstruction, images_dir: Path) -> ReconstructedScene:
    """Adapter: pycolmap's specific object types -> the generic scene format
    train_gaussian_splatting actually consumes."""
    points = recon.points3D
    points_xyz = np.array([p.xyz for p in points.values()], dtype=np.float32)
    points_rgb = np.array([p.color for p in points.values()], dtype=np.float32) / 255.0

    camera_viewmats, camera_Ks, camera_images = [], [], []
    for image in recon.images.values():
        if not image.has_pose:
            continue
        img_path = images_dir / image.name
        pixels = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        camera_images.append(pixels)
        camera_viewmats.append(_colmap_pose_to_viewmat(image))
        camera_Ks.append(_colmap_camera_to_K(recon.camera(image.camera_id)))

    return ReconstructedScene(points_xyz, points_rgb, camera_viewmats, camera_Ks, camera_images)


def _download_frames(frame_urls: list[str], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(frame_urls):
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        (dest_dir / f"frame_{i:04d}.jpg").write_bytes(response.content)


def run_colmap_sfm(job: JobPayload, workdir: Path) -> ReconstructedScene:
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
    return colmap_reconstruction_to_scene(reconstructions[best_id], images_dir)


def _init_gaussians_from_points(xyz: np.ndarray, rgb: np.ndarray, device: torch.device):
    """Seed Gaussians from a sparse/dense 3D point cloud (from COLMAP or DUSt3R --
    this function doesn't care which): position/color come straight from the
    points, scale from each point's distance to its nearest neighbor, rotation
    starts as identity, opacity starts partly transparent -- standard 3DGS
    initialization, refined by training below."""
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=2)
    # Real SfM data (unlike the clean synthetic test scene) can have a handful of
    # isolated outlier points whose nearest neighbor is very far away -- clipping
    # only the minimum let those get a huge initial scale, which then dominates
    # the whole render as one giant blob covering everything else. Clip the
    # maximum too, relative to the scene's own distance distribution (95th
    # percentile) rather than a fixed number, since scene scale varies.
    max_reasonable_dist = np.percentile(dists[:, 1], 90)
    nearest_neighbor_dist = np.clip(dists[:, 1], 1e-4, max_reasonable_dist)

    n = xyz.shape[0]
    means = torch.tensor(xyz, device=device).requires_grad_(True)
    log_scales = np.log(nearest_neighbor_dist)[:, None].repeat(3, axis=1)
    scales = torch.tensor(log_scales, dtype=torch.float32, device=device).requires_grad_(True)
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    quats = quats.clone().requires_grad_(True)
    opacities = torch.full((n,), -2.0, device=device).requires_grad_(True)  # sigmoid(-2) ~= 0.12
    colors = torch.tensor(rgb, device=device).requires_grad_(True)

    # Same bound used at init, reused during training to stop gradient descent from
    # growing a few Gaussians huge again later (see train_gaussian_splatting).
    max_log_scale = float(np.log(max_reasonable_dist))

    return means, quats, scales, opacities, colors, max_log_scale


def _densify_and_prune(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    xyz_grad_accum: torch.Tensor,
    xyz_grad_count: torch.Tensor,
    max_log_scale: float,
    device: torch.device,
    grad_threshold: float = 0.0002,
    prune_opacity_threshold: float = 0.005,
):
    """Adaptive density control: grow detail where training is struggling, remove
    Gaussians that faded to near-invisible. This is what lets the scene end up
    with more than the fixed number of points SfM originally gave it.

    A Gaussian's *position* gradient tells us how much moving it would reduce
    loss -- large accumulated gradient over many steps means it's being pulled
    in inconsistent directions, a sign one Gaussian is trying (and failing) to
    cover a region that really needs several. Small-and-struggling gets CLONED
    (duplicated, so the copy can specialize); large-and-struggling gets SPLIT
    into two smaller ones sampled near the original. Opacity near zero means a
    Gaussian contributes almost nothing to any render -- PRUNED (removed).
    """
    avg_grad = xyz_grad_accum / xyz_grad_count.clamp(min=1)
    real_scales = torch.exp(scales)
    real_opacities = torch.sigmoid(opacities)

    high_grad = avg_grad > grad_threshold
    is_large = real_scales.max(dim=-1).values > 0.5 * np.exp(max_log_scale)
    split_mask = high_grad & is_large
    clone_mask = high_grad & ~is_large
    prune_mask = real_opacities < prune_opacity_threshold

    keep_mask = ~(split_mask | prune_mask)

    new_means = [means[keep_mask]]
    new_quats = [quats[keep_mask]]
    new_scales = [scales[keep_mask]]
    new_opacities = [opacities[keep_mask]]
    new_colors = [colors[keep_mask]]

    if clone_mask.any():
        new_means.append(means[clone_mask])
        new_quats.append(quats[clone_mask])
        new_scales.append(scales[clone_mask])
        new_opacities.append(opacities[clone_mask])
        new_colors.append(colors[clone_mask])

    if split_mask.any():
        n_split = int(split_mask.sum().item())
        # Two children per split Gaussian, offset randomly within roughly its own
        # extent, scaled down (a standard 3DGS convention) so total coverage stays
        # similar instead of ballooning.
        for _ in range(2):
            offset = torch.randn(n_split, 3, device=device) * real_scales[split_mask]
            new_means.append((means[split_mask] + offset).detach())
            new_quats.append(quats[split_mask].detach())
            new_scales.append((scales[split_mask] - np.log(1.6)).detach())
            new_opacities.append(opacities[split_mask].detach())
            new_colors.append(colors[split_mask].detach())

    means = torch.cat(new_means).detach().requires_grad_(True)
    quats = torch.cat(new_quats).detach().requires_grad_(True)
    scales = torch.cat(new_scales).detach().requires_grad_(True)
    opacities = torch.cat(new_opacities).detach().requires_grad_(True)
    colors = torch.cat(new_colors).detach().requires_grad_(True)

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


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d /= g1d.sum()
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)  # outer product -> 2D Gaussian blur kernel
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim(render: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity between two (H, W, C) images in [0, 1] -- classical
    image-processing math (local-window mean/variance/covariance via Gaussian-
    blurred convolutions), not a neural network. Used only as a differentiable
    loss term: torch.nn.functional.conv2d already tracks gradients, so this
    plugs into the same optimizer.step() as the L1 term below with no extra
    wiring. See train_gaussian_splatting's loss line for why this is here --
    pure pixel-difference (L1) loss famously biases toward blurry, averaged-out
    results; SSIM rewards matching local structure/contrast instead, which is
    why the reference 3D Gaussian Splatting paper trains with both, not L1 alone.
    """
    C1, C2 = 0.01**2, 0.03**2
    x = render.permute(2, 0, 1).unsqueeze(0)  # (H,W,C) -> (1,C,H,W)
    y = target.permute(2, 0, 1).unsqueeze(0)
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma=1.5, channels=channels, device=x.device)
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)
    mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean()


def _make_optimizers(params: torch.nn.ParameterDict) -> dict[str, torch.optim.Optimizer]:
    """One optimizer per parameter, as required by gsplat's strategy API.

    DefaultStrategy edits both the Gaussian tensors and their corresponding
    Adam state during duplicate/split/prune operations. A single multi-group
    optimizer does not satisfy that API, and recreating Adam after every
    refinement (the old implementation) discards its accumulated momentum and
    variance estimates.
    """
    learning_rates = {
        "means": 1.6e-4,
        "quats": 1e-3,
        "scales": 5e-3,
        "opacities": 5e-2,
        "colors": 2.5e-3,
    }
    return {
        name: torch.optim.Adam([params[name]], lr=learning_rates[name])
        for name in params.keys()
    }


def train_gaussian_splatting(
    scene: ReconstructedScene,
    num_iterations: int = 3000,
    densify_from: int = 500,
    densify_until: int | None = None,
    densify_interval: int = 100,
) -> dict:
    """Per-scene optimization: differentiable rasterization + photometric loss,
    refining the SfM-initialized Gaussians against the real extracted frames,
    with adaptive density control (split/clone/prune) growing detail beyond
    SfM's original fixed point count -- see _densify_and_prune.

    Backend-agnostic: only depends on ReconstructedScene's plain arrays, not on
    COLMAP specifically -- DUSt3R (or anything else) can feed this the same way.

    UNVERIFIED -- see module docstring. Written against gsplat's documented API,
    never actually executed (no local CUDA GPU to run it on).
    """
    device = torch.device("cuda")
    densify_until = num_iterations if densify_until is None else densify_until

    means, quats, scales, opacities, colors, _max_log_scale = _init_gaussians_from_points(
        scene.points_xyz, scene.points_rgb, device
    )
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "quats": torch.nn.Parameter(quats),
            "scales": torch.nn.Parameter(scales),
            "opacities": torch.nn.Parameter(opacities),
            "colors": torch.nn.Parameter(colors),
        }
    )
    optimizers = _make_optimizers(params)

    # DefaultStrategy is gsplat's maintained implementation of the original
    # 3DGS adaptive density-control algorithm. Crucially, it accumulates
    # gradients of projected 2D Gaussian positions (where the image lacks
    # detail), rather than our old world-space means.grad heuristic. It also
    # preserves each Adam optimizer's state while changing tensor sizes.
    strategy = DefaultStrategy(
        refine_start_iter=densify_from,
        refine_stop_iter=densify_until,
        refine_every=densify_interval,
        verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    scene_extent = np.ptp(scene.points_xyz, axis=0)
    scene_scale = max(float(np.linalg.norm(scene_extent)), 1e-6)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    cameras = []
    for pixels_np, viewmat_np, K_np in zip(scene.camera_images, scene.camera_viewmats, scene.camera_Ks):
        pixels = torch.tensor(pixels_np, device=device)
        viewmat = torch.tensor(viewmat_np, device=device)
        K = torch.tensor(K_np, device=device)
        cameras.append({"pixels": pixels, "viewmat": viewmat, "K": K, "h": pixels.shape[0], "w": pixels.shape[1]})

    if not cameras:
        raise RuntimeError("No registered cameras with poses to train against")

    for step in range(num_iterations):
        cam = cameras[step % len(cameras)]
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        # scales/opacities are stored unconstrained (log-scale, logit) so gradient
        # descent can move them freely -- exp()/sigmoid() here converts them to the
        # real positive-scale / 0-1-opacity values the renderer actually needs.
        # Forgetting this step renders as solid black: raw negative "opacity" and
        # near-zero/negative "scale" are both effectively invisible.
        render, _alpha, _meta = rasterization(
            params["means"],
            params["quats"],
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            params["colors"],
            cam["viewmat"].unsqueeze(0),
            cam["K"].unsqueeze(0),
            cam["w"],
            cam["h"],
            sh_degree=None,  # plain RGB per Gaussian for the first version, no
            # view-dependent color yet -- spherical harmonics is a real quality
            # improvement to add once the basic loop is confirmed working
        )
        # Pure L1 (pixel-difference) loss biases the optimizer toward blurry,
        # averaged-out results in ambiguous regions -- confirmed on our own
        # A/B render check (real photo vs. server-side render, same pose:
        # hazy/soft even after fixing the training-resolution bottleneck).
        # SSIM_LAMBDA=0.2 matches the reference 3D Gaussian Splatting paper's
        # L1 + D-SSIM loss, which exists specifically to counter this.
        SSIM_LAMBDA = 0.2
        l1_loss = torch.abs(render[0] - cam["pixels"]).mean()
        d_ssim_loss = 1.0 - _ssim(render[0], cam["pixels"])
        loss = (1.0 - SSIM_LAMBDA) * l1_loss + SSIM_LAMBDA * d_ssim_loss
        strategy.step_pre_backward(params, optimizers, strategy_state, step, _meta)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        strategy.step_post_backward(params, optimizers, strategy_state, step, _meta, packed=True)

    return {
        "means": params["means"].detach().cpu().numpy(),
        "quats": params["quats"].detach().cpu().numpy(),
        "scales": torch.exp(params["scales"]).detach().cpu().numpy(),
        "opacities": torch.sigmoid(params["opacities"]).detach().cpu().numpy(),
        "colors": params["colors"].detach().cpu().numpy(),
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
        scene = run_colmap_sfm(job, Path(tmp))
        gaussians = train_gaussian_splatting(scene)
        return evaluate_and_upload(job, gaussians)


if __name__ == "__main__":
    import runpod  # type: ignore[import-not-found]

    runpod.serverless.start({"handler": handler})
