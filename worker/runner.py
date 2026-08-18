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
try:
    import pycolmap
except ImportError:  # DUSt3R-only workers do not need COLMAP installed.
    pycolmap = None
import requests
import torch
import torch.nn.functional as F
from gsplat import rasterization
from PIL import Image
from scipy.spatial import cKDTree
from gsplat.strategy import DefaultStrategy
try:
    from training_schedule import (
        camera_cycle_densification_schedule as _camera_cycle_densification_schedule,
        camera_order_for_cycle as _camera_order_for_cycle,
    )
except ImportError:  # Package-style imports used by local tests/tooling.
    from worker.training_schedule import (
        camera_cycle_densification_schedule as _camera_cycle_densification_schedule,
        camera_order_for_cycle as _camera_order_for_cycle,
    )

SH_C0 = 0.28209479177387814


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


@dataclass
class GaussianTrainingState:
    """Live, resumable optimization state for chunked in-session training.

    This intentionally stays on the GPU. It preserves Adam momentum/variance,
    DefaultStrategy's accumulated screen-space statistics, and the absolute
    training step so a later call can continue rather than restart.
    """

    params: torch.nn.ParameterDict
    optimizers: dict[str, torch.optim.Optimizer]
    strategy: DefaultStrategy
    strategy_state: dict
    step: int
    learning_rate_decay_applied: bool = False
    exposure_log_gains: torch.nn.Parameter | None = None
    exposure_biases: torch.nn.Parameter | None = None
    exposure_optimizer: torch.optim.Optimizer | None = None
    sh_degree: int | None = None
    camera_pose_deltas: torch.nn.Parameter | None = None
    camera_pose_optimizer: torch.optim.Optimizer | None = None
    camera_sampling: str = "sequential"
    camera_sampling_seed: int = 0


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
    if pycolmap is None:
        raise RuntimeError("COLMAP reconstruction requires the optional pycolmap package")
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


def _init_gaussians_from_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    device: torch.device,
    sh_degree: int | None = None,
):
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
    colors = torch.tensor(rgb, device=device)
    if sh_degree is not None:
        coefficient_count = (sh_degree + 1) ** 2
        coefficients = torch.zeros(
            (n, coefficient_count, 3), dtype=colors.dtype, device=device
        )
        # Preserve the input RGB exactly at initialization. Higher coefficients
        # begin at zero and learn view-dependent appearance during training.
        coefficients[:, 0, :] = (colors - 0.5) / SH_C0
        colors = coefficients
    colors = colors.requires_grad_(True)

    # Same bound used at init, reused during training to stop gradient descent from
    # growing a few Gaussians huge again later (see train_gaussian_splatting).
    max_log_scale = float(np.log(max_reasonable_dist))

    return means, quats, scales, opacities, colors, max_log_scale


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


def _apply_late_learning_rate_decay(
    optimizers: dict[str, torch.optim.Optimizer], factor: float = 0.1
) -> None:
    """Reduce late-stage step sizes after coarse scene formation converges.

    A real Colab A/B test at step 9,000 showed that reducing all parameter
    learning rates to 10% raised five-view mean PSNR from 26.00 to 29.22 dB in
    1,000 steps. Apply the proven change once; resumable state tracks whether
    it has already happened so continuation cannot compound it accidentally.
    """
    for optimizer in optimizers.values():
        for param_group in optimizer.param_groups:
            param_group["lr"] *= factor


def _skew_symmetric(vectors: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3) vectors into cross-product matrices."""
    x, y, z = vectors.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
    ).reshape(vectors.shape[:-1] + (3, 3))


def _pose_deltas_to_matrices(deltas: torch.Tensor) -> torch.Tensor:
    """Differentiable SE(3) corrections from rotation-vector + translation.

    Rodrigues' formula maps each three-value axis-angle rotation into a valid
    orthonormal matrix. Taylor-safe coefficients keep gradients finite at the
    identity pose where every refinement begins.
    """
    rotation_vectors = deltas[..., :3]
    translations = deltas[..., 3:]
    theta_sq = (rotation_vectors * rotation_vectors).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq.clamp_min(1e-12))
    A = torch.where(
        theta_sq < 1e-8,
        1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
        torch.sin(theta) / theta,
    )
    B = torch.where(
        theta_sq < 1e-8,
        0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
        (1.0 - torch.cos(theta)) / theta_sq.clamp_min(1e-12),
    )
    K = _skew_symmetric(rotation_vectors)
    identity3 = torch.eye(3, dtype=deltas.dtype, device=deltas.device).expand(
        deltas.shape[:-1] + (3, 3)
    )
    rotations = identity3 + A[..., None] * K + B[..., None] * (K @ K)
    matrices = torch.eye(4, dtype=deltas.dtype, device=deltas.device).repeat(
        *deltas.shape[:-1], 1, 1
    )
    matrices[..., :3, :3] = rotations
    matrices[..., :3, 3] = translations
    return matrices


def _params_from_exported_gaussians(
    gaussians: dict, device: torch.device, sh_degree: int | None = None
) -> torch.nn.ParameterDict:
    """Restore trainable raw parameters from the activated export representation.

    Exports contain positive scales and 0..1 opacities. Training stores their
    unconstrained log/logit forms, so warm-starting requires the inverse
    transforms below. This reuses the optimized scene, but cannot reconstruct
    Adam or density-strategy history that was not returned by an older run.
    """
    scales = np.log(np.clip(gaussians["scales"], 1e-8, None))
    opacities = np.clip(gaussians["opacities"], 1e-6, 1 - 1e-6)
    opacity_logits = np.log(opacities / (1 - opacities))
    colors = torch.tensor(gaussians["colors"], device=device)
    if sh_degree is not None:
        if "sh_coeffs" in gaussians:
            colors = torch.tensor(gaussians["sh_coeffs"], device=device)
            expected = (sh_degree + 1) ** 2
            if colors.shape[1] != expected:
                raise ValueError(
                    f"Export contains {colors.shape[1]} SH coefficients, expected {expected}"
                )
        else:
            coefficient_count = (sh_degree + 1) ** 2
            sh_coefficients = torch.zeros(
                (len(colors), coefficient_count, 3), dtype=colors.dtype, device=device
            )
            # gsplat evaluates SH color as 0.5 + C0 * dc + directional terms.
            sh_coefficients[:, 0, :] = (colors - 0.5) / SH_C0
            colors = sh_coefficients
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.tensor(gaussians["means"], device=device)),
            "quats": torch.nn.Parameter(torch.tensor(gaussians["quats"], device=device)),
            "scales": torch.nn.Parameter(
                torch.tensor(scales, dtype=torch.float32, device=device)
            ),
            "opacities": torch.nn.Parameter(
                torch.tensor(opacity_logits, dtype=torch.float32, device=device)
            ),
            "colors": torch.nn.Parameter(colors),
        }
    )


def _export_gaussian_params(
    params: torch.nn.ParameterDict, sh_degree: int | None = None
) -> dict:
    result = {
        "means": params["means"].detach().cpu().numpy(),
        "quats": params["quats"].detach().cpu().numpy(),
        "scales": torch.exp(params["scales"]).detach().cpu().numpy(),
        "opacities": torch.sigmoid(params["opacities"]).detach().cpu().numpy(),
    }
    if sh_degree is None:
        result["colors"] = params["colors"].detach().cpu().numpy()
    else:
        coefficients = params["colors"].detach()
        result["colors"] = (0.5 + SH_C0 * coefficients[:, 0, :]).cpu().numpy()
        result["sh_coeffs"] = coefficients.cpu().numpy()
        result["sh_degree"] = sh_degree
    return result


def train_gaussian_splatting(
    scene: ReconstructedScene,
    num_iterations: int = 3000,
    densify_from: int = 500,
    densify_until: int | None = None,
    densify_interval: int = 100,
    *,
    initial_gaussians: dict | None = None,
    step_offset: int = 0,
    training_state: GaussianTrainingState | None = None,
    return_training_state: bool = False,
    learning_rate_decay_step: int | None = 9000,
    learning_rate_decay_factor: float = 0.1,
    optimize_camera_exposure: bool = True,
    exposure_learning_rate: float = 1e-3,
    exposure_regularization: float = 1e-3,
    sh_degree: int | None = None,
    optimize_camera_poses: bool = False,
    camera_pose_refinement_start_step: int = 15000,
    camera_pose_learning_rate: float = 1e-5,
    camera_rotation_regularization: float = 1e-3,
    camera_translation_regularization: float = 1e-3,
    camera_sampling: str = "sequential",
    camera_sampling_seed: int = 0,
    densify_interval_camera_cycles: int | None = None,
) -> dict | tuple[dict, GaussianTrainingState]:
    """Per-scene optimization: differentiable rasterization + photometric loss,
    refining the SfM-initialized Gaussians against the real extracted frames,
    with adaptive density control (split/clone/prune) growing detail beyond
    SfM's original fixed point count -- via gsplat's own DefaultStrategy
    (nerfstudio-project/gsplat), not a hand-rolled implementation: it follows
    the same split/clone/prune algorithm as the original 3DGS paper, and using
    the library's own tested version instead of reimplementing it is what
    made per-parameter Adam optimizers (see _make_optimizers) necessary in
    the first place -- DefaultStrategy's split/prune operations need to edit
    each parameter's Adam state directly, which a single multi-group
    optimizer doesn't expose.

    Backend-agnostic: only depends on ReconstructedScene's plain arrays, not on
    COLMAP specifically -- DUSt3R (or anything else) can feed this the same way.

    The core gsplat 1.5.3 screen-space strategy path has passed a real Colab T4
    smoke run. The resumable-state and exported-scene warm-start paths remain
    UNVALIDATED UNTIL A REAL COLAB GPU RUN.
    """
    device = torch.device("cuda")
    camera_count = len(scene.camera_images)
    if camera_count == 0:
        raise RuntimeError("No registered cameras with poses to train against")
    if camera_sampling not in {"sequential", "shuffled_cycle"}:
        raise ValueError(
            "camera_sampling must be 'sequential' or 'shuffled_cycle'"
        )
    if densify_interval_camera_cycles is not None:
        _, densify_interval = _camera_cycle_densification_schedule(
            camera_count, densify_from, step_offset, densify_interval_camera_cycles
        )
    if training_state is not None and initial_gaussians is not None:
        raise ValueError("Pass training_state or initial_gaussians, not both")

    if training_state is not None:
        params = training_state.params
        optimizers = training_state.optimizers
        strategy = training_state.strategy
        strategy_state = training_state.strategy_state
        start_step = training_state.step
        learning_rate_decay_applied = training_state.learning_rate_decay_applied
        exposure_log_gains = training_state.exposure_log_gains
        exposure_biases = training_state.exposure_biases
        exposure_optimizer = training_state.exposure_optimizer
        if sh_degree != training_state.sh_degree:
            raise ValueError(
                f"Requested sh_degree={sh_degree}, but saved state uses "
                f"sh_degree={training_state.sh_degree}"
            )
        camera_pose_deltas = training_state.camera_pose_deltas
        camera_pose_optimizer = training_state.camera_pose_optimizer
        if (
            camera_sampling != training_state.camera_sampling
            or camera_sampling_seed != training_state.camera_sampling_seed
        ):
            raise ValueError(
                "Requested camera sampling configuration does not match saved state"
            )
    else:
        start_step = step_offset
        if initial_gaussians is None:
            means, quats, scales, opacities, colors, _max_log_scale = _init_gaussians_from_points(
                scene.points_xyz, scene.points_rgb, device, sh_degree
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
        else:
            params = _params_from_exported_gaussians(initial_gaussians, device, sh_degree)

        optimizers = _make_optimizers(params)
        stop_step = start_step + num_iterations
        densify_until = stop_step if densify_until is None else densify_until
        # A warm start has no saved screen-gradient history. Give it 500 steps
        # to accumulate representative statistics before its first refinement.
        effective_densify_from = max(densify_from, start_step + 500)
        if densify_interval_camera_cycles is not None:
            # DefaultStrategy refines on global ``refine_every`` boundaries.
            # Aligning its start and making the interval a whole number of
            # camera cycles ensures every refinement window contains every
            # camera equally often. The default step-based behavior is kept
            # exactly when this opt-in setting is absent.
            effective_densify_from, densify_interval = (
                _camera_cycle_densification_schedule(
                    camera_count,
                    densify_from,
                    start_step,
                    densify_interval_camera_cycles,
                )
            )
        strategy = DefaultStrategy(
            refine_start_iter=effective_densify_from,
            refine_stop_iter=densify_until,
            refine_every=densify_interval,
            verbose=True,
        )
        strategy.check_sanity(params, optimizers)
        scene_extent = np.ptp(scene.points_xyz, axis=0)
        scene_scale = max(float(np.linalg.norm(scene_extent)), 1e-6)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
        learning_rate_decay_applied = False
        exposure_log_gains = None
        exposure_biases = None
        exposure_optimizer = None
        camera_pose_deltas = None
        camera_pose_optimizer = None

    stop_step = start_step + num_iterations
    scene_scale = float(strategy_state["scene_scale"])

    cameras = []
    for pixels_np, viewmat_np, K_np in zip(scene.camera_images, scene.camera_viewmats, scene.camera_Ks):
        pixels = torch.tensor(pixels_np, device=device)
        viewmat = torch.tensor(viewmat_np, device=device)
        K = torch.tensor(K_np, device=device)
        cameras.append({"pixels": pixels, "viewmat": viewmat, "K": K, "h": pixels.shape[0], "w": pixels.shape[1]})

    if not cameras:
        raise RuntimeError("No registered cameras with poses to train against")

    if (
        training_state is not None
        and densify_interval_camera_cycles is not None
        and strategy.refine_every != densify_interval
    ):
        raise ValueError(
            "Saved densification interval does not match requested camera-cycle "
            f"schedule ({strategy.refine_every} != {densify_interval})"
        )

    if optimize_camera_exposure and exposure_optimizer is None:
        # log-gain keeps the multiplicative correction positive. Gain=1 and
        # bias=0 are identity, so a new run begins with exactly the old render.
        exposure_log_gains = torch.nn.Parameter(
            torch.zeros((len(cameras), 3), device=device)
        )
        exposure_biases = torch.nn.Parameter(
            torch.zeros((len(cameras), 3), device=device)
        )
        exposure_optimizer = torch.optim.Adam(
            [exposure_log_gains, exposure_biases], lr=exposure_learning_rate
        )
    elif optimize_camera_exposure and (
        exposure_log_gains is None
        or exposure_biases is None
        or len(exposure_log_gains) != len(cameras)
    ):
        raise ValueError("Saved exposure state does not match the scene camera count")

    if optimize_camera_poses and camera_pose_optimizer is None:
        camera_pose_deltas = torch.nn.Parameter(
            torch.zeros((len(cameras), 6), device=device)
        )
        camera_pose_optimizer = torch.optim.Adam(
            [camera_pose_deltas], lr=camera_pose_learning_rate
        )
    elif optimize_camera_poses and (
        camera_pose_deltas is None or len(camera_pose_deltas) != len(cameras)
    ):
        raise ValueError("Saved camera-pose state does not match the scene camera count")

    sampling_cycle = None
    sampling_order = None
    for step in range(start_step, stop_step):
        if (
            learning_rate_decay_step is not None
            and not learning_rate_decay_applied
            and step >= learning_rate_decay_step
        ):
            _apply_late_learning_rate_decay(optimizers, learning_rate_decay_factor)
            learning_rate_decay_applied = True
            print(
                f"Step {step}: reduced all learning rates by "
                f"{learning_rate_decay_factor:g} for late-stage refinement."
            )
        if camera_sampling == "sequential":
            camera_index = step % len(cameras)
        else:
            cycle = step // len(cameras)
            if cycle != sampling_cycle:
                # Deriving each permutation solely from its global cycle makes
                # resumed runs select exactly the same cameras as uninterrupted
                # runs without adding sampler state to checkpoints.
                sampling_order = _camera_order_for_cycle(
                    len(cameras), cycle, camera_sampling, camera_sampling_seed
                )
                sampling_cycle = cycle
            camera_index = int(sampling_order[step % len(cameras)])
        cam = cameras[camera_index]
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if exposure_optimizer is not None:
            exposure_optimizer.zero_grad(set_to_none=True)
        pose_refinement_active = (
            camera_pose_optimizer is not None
            and step >= camera_pose_refinement_start_step
        )
        if pose_refinement_active:
            camera_pose_optimizer.zero_grad(set_to_none=True)
        viewmat = cam["viewmat"]
        pose_penalty = torch.zeros((), device=device)
        if pose_refinement_active:
            # Fix camera 0 as the world-frame gauge. Multiplying its delta by
            # zero also prevents it from receiving an optimizer update.
            anchor_mask = 0.0 if camera_index == 0 else 1.0
            pose_delta = camera_pose_deltas[camera_index] * anchor_mask
            correction = _pose_deltas_to_matrices(pose_delta.unsqueeze(0))[0]
            viewmat = correction @ viewmat
            rotation_delta = pose_delta[:3]
            translation_delta = pose_delta[3:] / scene_scale
            pose_penalty = (
                camera_rotation_regularization * rotation_delta.square().mean()
                + camera_translation_regularization * translation_delta.square().mean()
            )
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
            viewmat.unsqueeze(0),
            cam["K"].unsqueeze(0),
            cam["w"],
            cam["h"],
            sh_degree=sh_degree,
        )
        # Pure L1 (pixel-difference) loss biases the optimizer toward blurry,
        # averaged-out results in ambiguous regions -- confirmed on our own
        # A/B render check (real photo vs. server-side render, same pose:
        # hazy/soft even after fixing the training-resolution bottleneck).
        # SSIM_LAMBDA=0.2 matches the reference 3D Gaussian Splatting paper's
        # L1 + D-SSIM loss, which exists specifically to counter this.
        SSIM_LAMBDA = 0.2
        loss_render = render[0]
        exposure_penalty = torch.zeros((), device=device)
        if exposure_optimizer is not None:
            log_gain = exposure_log_gains[camera_index]
            bias = exposure_biases[camera_index]
            loss_render = loss_render * torch.exp(log_gain) + bias
            # Keep the learned camera transform close to identity. This lets it
            # absorb measured exposure/white-balance shifts without becoming a
            # substitute for correct geometry or scene appearance.
            exposure_penalty = exposure_regularization * (
                log_gain.square().mean() + bias.square().mean()
            )
        l1_loss = torch.abs(loss_render - cam["pixels"]).mean()
        d_ssim_loss = 1.0 - _ssim(loss_render, cam["pixels"])
        # Needle-like ellipsoids cheaply cover uncertain pixels in observed
        # views but become catastrophic streaks as soon as the user moves.
        # Penalize only log-axis spreads beyond a generous 20:1 ratio.
        log_scale_span = params["scales"].amax(dim=1) - params["scales"].amin(dim=1)
        anisotropy_penalty = torch.relu(log_scale_span - np.log(20.0)).square().mean()
        loss = (1.0 - SSIM_LAMBDA) * l1_loss + SSIM_LAMBDA * d_ssim_loss
        loss = loss + 1e-4 * anisotropy_penalty + exposure_penalty + pose_penalty
        strategy.step_pre_backward(params, optimizers, strategy_state, step, _meta)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        if exposure_optimizer is not None:
            exposure_optimizer.step()
        if pose_refinement_active:
            camera_pose_optimizer.step()
        strategy.step_post_backward(params, optimizers, strategy_state, step, _meta, packed=True)

    gaussians = _export_gaussian_params(params, sh_degree)
    if not return_training_state:
        return gaussians
    state = GaussianTrainingState(
        params=params,
        optimizers=optimizers,
        strategy=strategy,
        strategy_state=strategy_state,
        step=stop_step,
        learning_rate_decay_applied=learning_rate_decay_applied,
        exposure_log_gains=exposure_log_gains,
        exposure_biases=exposure_biases,
        exposure_optimizer=exposure_optimizer,
        sh_degree=sh_degree,
        camera_pose_deltas=camera_pose_deltas,
        camera_pose_optimizer=camera_pose_optimizer,
        camera_sampling=camera_sampling,
        camera_sampling_seed=camera_sampling_seed,
    )
    return gaussians, state


def save_training_state(state: GaussianTrainingState, path: Path) -> None:
    """Serialize resumable Gaussian training state to disk.

    UNVALIDATED UNTIL A REAL COLAB/RUNPOD RUN exercises a save+load+continue
    cycle end to end -- the in-memory resume path (training_state passed
    straight back into train_gaussian_splatting within one Python process)
    has real Colab T4 usage; round-tripping through disk has not been tested.

    Without this, a Colab disconnect mid-training loses everything: Adam's
    per-parameter momentum/variance and DefaultStrategy's accumulated
    screen-space gradient statistics are real optimization state, not just
    the current Gaussian values -- reloading only the exported
    means/quats/scales/opacities/colors and calling train_gaussian_splatting
    again would silently restart optimization dynamics from zero even though
    the visible Gaussians look unchanged. This is exactly the "persist each
    stage so a failure never requires starting over" requirement in
    CLAUDE_HANDOFF.md.

    `strategy` itself isn't saved -- gsplat's DefaultStrategy (verified
    against its real source) holds no learned state, only the
    refine_start_iter/refine_stop_iter/refine_every config it was
    constructed with. That config is saved and used to reconstruct an
    equivalent DefaultStrategy on load instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "params": {name: tensor.detach().cpu() for name, tensor in state.params.items()},
            "optimizers": {name: opt.state_dict() for name, opt in state.optimizers.items()},
            "strategy_config": {
                "refine_start_iter": state.strategy.refine_start_iter,
                "refine_stop_iter": state.strategy.refine_stop_iter,
                "refine_every": state.strategy.refine_every,
            },
            "strategy_state": {
                key: (value.detach().cpu() if torch.is_tensor(value) else value)
                for key, value in state.strategy_state.items()
            },
            "step": state.step,
            "learning_rate_decay_applied": state.learning_rate_decay_applied,
            "exposure_log_gains": (
                state.exposure_log_gains.detach().cpu() if state.exposure_log_gains is not None else None
            ),
            "exposure_biases": (
                state.exposure_biases.detach().cpu() if state.exposure_biases is not None else None
            ),
            "exposure_optimizer": (
                state.exposure_optimizer.state_dict() if state.exposure_optimizer is not None else None
            ),
            "sh_degree": state.sh_degree,
            "camera_pose_deltas": (
                state.camera_pose_deltas.detach().cpu() if state.camera_pose_deltas is not None else None
            ),
            "camera_pose_optimizer": (
                state.camera_pose_optimizer.state_dict() if state.camera_pose_optimizer is not None else None
            ),
            "camera_sampling": state.camera_sampling,
            "camera_sampling_seed": state.camera_sampling_seed,
        },
        path,
    )


def load_training_state(path: Path, device: torch.device) -> GaussianTrainingState:
    """Inverse of save_training_state -- see its docstring for what's saved and why."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    params = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(tensor.to(device)) for name, tensor in checkpoint["params"].items()}
    )
    optimizers = _make_optimizers(params)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint["optimizers"][name])

    strategy = DefaultStrategy(
        refine_start_iter=checkpoint["strategy_config"]["refine_start_iter"],
        refine_stop_iter=checkpoint["strategy_config"]["refine_stop_iter"],
        refine_every=checkpoint["strategy_config"]["refine_every"],
        verbose=True,
    )
    strategy_state = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in checkpoint["strategy_state"].items()
    }

    def _load_optional_param(key: str) -> torch.nn.Parameter | None:
        value = checkpoint[key]
        return torch.nn.Parameter(value.to(device)) if value is not None else None

    exposure_log_gains = _load_optional_param("exposure_log_gains")
    exposure_biases = _load_optional_param("exposure_biases")
    exposure_optimizer = None
    if checkpoint["exposure_optimizer"] is not None:
        # lr is a required constructor arg but gets fully overwritten by
        # load_state_dict below (it restores each param_group's saved lr) --
        # same reasoning for camera_pose_optimizer further down.
        exposure_optimizer = torch.optim.Adam([exposure_log_gains, exposure_biases], lr=1e-3)
        exposure_optimizer.load_state_dict(checkpoint["exposure_optimizer"])

    camera_pose_deltas = _load_optional_param("camera_pose_deltas")
    camera_pose_optimizer = None
    if checkpoint["camera_pose_optimizer"] is not None:
        camera_pose_optimizer = torch.optim.Adam([camera_pose_deltas], lr=1e-5)
        camera_pose_optimizer.load_state_dict(checkpoint["camera_pose_optimizer"])

    return GaussianTrainingState(
        params=params,
        optimizers=optimizers,
        strategy=strategy,
        strategy_state=strategy_state,
        step=checkpoint["step"],
        learning_rate_decay_applied=checkpoint["learning_rate_decay_applied"],
        exposure_log_gains=exposure_log_gains,
        exposure_biases=exposure_biases,
        exposure_optimizer=exposure_optimizer,
        sh_degree=checkpoint["sh_degree"],
        camera_pose_deltas=camera_pose_deltas,
        camera_pose_optimizer=camera_pose_optimizer,
        camera_sampling=checkpoint.get("camera_sampling", "sequential"),
        camera_sampling_seed=checkpoint.get("camera_sampling_seed", 0),
    )


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
