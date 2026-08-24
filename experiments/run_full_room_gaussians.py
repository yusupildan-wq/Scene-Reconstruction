"""Validate checkpointed DUSt3R geometry and train Gaussians without Jupyter.

This is the second resumable stage of the full-room pipeline.  It consumes the
files produced by ``run_full_room_reconstruction.py`` and never reruns DUSt3R.
Every successful training chunk writes both exact optimizer state and a normal
Gaussian scene export before returning.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))

from experiments.convert_bin_to_ply import convert_npz
from experiments.robust_fusion import (
    RobustFusionConfig,
    concatenation_fusion,
    robust_consensus_fusion,
)


PROFILES = {
    "baseline": {
        "iterations": 3000,
        "target_long_edge": 1024,
        "max_initial_points": 160_000,
        "densify_until": 2400,
        "learning_rate_decay_step": None,
        "sh_degree": None,
        "exclude_weak_views": False,
    },
    "photoreal": {
        "iterations": 8000,
        "target_long_edge": 1440,
        "max_initial_points": 400_000,
        "densify_until": 6000,
        "learning_rate_decay_step": 7000,
        "sh_degree": 2,
        "exclude_weak_views": True,
    },
}


def resolve_profile(name: str, args: argparse.Namespace) -> dict:
    config = dict(PROFILES[name])
    for key in ("iterations", "target_long_edge", "max_initial_points"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def diagnostic_weak_views(run_dir: Path) -> list[int]:
    path = run_dir / "cross_view_diagnostics.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted({int(index) for index in data.get("weak_views", [])})


def _crop_region(width: int, height: int, size: int = 512, patch: int = 16):
    scale = size / max(width, height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    cx, cy = resized_w // 2, resized_h // 2
    half_w = ((2 * cx) // patch) * patch / 2
    half_h = ((2 * cy) // patch) * patch / 2
    if resized_w == resized_h:
        half_h = 3 * half_w / 4
    crop = tuple(
        value / scale
        for value in (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    )
    return crop, (int(2 * half_w), int(2 * half_h))


def _load_image(path: Path, width: int, height: int) -> np.ndarray:
    image = exif_transpose(Image.open(path)).convert("RGB")
    crop, _ = _crop_region(*image.size)
    image = image.crop(tuple(round(value) for value in crop))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _reprojection_summary(points, masks, intrinsics, viewmats) -> dict[str, float]:
    errors = []
    for point_map, mask, K, viewmat in zip(points, masks, intrinsics, viewmats):
        height, width = mask.shape
        expected_x, expected_y = np.meshgrid(np.arange(width), np.arange(height))
        expected = np.stack((expected_x.ravel(), expected_y.ravel()), axis=1)
        world = point_map.reshape(-1, 3)
        valid = mask.ravel().copy()
        homogeneous = np.concatenate(
            (world, np.ones((len(world), 1), dtype=world.dtype)), axis=1
        )
        camera = (viewmat @ homogeneous.T).T[:, :3]
        valid &= np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
        projected = (K @ camera[valid].T).T
        projected = projected[:, :2] / projected[:, 2:3]
        errors.append(np.linalg.norm(projected - expected[valid], axis=1))
    merged = np.concatenate(errors)
    return {
        "median_pixels": float(np.median(merged)),
        "p95_pixels": float(np.percentile(merged, 95)),
    }


def _cross_view_pair_score(
    source_points: np.ndarray,
    source_mask: np.ndarray,
    target_points: np.ndarray,
    target_mask: np.ndarray,
    target_K: np.ndarray,
    target_viewmat: np.ndarray,
    *,
    relative_tolerance: float = 0.03,
    sample_limit: int = 20_000,
) -> dict[str, float]:
    """Measure whether one view's 3D points agree with another view's surface.

    Self-reprojection only verifies that a point map agrees with the camera that
    produced it.  This projects source-view world points into a *different*
    camera and compares them with that camera's independently predicted world
    points.  The distance tolerance is relative to camera depth so it works at
    different room scales.
    """
    source = source_points[source_mask]
    if len(source) > sample_limit:
        # Evenly spaced sampling is deterministic and covers the whole image,
        # unlike a random sample that can accidentally overrepresent a wall.
        source = source[np.linspace(0, len(source) - 1, sample_limit, dtype=int)]
    homogeneous = np.concatenate(
        (source, np.ones((len(source), 1), dtype=source.dtype)), axis=1
    )
    camera = (target_viewmat @ homogeneous.T).T[:, :3]
    in_front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    camera = camera[in_front]
    source = source[in_front]
    projected = (target_K @ camera.T).T
    xy = projected[:, :2] / projected[:, 2:3]
    px = np.rint(xy[:, 0]).astype(np.int64)
    py = np.rint(xy[:, 1]).astype(np.int64)
    height, width = target_mask.shape
    overlap = (
        (px >= 0) & (px < width) & (py >= 0) & (py < height)
    )
    px, py = px[overlap], py[overlap]
    source = source[overlap]
    depths = camera[overlap, 2]
    target_valid = target_mask[py, px]
    px, py = px[target_valid], py[target_valid]
    source = source[target_valid]
    depths = depths[target_valid]
    if not len(source):
        return {"overlap": 0.0, "median_relative_error": float("inf"), "inlier_ratio": 0.0}
    distances = np.linalg.norm(source - target_points[py, px], axis=1)
    relative_error = distances / np.maximum(np.abs(depths), 1e-6)
    return {
        "overlap": float(len(source) / max(int(source_mask.sum()), 1)),
        "median_relative_error": float(np.median(relative_error)),
        "inlier_ratio": float(np.mean(relative_error <= relative_tolerance)),
    }


def cross_view_diagnostics(points, masks, intrinsics, poses, radius: int = 2) -> dict:
    """Score local camera pairs and identify views that disagree with neighbors."""
    pair_metrics = []
    per_view = [[] for _ in points]
    viewmats = [np.linalg.inv(pose).astype(np.float32) for pose in poses]
    for source_index in range(len(points)):
        for target_index in range(source_index + 1, min(len(points), source_index + radius + 1)):
            forward = _cross_view_pair_score(
                points[source_index], masks[source_index],
                points[target_index], masks[target_index], intrinsics[target_index],
                viewmats[target_index],
            )
            backward = _cross_view_pair_score(
                points[target_index], masks[target_index],
                points[source_index], masks[source_index], intrinsics[source_index],
                viewmats[source_index],
            )
            score = {
                "source": source_index,
                "target": target_index,
                "overlap": float((forward["overlap"] + backward["overlap"]) / 2),
                "median_relative_error": float((forward["median_relative_error"] + backward["median_relative_error"]) / 2),
                "inlier_ratio": float((forward["inlier_ratio"] + backward["inlier_ratio"]) / 2),
            }
            pair_metrics.append(score)
            per_view[source_index].append(score["inlier_ratio"])
            per_view[target_index].append(score["inlier_ratio"])
    view_scores = [float(np.median(values)) if values else 0.0 for values in per_view]
    finite_errors = [item["median_relative_error"] for item in pair_metrics if np.isfinite(item["median_relative_error"])]
    return {
        "radius": radius,
        "relative_tolerance": 0.03,
        "median_pair_inlier_ratio": float(np.median([item["inlier_ratio"] for item in pair_metrics])),
        "median_pair_relative_error": float(np.median(finite_errors)) if finite_errors else None,
        "view_scores": view_scores,
        "weak_views": [index for index, score in enumerate(view_scores) if score < 0.35],
        "pairs": pair_metrics,
    }


def cross_view_diagnostics_from_checkpoint(
    run_dir: Path, radius: int = 2
) -> dict:
    """Run cross-view checks with bounded RAM and no training dependencies."""
    geometry = run_dir / "geometry"
    marker = geometry / "COMPLETE.json"
    if not marker.exists():
        raise RuntimeError(f"Incomplete geometry checkpoint: {marker} is missing")
    camera_data = np.load(geometry / "cameras.npz")
    poses = camera_data["poses"].astype(np.float32)
    intrinsics = camera_data["intrinsics"].astype(np.float32)
    viewmats = [np.linalg.inv(pose).astype(np.float32) for pose in poses]
    pair_metrics = []
    per_view = [[] for _ in poses]
    for source_index in range(len(poses)):
        source_points = np.load(
            geometry / f"points_{source_index:04d}.npy", mmap_mode="r"
        )
        source_mask = np.load(
            geometry / f"mask_{source_index:04d}.npy", mmap_mode="r"
        )
        for target_index in range(
            source_index + 1, min(len(poses), source_index + radius + 1)
        ):
            target_points = np.load(
                geometry / f"points_{target_index:04d}.npy", mmap_mode="r"
            )
            target_mask = np.load(
                geometry / f"mask_{target_index:04d}.npy", mmap_mode="r"
            )
            forward = _cross_view_pair_score(
                source_points, source_mask, target_points, target_mask,
                intrinsics[target_index], viewmats[target_index],
            )
            backward = _cross_view_pair_score(
                target_points, target_mask, source_points, source_mask,
                intrinsics[source_index], viewmats[source_index],
            )
            score = {
                "source": source_index,
                "target": target_index,
                "overlap": float((forward["overlap"] + backward["overlap"]) / 2),
                "median_relative_error": float(
                    (forward["median_relative_error"] + backward["median_relative_error"]) / 2
                ),
                "inlier_ratio": float(
                    (forward["inlier_ratio"] + backward["inlier_ratio"]) / 2
                ),
            }
            pair_metrics.append(score)
            per_view[source_index].append(score["inlier_ratio"])
            per_view[target_index].append(score["inlier_ratio"])
        print(f"Cross-view diagnostic: {source_index + 1}/{len(poses)} views", flush=True)
    view_scores = [float(np.median(values)) if values else 0.0 for values in per_view]
    finite_errors = [
        item["median_relative_error"] for item in pair_metrics
        if np.isfinite(item["median_relative_error"])
    ]
    return {
        "radius": radius,
        "relative_tolerance": 0.03,
        "median_pair_inlier_ratio": float(
            np.median([item["inlier_ratio"] for item in pair_metrics])
        ),
        "median_pair_relative_error": (
            float(np.median(finite_errors)) if finite_errors else None
        ),
        "view_scores": view_scores,
        "weak_views": [index for index, score in enumerate(view_scores) if score < 0.35],
        "pairs": pair_metrics,
    }


def load_scene(
    run_dir: Path,
    target_long_edge: int,
    max_initial_points: int,
    excluded_view_indices: list[int] | None = None,
    fusion_mode: str = "concatenation",
    fusion_config: RobustFusionConfig | None = None,
):
    from runner import ReconstructedScene

    geometry = run_dir / "geometry"
    marker = geometry / "COMPLETE.json"
    if not marker.exists():
        raise RuntimeError(f"Incomplete geometry checkpoint: {marker} is missing")
    camera_data = np.load(geometry / "cameras.npz")
    poses = camera_data["poses"].astype(np.float32)  # camera-to-world
    intrinsics = camera_data["intrinsics"].astype(np.float32)
    frame_names = camera_data["frame_names"].astype(str)
    frame_paths = [run_dir / "frames" / name for name in frame_names]
    points = [np.load(geometry / f"points_{i:04d}.npy") for i in range(len(poses))]
    masks = [np.load(geometry / f"mask_{i:04d}.npy").astype(bool) for i in range(len(poses))]
    counts = np.asarray([int(mask.sum()) for mask in masks])
    threshold = max(1, int(np.median(counts) * 0.1))
    forced_excluded = set(excluded_view_indices or [])
    trusted = [
        int(index) for index in np.flatnonzero(counts >= threshold)
        if int(index) not in forced_excluded
    ]
    excluded = sorted(set(np.flatnonzero(counts < threshold).tolist()) | forced_excluded)
    if len(trusted) < 3:
        raise RuntimeError("Fewer than three geometrically supported views remain")

    viewmats = [np.linalg.inv(poses[i]).astype(np.float32) for i in trusted]
    trusted_points = [points[i] for i in trusted]
    trusted_masks = [masks[i] for i in trusted]
    trusted_Ks = [intrinsics[i] for i in trusted]
    reprojection = _reprojection_summary(
        trusted_points, trusted_masks, trusted_Ks, viewmats
    )
    if reprojection["median_pixels"] > 2.0 or reprojection["p95_pixels"] > 10.0:
        raise RuntimeError(f"Geometry failed reprojection validation: {reprojection}")

    rng = np.random.default_rng(0)
    per_view = max(1, max_initial_points // len(trusted))
    initial_xyz, initial_rgb = [], []
    camera_images, scaled_Ks = [], []
    for new_index, old_index in enumerate(trusted):
        point_map, mask = points[old_index], masks[old_index]
        low_h, low_w = mask.shape
        low_image = _load_image(frame_paths[old_index], low_w, low_h)
        xyz = point_map[mask]
        rgb = low_image[mask]
        if len(xyz) > per_view:
            selected = rng.choice(len(xyz), per_view, replace=False)
            xyz, rgb = xyz[selected], rgb[selected]
        initial_xyz.append(xyz)
        initial_rgb.append(rgb)

        source = exif_transpose(Image.open(frame_paths[old_index])).convert("RGB")
        crop, processed_size = _crop_region(*source.size)
        cropped = source.crop(tuple(round(value) for value in crop))
        scale = min(1.0, target_long_edge / max(cropped.size))
        final_size = tuple(round(value * scale) for value in cropped.size)
        target = cropped.resize(final_size, Image.Resampling.LANCZOS)
        camera_images.append(np.asarray(target, dtype=np.float32) / 255.0)
        K = intrinsics[old_index].copy()
        sx, sy = final_size[0] / processed_size[0], final_size[1] / processed_size[1]
        K[0, (0, 2)] *= sx
        K[1, (1, 2)] *= sy
        scaled_Ks.append(K)

    if fusion_mode == "concatenation":
        fused_xyz, fused_rgb, fusion_stats = concatenation_fusion(
            initial_xyz, initial_rgb
        )
    elif fusion_mode == "robust_consensus":
        fused_xyz, fused_rgb, fusion_stats = robust_consensus_fusion(
            initial_xyz, initial_rgb, fusion_config
        )
    else:
        raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    scene = ReconstructedScene(
        points_xyz=fused_xyz,
        points_rgb=fused_rgb,
        camera_viewmats=viewmats,
        camera_Ks=scaled_Ks,
        camera_images=camera_images,
    )
    report = {
        "total_views": len(poses),
        "trusted_views": len(trusted),
        "excluded_views": excluded,
        "valid_points_min": int(counts.min()),
        "valid_points_median": float(np.median(counts)),
        "valid_points_max": int(counts.max()),
        "initial_points": int(len(scene.points_xyz)),
        "fusion": fusion_stats,
        "target_size": list(camera_images[0].shape[:2]),
        "reprojection": reprojection,
    }
    return scene, report


def _load_exposure_parameters(
    output_dir: Path, camera_count: int, expected_step: int | None = None
):
    """Read learned exposure transforms without constructing training state."""
    import torch

    checkpoint = output_dir / "checkpoints" / "training_state_latest.pt"
    if not checkpoint.exists():
        return None
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if expected_step is not None and int(data.get("step", -1)) != expected_step:
        raise RuntimeError(
            "Exposure checkpoint step does not match Gaussian export step"
        )
    gains = data.get("exposure_log_gains")
    biases = data.get("exposure_biases")
    if gains is None or biases is None:
        return None
    if tuple(gains.shape) != (camera_count, 3) or tuple(biases.shape) != (
        camera_count,
        3,
    ):
        raise RuntimeError(
            "Saved exposure parameters do not match evaluation camera count"
        )
    return gains, biases


def _image_quality(rendered, target, ssim_function) -> dict[str, float]:
    import torch

    mse = torch.mean((rendered - target) ** 2)
    return {
        "psnr": float(-10.0 * torch.log10(mse.clamp_min(1e-12))),
        "ssim": float(ssim_function(rendered, target)),
    }


def evaluate_latest(scene, output_dir: Path, views: int = 8) -> None:
    import torch
    from gsplat import rasterization
    from runner import _ssim

    def step_number(path: Path) -> int:
        match = re.search(r"step(\d+)", path.stem)
        return int(match.group(1)) if match else -1

    exports = sorted(output_dir.glob("gaussians_step*.npz"), key=step_number)
    if not exports:
        raise RuntimeError(f"No Gaussian export found in {output_dir}")
    export = exports[-1]
    data = np.load(export)
    device = torch.device("cuda")
    means = torch.tensor(data["means"], device=device)
    quats = torch.tensor(data["quats"], device=device)
    scales = torch.tensor(data["scales"], device=device)
    opacities = torch.tensor(data["opacities"], device=device)
    sh_degree = int(data["sh_degree"]) if "sh_degree" in data else None
    color_key = "sh_coeffs" if sh_degree is not None else "colors"
    colors = torch.tensor(data[color_key], device=device)
    exposure = _load_exposure_parameters(
        output_dir, len(scene.camera_images), expected_step=step_number(export)
    )
    if exposure is not None:
        exposure_log_gains = exposure[0].to(device)
        exposure_biases = exposure[1].to(device)
    indices = np.linspace(0, len(scene.camera_images) - 1, views, dtype=int)
    metrics, rows = [], []
    with torch.no_grad():
        for index in indices:
            target = torch.tensor(scene.camera_images[index], device=device)
            height, width = target.shape[:2]
            render, _, _ = rasterization(
                means, quats, scales, opacities, colors,
                torch.tensor(scene.camera_viewmats[index], device=device).unsqueeze(0),
                torch.tensor(scene.camera_Ks[index], device=device).unsqueeze(0),
                width, height, sh_degree=sh_degree,
            )
            rendered = render[0].clamp(0, 1)
            canonical = _image_quality(rendered, target, _ssim)
            adjusted = None
            adjusted_render = None
            if exposure is not None:
                adjusted_render = (
                    render[0] * torch.exp(exposure_log_gains[index])
                    + exposure_biases[index]
                ).clamp(0, 1)
                adjusted = _image_quality(adjusted_render, target, _ssim)
            metrics.append(
                {
                    "view": int(index),
                    # Preserve the original fields as canonical metrics for
                    # downstream readers while adding an explicit comparison.
                    **canonical,
                    "canonical": canonical,
                    "exposure_adjusted": adjusted,
                }
            )
            real_u8 = (target.cpu().numpy() * 255).astype(np.uint8)
            render_u8 = (rendered.cpu().numpy() * 255).astype(np.uint8)
            panels = [real_u8, render_u8]
            if adjusted_render is not None:
                panels.append(
                    (adjusted_render.cpu().numpy() * 255).astype(np.uint8)
                )
            rows.append(np.concatenate(panels, axis=1))
    adjusted_metrics = [
        item["exposure_adjusted"]
        for item in metrics
        if item["exposure_adjusted"] is not None
    ]
    report = {
        "source": export.name,
        "mean_psnr": float(np.mean([item["psnr"] for item in metrics])),
        "mean_ssim": float(np.mean([item["ssim"] for item in metrics])),
        "canonical": {
            "mean_psnr": float(np.mean([item["psnr"] for item in metrics])),
            "mean_ssim": float(np.mean([item["ssim"] for item in metrics])),
        },
        "exposure_adjusted": (
            {
                "mean_psnr": float(
                    np.mean([item["psnr"] for item in adjusted_metrics])
                ),
                "mean_ssim": float(
                    np.mean([item["ssim"] for item in adjusted_metrics])
                ),
            }
            if adjusted_metrics
            else None
        ),
        "views": metrics,
    }
    report_path = output_dir / "evaluation_latest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    montage = Image.fromarray(np.concatenate(rows, axis=0))
    montage.thumbnail((1600, 10000), Image.Resampling.LANCZOS)
    montage_path = output_dir / "evaluation_latest.jpg"
    montage.save(montage_path, quality=92)
    print(json.dumps(report, indent=2))
    print(f"Evaluation montage: {montage_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="baseline")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--target-long-edge", type=int)
    parser.add_argument("--max-initial-points", type=int)
    parser.add_argument(
        "--fusion-mode",
        choices=("concatenation", "robust_consensus"),
        default="concatenation",
    )
    parser.add_argument("--fusion-voxel-size", type=float, default=0.01)
    parser.add_argument("--fusion-min-view-support", type=int, default=2)
    parser.add_argument(
        "--fusion-max-position-disagreement", type=float, default=0.02
    )
    parser.add_argument("--fusion-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--diagnose-cross-view", action="store_true")
    parser.add_argument(
        "--camera-sampling",
        choices=("sequential", "shuffled_cycle"),
        default="sequential",
        help="Opt-in balanced camera-cycle order; sequential preserves baseline",
    )
    parser.add_argument("--camera-sampling-seed", type=int, default=0)
    parser.add_argument(
        "--densify-camera-cycles",
        type=int,
        help="Refine once per N complete camera cycles instead of every 100 steps",
    )
    args = parser.parse_args()
    config = resolve_profile(args.profile, args)
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.diagnose_cross_view:
        diagnostics = cross_view_diagnostics_from_checkpoint(args.run_dir)
        output = args.run_dir / "cross_view_diagnostics.json"
        output.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        summary = {key: value for key, value in diagnostics.items() if key != "pairs"}
        print(json.dumps(summary, indent=2))
        print(f"Cross-view diagnostics: {output}")
        return

    excluded = diagnostic_weak_views(args.run_dir) if config["exclude_weak_views"] else []
    scene, report = load_scene(
        args.run_dir,
        config["target_long_edge"],
        config["max_initial_points"],
        excluded,
        args.fusion_mode,
        RobustFusionConfig(
            voxel_size=args.fusion_voxel_size,
            min_view_support=args.fusion_min_view_support,
            max_position_disagreement=args.fusion_max_position_disagreement,
            mad_multiplier=args.fusion_mad_multiplier,
        ),
    )
    report["profile"] = args.profile
    report["training_conflict_config"] = {
        "camera_sampling": args.camera_sampling,
        "camera_sampling_seed": args.camera_sampling_seed,
        "densify_camera_cycles": args.densify_camera_cycles,
    }
    report_path = output_dir / "geometry_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.validate_only:
        print(f"Validation complete: {report_path}")
        return
    if args.evaluate_only:
        evaluate_latest(scene, output_dir)
        return

    import torch
    from runner import load_training_state, save_training_state, train_gaussian_splatting

    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian training requires a CUDA GPU")
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / "training_state_latest.pt"
    state = load_training_state(latest, torch.device("cuda")) if latest.exists() else None
    start_step = state.step if state is not None else 0
    gaussians, state = train_gaussian_splatting(
        scene,
        num_iterations=config["iterations"],
        densify_until=config["densify_until"],
        training_state=state,
        optimize_camera_exposure=True,
        optimize_camera_poses=False,
        learning_rate_decay_step=config["learning_rate_decay_step"],
        sh_degree=config["sh_degree"],
        camera_sampling=args.camera_sampling,
        camera_sampling_seed=args.camera_sampling_seed,
        densify_interval_camera_cycles=args.densify_camera_cycles,
        return_training_state=True,
    )
    save_training_state(state, latest)
    state_copy = checkpoints / f"training_state_step{state.step}.pt"
    save_training_state(state, state_copy)
    export = output_dir / f"gaussians_step{state.step}.npz"
    np.savez(export, **gaussians)
    ply = output_dir / f"gaussians_step{state.step}.ply"
    convert_npz(export, ply)
    print(
        f"Training complete: step {start_step} -> {state.step}, "
        f"{len(gaussians['means'])} Gaussians\nCheckpoint: {latest}\nExport: {ply}",
        flush=True,
    )


if __name__ == "__main__":
    main()
