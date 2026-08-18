"""Validate checkpointed DUSt3R geometry and train Gaussians without Jupyter.

This is the second resumable stage of the full-room pipeline.  It consumes the
files produced by ``run_full_room_reconstruction.py`` and never reruns DUSt3R.
Every successful training chunk writes both exact optimizer state and a normal
Gaussian scene export before returning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))

from experiments.convert_bin_to_ply import convert_npz


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


def load_scene(run_dir: Path, target_long_edge: int, max_initial_points: int):
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
    trusted = np.flatnonzero(counts >= threshold).tolist()
    excluded = np.flatnonzero(counts < threshold).tolist()
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

    scene = ReconstructedScene(
        points_xyz=np.concatenate(initial_xyz).astype(np.float32),
        points_rgb=np.concatenate(initial_rgb).astype(np.float32),
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
        "target_size": list(camera_images[0].shape[:2]),
        "reprojection": reprojection,
    }
    return scene, report


def evaluate_latest(scene, run_dir: Path, views: int = 8) -> None:
    import torch
    from gsplat import rasterization
    from runner import _ssim

    exports = sorted(run_dir.glob("gaussians_step*.npz"))
    if not exports:
        raise RuntimeError(f"No Gaussian export found in {run_dir}")
    export = exports[-1]
    data = np.load(export)
    device = torch.device("cuda")
    means = torch.tensor(data["means"], device=device)
    quats = torch.tensor(data["quats"], device=device)
    scales = torch.tensor(data["scales"], device=device)
    opacities = torch.tensor(data["opacities"], device=device)
    colors = torch.tensor(data["colors"], device=device)
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
                width, height,
            )
            rendered = render[0].clamp(0, 1)
            mse = torch.mean((rendered - target) ** 2)
            psnr = float(-10.0 * torch.log10(mse.clamp_min(1e-12)))
            ssim = float(_ssim(rendered, target))
            metrics.append({"view": int(index), "psnr": psnr, "ssim": ssim})
            real_u8 = (target.cpu().numpy() * 255).astype(np.uint8)
            render_u8 = (rendered.cpu().numpy() * 255).astype(np.uint8)
            rows.append(np.concatenate((real_u8, render_u8), axis=1))
    report = {
        "source": export.name,
        "mean_psnr": float(np.mean([item["psnr"] for item in metrics])),
        "mean_ssim": float(np.mean([item["ssim"] for item in metrics])),
        "views": metrics,
    }
    report_path = run_dir / "evaluation_latest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    montage = Image.fromarray(np.concatenate(rows, axis=0))
    montage.thumbnail((1600, 10000), Image.Resampling.LANCZOS)
    montage_path = run_dir / "evaluation_latest.jpg"
    montage.save(montage_path, quality=92)
    print(json.dumps(report, indent=2))
    print(f"Evaluation montage: {montage_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--target-long-edge", type=int, default=1024)
    parser.add_argument("--max-initial-points", type=int, default=160_000)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    scene, report = load_scene(
        args.run_dir, args.target_long_edge, args.max_initial_points
    )
    report_path = args.run_dir / "geometry_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.validate_only:
        print(f"Validation complete: {report_path}")
        return
    if args.evaluate_only:
        evaluate_latest(scene, args.run_dir)
        return

    import torch
    from runner import load_training_state, save_training_state, train_gaussian_splatting

    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian training requires a CUDA GPU")
    checkpoints = args.run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / "training_state_latest.pt"
    state = load_training_state(latest, torch.device("cuda")) if latest.exists() else None
    start_step = state.step if state is not None else 0
    gaussians, state = train_gaussian_splatting(
        scene,
        num_iterations=args.iterations,
        densify_until=2400,
        training_state=state,
        optimize_camera_exposure=True,
        optimize_camera_poses=False,
        return_training_state=True,
    )
    save_training_state(state, latest)
    state_copy = checkpoints / f"training_state_step{state.step}.pt"
    save_training_state(state, state_copy)
    export = args.run_dir / f"gaussians_step{state.step}.npz"
    np.savez(export, **gaussians)
    ply = args.run_dir / f"gaussians_step{state.step}.ply"
    convert_npz(export, ply)
    print(
        f"Training complete: step {start_step} -> {state.step}, "
        f"{len(gaussians['means'])} Gaussians\nCheckpoint: {latest}\nExport: {ply}",
        flush=True,
    )


if __name__ == "__main__":
    main()
