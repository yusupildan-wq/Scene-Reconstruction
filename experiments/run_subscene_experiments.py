"""Controlled nested-view Gaussian experiments for the full-room checkpoint.

This module is deliberately separate from the production/direct full-room CLI.
It never reconstructs geometry and writes each treatment/subset into its own
directory.  The default region (views 46--61, centred on the sharp and strongly
consistent 53/54 transition) is nested so view-count is the only changing
dataset variable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))

from experiments.run_full_room_gaussians import (
    _crop_region,
    _load_image,
    _reprojection_summary,
)


DEFAULT_NESTED_SUBSETS = {
    1: (54,),
    4: (52, 53, 54, 55),
    8: tuple(range(50, 58)),
    16: tuple(range(46, 62)),
}
DEFAULT_EVALUATION_VIEWS = (54,)


@dataclass(frozen=True)
class ExperimentConfig:
    updates_per_camera: int = 300
    points_per_camera: int = 3_174
    seed: int = 20260818
    target_long_edge: int = 1440
    evaluation_views: tuple[int, ...] = DEFAULT_EVALUATION_VIEWS
    fusion_mode: str = "concatenation"
    training_mode: str = "baseline"
    sh_degree: int | None = 2


def parse_index_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("view lists must contain integers") from exc
    if not result or len(set(result)) != len(result) or any(index < 0 for index in result):
        raise argparse.ArgumentTypeError("view lists must be non-empty, unique, non-negative")
    return result


def nested_subsets(region: Sequence[int], counts: Sequence[int]) -> dict[int, tuple[int, ...]]:
    """Select centered, nested subsets while preserving chronological order."""
    region = tuple(region)
    if len(set(region)) != len(region) or tuple(sorted(region)) != region:
        raise ValueError("region views must be unique and strictly increasing")
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("subset counts must be positive")
    if max(counts) > len(region):
        raise ValueError("largest subset exceeds the selected region")
    result = {}
    for count in sorted(set(counts)):
        # For an even-sized region, use the later of its two centre views.  The
        # verified 53/54 transition is excellent and view 54 is therefore the
        # one fixed evaluation identity present in every default subset.
        start = (len(region) - count + 1) // 2
        result[count] = region[start : start + count]
    selected = list(result.values())
    for smaller, larger in zip(selected, selected[1:]):
        if not set(smaller).issubset(larger):
            raise AssertionError("generated subsets are not nested")
    return result


def _json_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _sample_observations(
    points_by_view: Sequence[np.ndarray],
    colors_by_view: Sequence[np.ndarray],
    original_view_indices: Sequence[int],
    points_per_camera: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(points_by_view) != len(original_view_indices):
        raise ValueError("original view identities must match observation arrays")
    sampled_points, sampled_colors = [], []
    for original_index, points, colors in zip(
        original_view_indices, points_by_view, colors_by_view
    ):
        if len(points) != len(colors):
            raise ValueError("point/color observation counts differ")
        if not len(points):
            raise ValueError("a selected view has no valid observations")
        # Seed by persistent geometry identity, not subset-local position.  The
        # same camera therefore contributes exactly the same samples in every
        # nested 1/4/8/16-view treatment.
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(original_index)]))
        if len(points) > points_per_camera:
            chosen = rng.choice(len(points), points_per_camera, replace=False)
            points, colors = points[chosen], colors[chosen]
        sampled_points.append(np.asarray(points, dtype=np.float32))
        sampled_colors.append(np.asarray(colors, dtype=np.float32))
    return sampled_points, sampled_colors


def fuse_sampled_observations(
    points_by_view: Sequence[np.ndarray],
    colors_by_view: Sequence[np.ndarray],
    mode: str,
    *,
    robust_config=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Clean integration boundary shared with ``experiments.robust_fusion``."""
    from experiments.robust_fusion import (
        concatenation_fusion,
        robust_consensus_fusion,
    )

    if mode == "concatenation":
        return concatenation_fusion(points_by_view, colors_by_view)
    if mode == "robust_consensus":
        xyz, rgb, stats = robust_consensus_fusion(
            points_by_view, colors_by_view, config=robust_config
        )
        return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.float32), stats
    raise ValueError(f"unsupported fusion mode: {mode}")


def conflict_training_options(
    mode: str, camera_count: int, seed: int
) -> dict[str, int | str]:
    """Resolve the trainer treatment without changing baseline defaults.

    The conflict-aware cadence is the smallest whole-camera-cycle interval at
    least as long as the historical 100-step refinement window.  This makes
    every window contain every camera while keeping cadence comparable across
    the 1/4/8/16-view treatments.
    """
    if mode == "baseline":
        return {}
    if mode != "conflict_aware":
        raise ValueError(f"unsupported training mode: {mode}")
    if camera_count < 1:
        raise ValueError("camera_count must be positive")
    cycles = math.ceil(100 / camera_count)
    return {
        "camera_sampling": "shuffled_cycle",
        "camera_sampling_seed": seed,
        "densify_interval_camera_cycles": cycles,
    }


def resolve_robust_config(config, camera_count: int):
    """Make the one-view fusion treatment runnable without weakening others."""
    if config is None:
        return None
    if camera_count < 1:
        raise ValueError("camera_count must be positive")
    if config.min_view_support <= camera_count:
        return config
    return replace(config, min_view_support=camera_count)


def _pair_geometry_metrics(
    source_points: np.ndarray,
    source_mask: np.ndarray,
    target_points: np.ndarray,
    target_mask: np.ndarray,
    target_K: np.ndarray,
    target_viewmat: np.ndarray,
    sample_limit: int = 20_000,
) -> dict:
    source = source_points[source_mask]
    if len(source) > sample_limit:
        source = source[np.linspace(0, len(source) - 1, sample_limit, dtype=int)]
    hom = np.concatenate((source, np.ones((len(source), 1), dtype=source.dtype)), axis=1)
    camera = (target_viewmat @ hom.T).T[:, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    source, camera = source[valid], camera[valid]
    projected = (target_K @ camera.T).T
    xy = projected[:, :2] / projected[:, 2:3]
    px, py = np.rint(xy[:, 0]).astype(int), np.rint(xy[:, 1]).astype(int)
    height, width = target_mask.shape
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    source, camera, px, py = source[valid], camera[valid], px[valid], py[valid]
    valid = target_mask[py, px]
    source, camera, px, py = source[valid], camera[valid], px[valid], py[valid]
    if not len(source):
        return {"overlap": 0.0, "median_world_thickness": None,
                "p95_world_thickness": None, "median_relative_disagreement": None,
                "inlier_ratio_3pct": 0.0}
    distances = np.linalg.norm(source - target_points[py, px], axis=1)
    relative = distances / np.maximum(np.abs(camera[:, 2]), 1e-6)
    return {
        "overlap": float(len(source) / max(int(source_mask.sum()), 1)),
        "median_world_thickness": float(np.median(distances)),
        "p95_world_thickness": float(np.percentile(distances, 95)),
        "median_relative_disagreement": float(np.median(relative)),
        "inlier_ratio_3pct": float(np.mean(relative <= 0.03)),
    }


def geometry_consistency_report(points, masks, intrinsics, poses, original_indices) -> dict:
    pairs = []
    for offset in range(len(points) - 1):
        forward = _pair_geometry_metrics(
            points[offset], masks[offset], points[offset + 1], masks[offset + 1],
            intrinsics[offset + 1], np.linalg.inv(poses[offset + 1]),
        )
        backward = _pair_geometry_metrics(
            points[offset + 1], masks[offset + 1], points[offset], masks[offset],
            intrinsics[offset], np.linalg.inv(poses[offset]),
        )
        pair = {"source_original_view": int(original_indices[offset]),
                "target_original_view": int(original_indices[offset + 1])}
        for key in forward:
            values = [value for value in (forward[key], backward[key]) if value is not None]
            pair[key] = float(np.mean(values)) if values else None
        pairs.append(pair)
    finite = lambda key: [pair[key] for pair in pairs if pair[key] is not None]
    return {
        "pairs": pairs,
        "median_world_thickness": (float(np.median(finite("median_world_thickness")))
                                   if finite("median_world_thickness") else None),
        "p95_world_thickness": (float(np.median(finite("p95_world_thickness")))
                                if finite("p95_world_thickness") else None),
        "median_relative_disagreement": (
            float(np.median(finite("median_relative_disagreement")))
            if finite("median_relative_disagreement") else None
        ),
    }


def load_subscene(run_dir: Path, original_indices: Sequence[int], config: ExperimentConfig,
                  robust_config=None):
    from runner import ReconstructedScene

    geometry = run_dir / "geometry"
    if not (geometry / "COMPLETE.json").exists():
        raise RuntimeError("geometry checkpoint is incomplete")
    camera_data = np.load(geometry / "cameras.npz")
    poses_all = camera_data["poses"].astype(np.float32)
    intrinsics_all = camera_data["intrinsics"].astype(np.float32)
    frame_names = camera_data["frame_names"].astype(str)
    indices = tuple(int(index) for index in original_indices)
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(poses_all) for i in indices):
        raise ValueError("original view indices are invalid or duplicated")

    poses, intrinsics, points, masks, low_images = [], [], [], [], []
    for index in indices:
        frame_path = run_dir / "frames" / frame_names[index]
        point_map = np.load(geometry / f"points_{index:04d}.npy")
        mask = np.load(geometry / f"mask_{index:04d}.npy").astype(bool)
        if point_map.shape[:2] != mask.shape:
            raise RuntimeError(f"view {index}: point/mask shape mismatch")
        low_image = _load_image(frame_path, mask.shape[1], mask.shape[0])
        poses.append(poses_all[index]); intrinsics.append(intrinsics_all[index])
        points.append(point_map); masks.append(mask); low_images.append(low_image)

    viewmats = [np.linalg.inv(pose).astype(np.float32) for pose in poses]
    reprojection = _reprojection_summary(points, masks, intrinsics, viewmats)
    if reprojection["median_pixels"] > 2 or reprojection["p95_pixels"] > 10:
        raise RuntimeError(f"subscene failed reprojection validation: {reprojection}")
    geometry_report = geometry_consistency_report(points, masks, intrinsics, poses, indices)
    sampled_points, sampled_colors = _sample_observations(
        [point[mask] for point, mask in zip(points, masks)],
        [image[mask] for image, mask in zip(low_images, masks)],
        indices,
        config.points_per_camera, config.seed,
    )
    xyz, rgb, fusion_stats = fuse_sampled_observations(
        sampled_points, sampled_colors, config.fusion_mode, robust_config=robust_config
    )

    camera_images, scaled_Ks = [], []
    for index, K in zip(indices, intrinsics):
        path = run_dir / "frames" / frame_names[index]
        source = exif_transpose(Image.open(path)).convert("RGB")
        crop, processed_size = _crop_region(*source.size)
        cropped = source.crop(tuple(round(value) for value in crop))
        scale = min(1.0, config.target_long_edge / max(cropped.size))
        final_size = tuple(round(value * scale) for value in cropped.size)
        camera_images.append(np.asarray(cropped.resize(final_size, Image.Resampling.LANCZOS),
                                        dtype=np.float32) / 255.0)
        scaled = K.copy()
        scaled[0, (0, 2)] *= final_size[0] / processed_size[0]
        scaled[1, (1, 2)] *= final_size[1] / processed_size[1]
        scaled_Ks.append(scaled)
    scene = ReconstructedScene(xyz, rgb, viewmats, scaled_Ks, camera_images)
    return scene, {
        "original_view_indices": list(indices),
        "frame_names": [str(frame_names[i]) for i in indices],
        "reprojection": reprojection,
        "geometry_consistency": geometry_report,
        "fusion": fusion_stats,
        "initial_points_per_original_view": [len(item) for item in sampled_points],
    }


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = image[..., :3] @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1] = (gray[2:] - gray[:-2]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def image_metrics(rendered, target, ssim_function) -> dict[str, float]:
    rendered = np.clip(np.asarray(rendered, dtype=np.float32), 0, 1)
    target = np.clip(np.asarray(target, dtype=np.float32), 0, 1)
    mse = float(np.mean((rendered - target) ** 2))
    target_edges = _gradient_magnitude(target)
    render_edges = _gradient_magnitude(rendered)
    threshold = float(np.percentile(target_edges, 80))
    edge_mask = target_edges >= max(threshold, 1e-6)
    edge_mse = float(np.mean((rendered[edge_mask] - target[edge_mask]) ** 2))
    return {
        "psnr": float(-10 * math.log10(max(mse, 1e-12))),
        "ssim": float(ssim_function(rendered, target)),
        "edge_psnr": float(-10 * math.log10(max(edge_mse, 1e-12))),
        "gradient_mae": float(np.mean(np.abs(render_edges - target_edges))),
        "edge_threshold": threshold,
    }


def evaluate_experiment(scene, original_indices, evaluation_views, output_dir, state) -> dict:
    import torch
    from gsplat import rasterization
    from runner import _ssim

    missing = sorted(set(evaluation_views) - set(original_indices))
    if missing:
        raise ValueError(f"evaluation views are absent from subset: {missing}")
    rows, results = [], []
    with torch.no_grad():
        for original_index in evaluation_views:
            local_index = original_indices.index(original_index)
            target = torch.tensor(scene.camera_images[local_index], device="cuda")
            height, width = target.shape[:2]
            colors = state.params["colors"]
            render, _, _ = rasterization(
                state.params["means"], state.params["quats"], torch.exp(state.params["scales"]),
                torch.sigmoid(state.params["opacities"]), colors,
                torch.tensor(scene.camera_viewmats[local_index], device="cuda").unsqueeze(0),
                torch.tensor(scene.camera_Ks[local_index], device="cuda").unsqueeze(0),
                width, height, sh_degree=state.sh_degree,
            )
            raw_render = render[0]
            canonical = raw_render.clamp(0, 1)
            adjusted = canonical
            if state.exposure_log_gains is not None and state.exposure_biases is not None:
                adjusted = (raw_render * torch.exp(state.exposure_log_gains[local_index])
                            + state.exposure_biases[local_index]).clamp(0, 1)
            target_np, canonical_np, adjusted_np = (item.detach().cpu().numpy()
                                                     for item in (target, canonical, adjusted))
            ssim_np = lambda a, b: float(_ssim(torch.tensor(a), torch.tensor(b)))
            results.append({
                "original_view": int(original_index),
                "local_view": int(local_index),
                "canonical": image_metrics(canonical_np, target_np, ssim_np),
                "exposure_adjusted": image_metrics(adjusted_np, target_np, ssim_np),
            })
            rows.append(np.concatenate((target_np, canonical_np, adjusted_np), axis=1))
    summary = {
        family: {metric: float(np.mean([view[family][metric] for view in results]))
                 for metric in ("psnr", "ssim", "edge_psnr", "gradient_mae")}
        for family in ("canonical", "exposure_adjusted")
    }
    montage = (np.concatenate(rows, axis=0).clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(montage).save(output_dir / "evaluation.jpg", quality=92)
    return {"views": results, "mean": summary}


def _assert_resume_compatible(manifest_path: Path, expected: dict) -> None:
    if not manifest_path.exists():
        return
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Compare the JSON representation: dataclass tuples intentionally become
    # arrays on disk and must not make an otherwise identical resume fail.
    expected = json.loads(json.dumps(expected))
    comparable = {key: actual[key] for key in expected if key in actual}
    if comparable != expected:
        raise RuntimeError(f"existing experiment configuration differs: {manifest_path}")


def run_one(run_dir, output_dir, original_indices, config, robust_config=None) -> dict:
    import torch
    from experiments.convert_bin_to_ply import convert_npz
    from runner import load_training_state, save_training_state, train_gaussian_splatting

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_robust_config = robust_config
    if config.fusion_mode == "robust_consensus":
        # The one-view treatment cannot demonstrate cross-view support, but it
        # must remain runnable to reveal whether within-view voxel consolidation
        # alone changes quality. Larger subsets retain the requested threshold.
        effective_robust_config = resolve_robust_config(
            robust_config, len(original_indices)
        )
    trainer_options = conflict_training_options(
        config.training_mode, len(original_indices), config.seed
    )
    manifest = {**asdict(config), "evaluation_views": list(config.evaluation_views),
                "original_view_indices": list(original_indices),
                "evaluation_protocol": "exact_training_camera",
                "target_iterations": config.updates_per_camera * len(original_indices),
                "resolved_trainer_options": trainer_options}
    if effective_robust_config is not None:
        manifest["robust_fusion_config"] = asdict(effective_robust_config)
    manifest_path = output_dir / "experiment_manifest.json"
    _assert_resume_compatible(manifest_path, manifest)
    _json_write(manifest_path, manifest)
    scene, geometry_report = load_subscene(
        run_dir, original_indices, config, effective_robust_config
    )
    _json_write(output_dir / "pretraining_geometry.json", geometry_report)
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian experiment requires CUDA; setup reports were written, no training started")
    checkpoint = output_dir / "checkpoints" / "training_state_latest.pt"
    state = load_training_state(checkpoint, torch.device("cuda")) if checkpoint.exists() else None
    target_iterations = manifest["target_iterations"]
    completed = state.step if state is not None else 0
    if completed > target_iterations:
        raise RuntimeError("checkpoint exceeds configured target iterations")
    remaining = target_iterations - completed
    if remaining:
        gaussians, state = train_gaussian_splatting(
            scene, num_iterations=remaining, training_state=state,
            optimize_camera_exposure=True, optimize_camera_poses=False,
            sh_degree=config.sh_degree, return_training_state=True,
            **trainer_options,
        )
        save_training_state(state, checkpoint)
        export = output_dir / f"gaussians_step{state.step}.npz"
        np.savez(export, **gaussians)
        convert_npz(export, output_dir / f"gaussians_step{state.step}.ply")
    result = {"status": "complete", "manifest": manifest,
              "geometry": geometry_report,
              "evaluation": evaluate_experiment(scene, list(original_indices),
                                                config.evaluation_views, output_dir, state)}
    _json_write(output_dir / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fusion-mode", choices=("concatenation", "robust_consensus"), default="concatenation")
    parser.add_argument("--training-mode", choices=("baseline", "conflict_aware"), default="baseline")
    parser.add_argument("--subset-counts", type=parse_index_list, default=(1, 4, 8, 16))
    parser.add_argument("--region-views", type=parse_index_list, default=tuple(range(46, 62)))
    parser.add_argument("--evaluation-views", type=parse_index_list, default=DEFAULT_EVALUATION_VIEWS)
    parser.add_argument("--updates-per-camera", type=int, default=300)
    parser.add_argument("--points-per-camera", type=int, default=3_174)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--target-long-edge", type=int, default=1440)
    parser.add_argument("--sh-degree", type=int, default=2)
    parser.add_argument("--fusion-voxel-size", type=float)
    parser.add_argument("--fusion-min-view-support", type=int)
    parser.add_argument("--fusion-max-position-disagreement", type=float)
    parser.add_argument("--fusion-mad-multiplier", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_output = args.output_dir.resolve()
    resolved_run = args.run_dir.resolve()
    if resolved_output == resolved_run or resolved_output.is_relative_to(resolved_run):
        raise SystemExit("--output-dir must be outside the preserved reconstruction")
    if args.updates_per_camera <= 0 or args.points_per_camera <= 0 or args.target_long_edge <= 0:
        raise SystemExit("updates, points, and resolution must be positive")
    subsets = nested_subsets(args.region_views, args.subset_counts)
    if not set(args.evaluation_views).issubset(set(subsets[min(subsets)])):
        raise SystemExit("fixed evaluation views must be present in every nested subset")
    robust_config = None
    if args.fusion_mode == "robust_consensus":
        try:
            from experiments.robust_fusion import RobustFusionConfig
        except ImportError as exc:
            raise SystemExit("robust_consensus implementation is not integrated") from exc
        provided = {
            "voxel_size": args.fusion_voxel_size,
            "min_view_support": args.fusion_min_view_support,
            "max_position_disagreement": args.fusion_max_position_disagreement,
            "mad_multiplier": args.fusion_mad_multiplier,
        }
        robust_config = RobustFusionConfig(**{key: value for key, value in provided.items() if value is not None})
    config = ExperimentConfig(
        updates_per_camera=args.updates_per_camera,
        points_per_camera=args.points_per_camera,
        seed=args.seed,
        target_long_edge=args.target_long_edge,
        evaluation_views=tuple(args.evaluation_views),
        fusion_mode=args.fusion_mode,
        training_mode=args.training_mode,
        sh_degree=args.sh_degree,
    )
    experiment_summary = {"config": asdict(config), "subsets": []}
    for count, views in subsets.items():
        destination = (
            args.output_dir
            / args.fusion_mode
            / args.training_mode
            / f"views_{count:03d}"
        )
        result = run_one(args.run_dir, destination, views, config, robust_config)
        experiment_summary["subsets"].append({"view_count": count, "views": list(views),
                                               "result": str(destination / "result.json"),
                                               "metrics": result["evaluation"]["mean"]})
        _json_write(
            args.output_dir
            / args.fusion_mode
            / args.training_mode
            / "experiment_summary.json",
            experiment_summary,
        )


if __name__ == "__main__":
    main()
