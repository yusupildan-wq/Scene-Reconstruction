"""CPU diagnostics for DUSt3R camera and dense-geometry consistency.

These checks run before Gaussian optimization. They do not train a model and
do not require CUDA: they project one view's world points into neighboring
views and measure whether independently predicted geometry agrees there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _frame_number(path: str | Path) -> int:
    return int(Path(path).stem.rsplit("_", 1)[-1])


def diagnose_frame_gaps(frame_paths: Sequence[str | Path]) -> np.ndarray:
    """Report source-video gaps between the frames DUSt3R treats as neighbors."""
    frame_numbers = np.asarray([_frame_number(path) for path in frame_paths])
    gaps = np.diff(frame_numbers)
    if len(gaps):
        print(
            "Selected-frame gaps (source frames): "
            f"median={np.median(gaps):.0f}, max={gaps.max()}, "
            f">2x median={int((gaps > 2 * np.median(gaps)).sum())}"
        )
        for index in np.flatnonzero(gaps > 2 * np.median(gaps)):
            print(
                f"  discontinuity {index}->{index + 1}: "
                f"frame {frame_numbers[index]} -> {frame_numbers[index + 1]} "
                f"(gap {gaps[index]})"
            )
    return gaps


def diagnose_cross_view_consistency(
    pts3d,
    masks,
    imgs,
    camera_viewmats: Sequence[np.ndarray],
    intrinsics,
    to_numpy: Callable,
    pairs: Sequence[tuple[int, int]] | None = None,
    max_median_relative_depth_error: float | None = None,
    max_p95_relative_depth_error: float | None = None,
    label: str = "adjacent",
) -> dict[str, float]:
    """Compare views using geometry predicted independently per image.

    For each directed pair i->j, world points from i are projected into j. At
    the resulting target pixels we compare projected depth with j's own depth
    and source color with j's observed color. Occluded points are excluded
    from the color summary using a 5% relative-depth agreement gate.

    `pairs`: undirected (i, j) index pairs to check, each checked in both
    directions. Defaults to temporally-adjacent pairs (i, i+1) -- the original
    behavior of this function. Pass the loop-closure candidates from
    find_loop_closure_candidates() to specifically check whether frames that
    plausibly revisit the same physical area actually agree geometrically --
    that's a different, and for the "cross-room consistency" bottleneck a more
    informative, question than whether consecutive frames agree.

    If either max_*_error threshold is given and exceeded, raises RuntimeError
    -- matching the hard-stop pattern already used by the reprojection-error
    gate elsewhere in the pipeline, instead of only printing a report a human
    has to remember to read before spending GPU time.
    """
    point_maps = [to_numpy(points) for points in pts3d]
    valid_masks = [to_numpy(mask).astype(bool) for mask in masks]
    images = [to_numpy(image) for image in imgs]
    Ks = [to_numpy(K) for K in intrinsics]

    valid_counts = np.asarray([int(mask.sum()) for mask in valid_masks])
    print(
        "Valid DUSt3R points per view: "
        f"min={valid_counts.min()}, median={np.median(valid_counts):.0f}, "
        f"max={valid_counts.max()}, empty_views={int((valid_counts == 0).sum())}"
    )
    weak_threshold = max(1, int(np.median(valid_counts) * 0.1))
    weak_indices = np.flatnonzero(valid_counts < weak_threshold)
    if len(weak_indices):
        print(
            "Views below 10% of median valid geometry: "
            + ", ".join(f"{index} ({valid_counts[index]} points)" for index in weak_indices)
        )

    pair_labels: list[str] = []
    overlaps: list[float] = []
    depth_medians: list[float] = []
    color_medians: list[float] = []
    all_depth_errors: list[np.ndarray] = []
    all_color_errors: list[np.ndarray] = []

    if pairs is None:
        undirected_pairs = [(i, i + 1) for i in range(len(point_maps) - 1)]
    else:
        undirected_pairs = list(pairs)
    directed_pairs = list(undirected_pairs) + [(j, i) for i, j in undirected_pairs]
    if not directed_pairs:
        raise RuntimeError(f"No {label} pairs were provided to check.")
    for source_index, target_index in directed_pairs:
        source_points = point_maps[source_index].reshape(-1, 3)
        source_mask = valid_masks[source_index].reshape(-1)
        source_colors = images[source_index].reshape(-1, 3)
        target_points = point_maps[target_index]
        target_mask = valid_masks[target_index]
        target_image = images[target_index]
        target_h, target_w = target_mask.shape

        source_points = source_points[source_mask]
        source_colors = source_colors[source_mask]
        if not len(source_points):
            continue

        points_h = np.concatenate(
            [source_points, np.ones((len(source_points), 1), dtype=source_points.dtype)], axis=1
        )
        target_camera = (camera_viewmats[target_index] @ points_h.T).T[:, :3]
        projected = (Ks[target_index] @ target_camera.T).T
        projected_uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
        pixel_x = np.rint(projected_uv[:, 0]).astype(int)
        pixel_y = np.rint(projected_uv[:, 1]).astype(int)
        inside = (
            (target_camera[:, 2] > 1e-6)
            & (pixel_x >= 0)
            & (pixel_x < target_w)
            & (pixel_y >= 0)
            & (pixel_y < target_h)
        )
        candidate_indices = np.flatnonzero(inside)
        if not len(candidate_indices):
            continue
        target_valid = target_mask[pixel_y[candidate_indices], pixel_x[candidate_indices]]
        candidate_indices = candidate_indices[target_valid]
        if not len(candidate_indices):
            continue

        sampled_target_points = target_points[
            pixel_y[candidate_indices], pixel_x[candidate_indices]
        ]
        sampled_h = np.concatenate(
            [
                sampled_target_points,
                np.ones((len(sampled_target_points), 1), dtype=sampled_target_points.dtype),
            ],
            axis=1,
        )
        sampled_target_depth = (
            camera_viewmats[target_index] @ sampled_h.T
        ).T[:, 2]
        projected_depth = target_camera[candidate_indices, 2]
        relative_depth_error = np.abs(projected_depth - sampled_target_depth) / np.maximum(
            np.abs(sampled_target_depth), 1e-6
        )

        depth_consistent = relative_depth_error < 0.05
        target_colors = target_image[
            pixel_y[candidate_indices], pixel_x[candidate_indices]
        ]
        color_error = np.abs(source_colors[candidate_indices] - target_colors).mean(axis=1)

        pair_labels.append(f"{source_index}->{target_index}")
        overlaps.append(len(candidate_indices) / len(source_points))
        depth_medians.append(float(np.median(relative_depth_error)))
        color_medians.append(
            float(np.median(color_error[depth_consistent])) if depth_consistent.any() else np.nan
        )
        all_depth_errors.append(relative_depth_error)
        if depth_consistent.any():
            all_color_errors.append(color_error[depth_consistent])

    if not all_depth_errors:
        raise RuntimeError(f"No {label} overlap was measurable; stop before Gaussian training.")

    depth_errors = np.concatenate(all_depth_errors)
    color_errors = np.concatenate(all_color_errors) if all_color_errors else np.asarray([np.nan])
    summary = {
        "median_overlap": float(np.median(overlaps)),
        "median_relative_depth_error": float(np.median(depth_errors)),
        "p95_relative_depth_error": float(np.percentile(depth_errors, 95)),
        "median_color_mae": float(np.nanmedian(color_errors)),
    }
    print(f"Cross-view consistency ({label} pairs, {len(undirected_pairs)} checked):")
    for name, value in summary.items():
        print(f"  {name}: {value:.5f}")

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Adjacent-view geometry and appearance consistency ({label} pairs)")
    axes[0].bar(pair_labels, overlaps)
    axes[0].set_ylabel("Valid overlap fraction")
    axes[1].bar(pair_labels, depth_medians)
    axes[1].set_ylabel("Median relative depth error")
    axes[2].bar(pair_labels, color_medians)
    axes[2].set_ylabel("Median RGB MAE")
    axes[2].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    plt.show()

    if (
        max_median_relative_depth_error is not None
        and summary["median_relative_depth_error"] > max_median_relative_depth_error
    ) or (
        max_p95_relative_depth_error is not None
        and summary["p95_relative_depth_error"] > max_p95_relative_depth_error
    ):
        raise RuntimeError(
            f"{label.capitalize()}-pair geometry is inconsistent "
            f"(median_relative_depth_error={summary['median_relative_depth_error']:.4f}, "
            f"p95={summary['p95_relative_depth_error']:.4f}) -- stop before Gaussian training "
            "and inspect the plot above. This is the diagnostic CLAUDE_HANDOFF.md calls for: "
            "self-reprojection alone can't catch this because a bad camera pose and its own bad "
            "depth map can agree with each other -- this checks whether INDEPENDENTLY predicted "
            f"geometry from two different '{label}' views agrees."
        )
    return summary
