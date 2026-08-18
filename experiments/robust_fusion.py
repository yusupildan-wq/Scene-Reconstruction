"""Experimental robust fusion for aligned multi-view point observations.

This module is deliberately independent of Gaussian training.  It consumes the
same per-view samples used by the concatenation baseline and returns one robust
consensus observation for spatial cells supported by multiple source views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RobustFusionConfig:
    """Thresholds in the world-coordinate units of the aligned point maps."""

    voxel_size: float = 0.01
    min_view_support: int = 2
    max_position_disagreement: float = 0.02
    mad_multiplier: float = 3.0

    def validate(self) -> None:
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        if self.min_view_support < 1:
            raise ValueError("min_view_support must be at least one")
        if (
            not np.isfinite(self.max_position_disagreement)
            or self.max_position_disagreement <= 0
        ):
            raise ValueError(
                "max_position_disagreement must be finite and positive"
            )
        if not np.isfinite(self.mad_multiplier) or self.mad_multiplier <= 0:
            raise ValueError("mad_multiplier must be finite and positive")


def concatenation_fusion(
    points_by_view: Sequence[np.ndarray],
    colors_by_view: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the historical concatenation behavior and comparable statistics."""
    _validate_inputs(points_by_view, colors_by_view)
    points = np.concatenate(points_by_view).astype(np.float32, copy=False)
    colors = np.concatenate(colors_by_view).astype(np.float32, copy=False)
    count = int(len(points))
    return points, colors, {
        "mode": "concatenation",
        "input_observations": count,
        "fused_points": count,
        "rejected_observations": 0,
        "duplicate_observations_suppressed": 0,
        "supported_voxels": count,
        "support_distribution": {"min": 1, "median": 1.0, "p95": 1.0, "max": 1},
        "estimated_surface_thickness_before": {"median": 0.0, "p95": 0.0},
        "estimated_surface_thickness_after": {"median": 0.0, "p95": 0.0},
        "spatial_disagreement_before": {"median": 0.0, "p95": 0.0},
        "spatial_disagreement_after": {"median": 0.0, "p95": 0.0},
    }


def robust_consensus_fusion(
    points_by_view: Sequence[np.ndarray],
    colors_by_view: Sequence[np.ndarray],
    config: RobustFusionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fuse spatially co-located observations using cross-view consensus.

    Observations are first grouped by a regular 3D grid.  Within every cell,
    each source view is reduced to one median position/color, preventing a dense
    view from dominating support.  Cross-view position medians define the
    consensus; contributors beyond both a robust MAD bound and the configured
    absolute bound are rejected.  A cell is emitted only if enough distinct
    source views remain.  Colors are medians of the accepted per-view colors.

    This intentionally does not infer visibility or bridge neighboring voxels.
    Those require stronger geometric evidence than the current checkpoint
    stores and are therefore left for a later treatment.
    """
    config = config or RobustFusionConfig()
    config.validate()
    _validate_inputs(points_by_view, colors_by_view)

    input_count = int(sum(len(points) for points in points_by_view))
    entries: list[
        tuple[tuple[int, int, int], int, np.ndarray, np.ndarray, int]
    ] = []
    invalid_count = 0
    for view_index, (points, colors) in enumerate(zip(points_by_view, colors_by_view)):
        finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        invalid_count += int((~finite).sum())
        valid_points = np.asarray(points[finite], dtype=np.float64)
        valid_colors = np.asarray(colors[finite], dtype=np.float64)
        if not len(valid_points):
            continue
        keys = np.floor(valid_points / config.voxel_size).astype(np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        keys, valid_points, valid_colors = keys[order], valid_points[order], valid_colors[order]
        starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1]
        ends = np.r_[starts[1:], len(keys)]
        for start, end in zip(starts, ends):
            entries.append(
                (
                    tuple(int(value) for value in keys[start]),
                    view_index,
                    np.median(valid_points[start:end], axis=0),
                    np.median(valid_colors[start:end], axis=0),
                    int(end - start),
                )
            )

    entries.sort(key=lambda item: item[0])
    fused_points: list[np.ndarray] = []
    fused_colors: list[np.ndarray] = []
    supports: list[int] = []
    accepted_raw_observations = 0
    rejected_raw_observations = invalid_count
    disagreements: list[float] = []
    index = 0
    while index < len(entries):
        end = index + 1
        while end < len(entries) and entries[end][0] == entries[index][0]:
            end += 1
        group = entries[index:end]
        positions = np.stack([item[2] for item in group])
        colors = np.stack([item[3] for item in group])
        center = np.median(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        distance_median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - distance_median)))
        robust_limit = distance_median + config.mad_multiplier * max(mad, 1e-12)
        limit = min(config.max_position_disagreement, robust_limit)
        accepted = distances <= limit
        support = int(accepted.sum())
        if support >= config.min_view_support:
            consensus = np.median(positions[accepted], axis=0)
            residuals = np.linalg.norm(positions[accepted] - consensus, axis=1)
            fused_points.append(consensus)
            fused_colors.append(np.median(colors[accepted], axis=0))
            supports.append(support)
            disagreements.extend(float(value) for value in residuals)
            accepted_raw_observations += sum(
                item[4] for item, keep in zip(group, accepted) if keep
            )
            rejected_raw_observations += sum(
                item[4] for item, keep in zip(group, accepted) if not keep
            )
        else:
            rejected_raw_observations += sum(item[4] for item in group)
        index = end

    if not fused_points:
        raise RuntimeError(
            "Robust fusion rejected every spatial cell; relax voxel/support/"
            "disagreement thresholds"
        )

    fused_xyz = np.asarray(fused_points, dtype=np.float32)
    fused_rgb = np.clip(np.asarray(fused_colors, dtype=np.float32), 0.0, 1.0)
    support_values = np.asarray(supports, dtype=np.float64)
    disagreement_values = np.asarray(disagreements, dtype=np.float64)
    # Input observations consolidated within a view are duplicate-suppressed,
    # while cross-view representatives that fail consensus are rejected.
    duplicate_suppressed = accepted_raw_observations - len(fused_xyz)
    stats = {
        "mode": "robust_consensus",
        "config": {
            "voxel_size": config.voxel_size,
            "min_view_support": config.min_view_support,
            "max_position_disagreement": config.max_position_disagreement,
            "mad_multiplier": config.mad_multiplier,
        },
        "input_observations": input_count,
        "per_view_spatial_representatives": len(entries),
        "fused_points": int(len(fused_xyz)),
        "rejected_observations": int(rejected_raw_observations),
        "duplicate_observations_suppressed": int(duplicate_suppressed),
        "supported_voxels": int(len(fused_xyz)),
        "support_distribution": _distribution(support_values),
        "estimated_surface_thickness_before": _distribution(disagreement_values),
        "estimated_surface_thickness_after": {"median": 0.0, "p95": 0.0},
        "spatial_disagreement_before": _distribution(disagreement_values),
        "spatial_disagreement_after": {"median": 0.0, "p95": 0.0},
    }
    return fused_xyz, fused_rgb, stats


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if not len(values):
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _validate_inputs(
    points_by_view: Sequence[np.ndarray], colors_by_view: Sequence[np.ndarray]
) -> None:
    if not points_by_view or len(points_by_view) != len(colors_by_view):
        raise ValueError("points_by_view and colors_by_view must have equal nonzero length")
    for index, (points, colors) in enumerate(zip(points_by_view, colors_by_view)):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_by_view[{index}] must have shape (N, 3)")
        if colors.shape != points.shape:
            raise ValueError(f"colors_by_view[{index}] must match its point shape")
