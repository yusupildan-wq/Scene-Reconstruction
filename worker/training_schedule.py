"""CPU-only scheduling helpers for conflict-aware Gaussian experiments."""

from __future__ import annotations

import numpy as np


def camera_order_for_cycle(
    camera_count: int, cycle: int, mode: str = "sequential", seed: int = 0
) -> np.ndarray:
    """Return one balanced camera cycle, deterministically for resumability."""
    if camera_count < 1:
        raise ValueError("camera_count must be positive")
    if mode == "sequential":
        return np.arange(camera_count)
    if mode == "shuffled_cycle":
        return np.random.default_rng(seed + cycle).permutation(camera_count)
    raise ValueError("camera sampling mode must be 'sequential' or 'shuffled_cycle'")


def camera_cycle_densification_schedule(
    camera_count: int,
    requested_from: int,
    start_step: int,
    cycles_per_refinement: int,
) -> tuple[int, int]:
    """Return a full-cycle-aligned ``(start, interval)`` refinement schedule."""
    if camera_count < 1 or cycles_per_refinement < 1:
        raise ValueError("camera_count and cycles_per_refinement must be positive")
    interval = camera_count * cycles_per_refinement
    warm_start = max(requested_from, start_step + 500)
    aligned_start = (warm_start + interval - 1) // interval * interval
    return aligned_start, interval
