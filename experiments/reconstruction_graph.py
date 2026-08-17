"""Production pair-graph construction for full-room reconstruction.

The notebook is intentionally not the source of truth.  This module owns the
graph policy used by GPU launchers: dense temporal overlap for local geometry,
geometrically verified non-local edges for loop closure, graph validation, and
a JSON audit artifact that makes every expensive run reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from experiments.loop_closure_pairs import find_verified_loop_closures


@dataclass(frozen=True)
class ReconstructionProfile:
    name: str
    max_frames: int
    temporal_radius: int
    alignment_iterations: int
    dust3r_image_size: int = 512


COLAB_PROFILE = ReconstructionProfile(
    name="colab-quality-gate",
    max_frames=64,
    temporal_radius=3,
    alignment_iterations=300,
)

# Intended for an L40S-class 48 GB GPU and >=64 GB host RAM. More cameras add
# actual observations; a higher Gaussian iteration count cannot replace them.
PHOTOREAL_PROFILE = ReconstructionProfile(
    name="runpod-photoreal",
    max_frames=128,
    temporal_radius=5,
    alignment_iterations=600,
)


def temporal_pair_indices(frame_count: int, radius: int) -> set[tuple[int, int]]:
    if radius < 1:
        raise ValueError("temporal radius must be at least 1")
    return {
        (first, second)
        for first in range(frame_count)
        for second in range(first + 1, min(frame_count, first + radius + 1))
    }


def _assert_connected(frame_count: int, pairs: set[tuple[int, int]]) -> None:
    if frame_count == 0:
        raise RuntimeError("Cannot reconstruct an empty frame set")
    adjacency = [set() for _ in range(frame_count)]
    for first, second in pairs:
        adjacency[first].add(second)
        adjacency[second].add(first)
    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    if len(reached) != frame_count:
        missing = sorted(set(range(frame_count)) - reached)
        raise RuntimeError(f"Reconstruction pair graph is disconnected; missing {missing}")


def build_pair_graph(
    frame_paths: Sequence[str | Path],
    profile: ReconstructionProfile = PHOTOREAL_PROFILE,
    manifest_path: str | Path | None = None,
) -> dict:
    """Build and validate an auditable local-plus-loop-closure graph."""
    if len(frame_paths) > profile.max_frames:
        raise ValueError(
            f"Profile {profile.name} allows {profile.max_frames} frames, got "
            f"{len(frame_paths)}. Select frames before constructing the graph."
        )
    temporal = temporal_pair_indices(len(frame_paths), profile.temporal_radius)
    verified = find_verified_loop_closures(
        frame_paths,
        min_frame_gap=profile.temporal_radius + 2,
    )
    loop_pairs = {
        (edge.first, edge.second)
        for edge in verified
        if (edge.first, edge.second) not in temporal
    }
    combined = temporal | loop_pairs
    _assert_connected(len(frame_paths), combined)
    degrees = [0] * len(frame_paths)
    for first, second in combined:
        degrees[first] += 1
        degrees[second] += 1
    manifest = {
        "profile": asdict(profile),
        "frames": [str(path) for path in frame_paths],
        "temporal_pairs": [list(pair) for pair in sorted(temporal)],
        "verified_loop_closures": [edge.as_dict() for edge in verified],
        "combined_pairs": [list(pair) for pair in sorted(combined)],
        "minimum_degree": min(degrees),
        "maximum_degree": max(degrees),
    }
    if manifest_path is not None:
        destination = Path(manifest_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def dust3r_pairs(images: Sequence[dict], manifest: dict) -> list[tuple[dict, dict]]:
    """Convert the backend-independent graph into symmetric DUSt3R pairs."""
    undirected = [tuple(pair) for pair in manifest["combined_pairs"]]
    forward = [(images[first], images[second]) for first, second in undirected]
    return forward + [(second, first) for first, second in forward]
