"""Find candidate loop-closure frame pairs for DUSt3R's pair graph.

DUSt3R's own dust3r.image_pairs.make_pairs (verified against its real source)
only offers 'complete' (all pairs), 'swin'/'logwin' (temporal-window pairing,
optionally wrapping the *first and last* frame of the sequence -- its own
"explicit loop closure"), and 'oneref' (star graph to one reference frame).
None of these detect a camera *revisiting* the same physical area in the
middle of a capture -- e.g. walking past the desk again 40 seconds later --
which is exactly the "weak cross-room connections and loop closure" bottleneck
identified in CLAUDE_HANDOFF.md.

This module is classical, non-learned image processing (a small grayscale
thumbnail per frame, compared by cosine similarity) -- not a neural network,
no pretrained weights, nothing to train. It only proposes CANDIDATE pairs:
DUSt3R still predicts geometry for each pair independently, and the
cross-view consistency gate in dust3r_diagnostics.py still decides whether a
candidate pair actually agrees before it's trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def compute_frame_descriptors(frame_paths: Sequence[str | Path], thumb_size: int = 32) -> np.ndarray:
    """One small, L2-normalized grayscale thumbnail per frame, flattened to a
    vector. A coarse appearance fingerprint -- cheap enough to compare every
    frame against every other frame even for thousands of frames, and good
    enough to flag plausible revisits for DUSt3R to actually verify, not a
    claim of geometric match by itself."""
    descriptors = np.empty((len(frame_paths), thumb_size * thumb_size), dtype=np.float32)
    for row, path in enumerate(frame_paths):
        thumbnail = Image.open(path).convert("L").resize((thumb_size, thumb_size), Image.BILINEAR)
        vector = np.asarray(thumbnail, dtype=np.float32).reshape(-1)
        vector -= vector.mean()
        norm = np.linalg.norm(vector)
        descriptors[row] = vector / norm if norm > 1e-6 else vector
    return descriptors


def find_loop_closure_candidates(
    frame_paths: Sequence[str | Path],
    min_frame_gap: int = 10,
    top_k_per_frame: int = 2,
    similarity_threshold: float = 0.6,
) -> list[tuple[int, int]]:
    """Non-adjacent frame-index pairs whose thumbnails are similar enough to
    plausibly show the same physical area.

    min_frame_gap: only consider pairs already outside DUSt3R's own temporal
    window (no point re-adding a pair swin-N already covers).
    top_k_per_frame: cap candidates per frame so a handful of generic-looking
    frames (e.g. a blank wall) can't flood the pair graph with weak matches.
    similarity_threshold: cosine similarity in [-1, 1]; 0.6 is a deliberately
    conservative starting point -- false-positive candidate pairs just get
    caught by the cross-view consistency gate downstream, so this only needs
    to avoid flooding DUSt3R with obviously-unrelated pairs, not be exact.
    """
    descriptors = compute_frame_descriptors(frame_paths)
    similarity = descriptors @ descriptors.T
    frame_count = len(frame_paths)
    frame_indices = np.arange(frame_count)

    candidates: set[tuple[int, int]] = set()
    for i in range(frame_count):
        far_enough = np.abs(frame_indices - i) >= min_frame_gap
        scores = np.where(far_enough, similarity[i], -np.inf)
        ranked = np.argsort(scores)[::-1][:top_k_per_frame]
        for j in ranked:
            if scores[j] >= similarity_threshold:
                candidates.add((int(min(i, j)), int(max(i, j))))
    return sorted(candidates)
