"""Build reliable non-local image pairs for room-scale reconstruction.

Appearance similarity proposes possible revisits; ORB correspondences and
RANSAC then verify real image geometry before an edge is admitted. This avoids
the false loop closures caused by similar blank walls and bright windows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VerifiedLoopClosure:
    first: int
    second: int
    appearance_similarity: float
    match_count: int
    inlier_count: int
    inlier_ratio: float
    spatial_coverage: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_frame_descriptors(
    frame_paths: Sequence[str | Path], thumb_size: int = 48
) -> np.ndarray:
    """Normalized low-frequency colour descriptors used only for retrieval."""
    descriptors = np.empty(
        (len(frame_paths), thumb_size * thumb_size * 3), dtype=np.float32
    )
    for row, path in enumerate(frame_paths):
        thumbnail = Image.open(path).convert("RGB").resize(
            (thumb_size, thumb_size), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(thumbnail, dtype=np.float32) / 255.0
        pixels -= pixels.mean(axis=(0, 1), keepdims=True)
        vector = pixels.reshape(-1)
        norm = np.linalg.norm(vector)
        descriptors[row] = vector / norm if norm > 1e-6 else vector
    return descriptors


def _orb_features(path: str | Path, max_features: int):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read frame: {path}")
    scale = min(1.0, 960.0 / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    detector = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        fastThreshold=12,
    )
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors, image.shape[:2]


def _grid_coverage(
    points: np.ndarray, image_shape: tuple[int, int], grid_size: int = 4
) -> float:
    """Fraction of image grid cells containing an inlier correspondence."""
    if not len(points):
        return 0.0
    height, width = image_shape
    x = np.clip(
        (points[:, 0] / max(width, 1) * grid_size).astype(int), 0, grid_size - 1
    )
    y = np.clip(
        (points[:, 1] / max(height, 1) * grid_size).astype(int), 0, grid_size - 1
    )
    return len(set(zip(x.tolist(), y.tolist()))) / float(grid_size * grid_size)


def _verify_pair(
    first_features,
    second_features,
    *,
    ratio_test: float,
    ransac_threshold_px: float,
    min_matches: int,
    min_inliers: int,
    min_inlier_ratio: float,
    min_spatial_coverage: float,
):
    keypoints_a, descriptors_a, shape_a = first_features
    keypoints_b, descriptors_b, _shape_b = second_features
    if descriptors_a is None or descriptors_b is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    neighbors = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = [
        pair[0]
        for pair in neighbors
        if len(pair) == 2 and pair[0].distance < ratio_test * pair[1].distance
    ]
    if len(good) < min_matches:
        return None
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in good])
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in good])
    _matrix, mask = cv2.findFundamentalMat(
        points_a, points_b, cv2.FM_RANSAC, ransac_threshold_px, 0.999
    )
    if mask is None:
        return None
    inliers = mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(good)
    coverage = _grid_coverage(points_a[inliers], shape_a)
    if (
        inlier_count < min_inliers
        or inlier_ratio < min_inlier_ratio
        or coverage < min_spatial_coverage
    ):
        return None
    return len(good), inlier_count, inlier_ratio, coverage


def find_verified_loop_closures(
    frame_paths: Sequence[str | Path],
    *,
    min_frame_gap: int = 8,
    retrieval_candidates_per_frame: int = 8,
    max_verified_per_frame: int = 3,
    similarity_threshold: float = 0.35,
    max_features: int = 2500,
    ratio_test: float = 0.78,
    ransac_threshold_px: float = 1.5,
    min_matches: int = 28,
    min_inliers: int = 20,
    min_inlier_ratio: float = 0.35,
    min_spatial_coverage: float = 0.20,
) -> list[VerifiedLoopClosure]:
    """Retrieve and geometrically verify non-temporal room revisits."""
    if len(frame_paths) < 2:
        return []
    descriptors = compute_frame_descriptors(frame_paths)
    similarity = descriptors @ descriptors.T
    indices = np.arange(len(frame_paths))
    proposed: set[tuple[int, int]] = set()
    for first in range(len(frame_paths)):
        allowed = np.abs(indices - first) >= min_frame_gap
        scores = np.where(allowed, similarity[first], -np.inf)
        for second in np.argsort(scores)[::-1][:retrieval_candidates_per_frame]:
            if scores[second] >= similarity_threshold:
                proposed.add((min(first, int(second)), max(first, int(second))))

    features = [_orb_features(path, max_features) for path in frame_paths]
    accepted: list[VerifiedLoopClosure] = []
    for first, second in sorted(
        proposed, key=lambda pair: similarity[pair], reverse=True
    ):
        verified = _verify_pair(
            features[first],
            features[second],
            ratio_test=ratio_test,
            ransac_threshold_px=ransac_threshold_px,
            min_matches=min_matches,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            min_spatial_coverage=min_spatial_coverage,
        )
        if verified is None:
            continue
        match_count, inlier_count, inlier_ratio, coverage = verified
        accepted.append(
            VerifiedLoopClosure(
                first,
                second,
                float(similarity[first, second]),
                match_count,
                inlier_count,
                inlier_ratio,
                coverage,
            )
        )

    counts = np.zeros(len(frame_paths), dtype=int)
    balanced: list[VerifiedLoopClosure] = []
    for edge in sorted(
        accepted,
        key=lambda item: (item.inlier_count, item.spatial_coverage),
        reverse=True,
    ):
        if (
            counts[edge.first] >= max_verified_per_frame
            or counts[edge.second] >= max_verified_per_frame
        ):
            continue
        balanced.append(edge)
        counts[edge.first] += 1
        counts[edge.second] += 1
    return sorted(balanced, key=lambda edge: (edge.first, edge.second))


def find_loop_closure_candidates(
    frame_paths: Sequence[str | Path], **kwargs
) -> list[tuple[int, int]]:
    """Compatibility wrapper returning only verified pair indices."""
    return [
        (edge.first, edge.second)
        for edge in find_verified_loop_closures(frame_paths, **kwargs)
    ]
