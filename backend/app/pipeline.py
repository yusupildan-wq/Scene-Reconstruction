"""CPU-side preprocessing that runs on the backend host, before GPU dispatch.

Frame extraction and blur filtering are classical, deterministic computer vision:
no learned weights, no training, just pixel statistics. They belong on the cheap
CPU host, not the GPU worker -- there's no reason to pay for GPU time while decoding
video and computing a Laplacian variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameExtractionResult:
    frame_paths: list[Path]
    total_frames_seen: int
    selected_frame_count: int


def _blur_score(image: np.ndarray) -> float:
    """Variance of the Laplacian: a sharp image has high-frequency edges, which the
    Laplacian responds strongly to, giving high variance. A blurry image's edges are
    smoothed out, giving low variance. Cheap, standard blur-detection heuristic."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: Path,
    output_dir: Path,
    sample_every_n_frames: int = 10,
    blur_threshold: float = 15.0,
    max_selected_frames: int | None = 64,
) -> FrameExtractionResult:
    """Sample every Nth frame (video from a slow walkthrough has huge redundancy
    between adjacent frames -- SfM needs viewpoint diversity, not every frame), then
    drop frames below a sharpness threshold (motion-blurred frames hurt feature
    matching more than they help)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_index = 0
    selected_frames: list[tuple[Path, float]] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every_n_frames == 0:
                sharpness = _blur_score(frame)
                if sharpness >= blur_threshold:
                    frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    selected_frames.append((frame_path, sharpness))
            frame_index += 1
    finally:
        capture.release()

    # Split the entire capture timeline into equal bins and take the sharpest
    # candidate from each bin. The previous uniform-index subsampling preserved
    # time coverage but could retain a barely-passing motion-blurred frame while
    # deleting a much sharper frame only moments away. Dense reconstruction
    # needs both coverage AND crisp texture for correspondence and supervision.
    if max_selected_frames and len(selected_frames) > max_selected_frames:
        bin_edges = np.linspace(0, len(selected_frames), max_selected_frames + 1)
        keep: set[Path] = set()
        for bin_index in range(max_selected_frames):
            start = int(np.floor(bin_edges[bin_index]))
            stop = int(np.floor(bin_edges[bin_index + 1]))
            stop = max(stop, start + 1)
            path, _score = max(selected_frames[start:stop], key=lambda item: item[1])
            keep.add(path)
        for path, _score in selected_frames:
            if path not in keep:
                path.unlink()
        selected_frames = [item for item in selected_frames if item[0] in keep]

    selected_paths = [path for path, _score in selected_frames]

    return FrameExtractionResult(
        frame_paths=selected_paths,
        total_frames_seen=frame_index,
        selected_frame_count=len(selected_paths),
    )
