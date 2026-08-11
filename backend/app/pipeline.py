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
    blur_threshold: float = 60.0,
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
    selected_paths: list[Path] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every_n_frames == 0:
                if _blur_score(frame) >= blur_threshold:
                    frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    selected_paths.append(frame_path)
            frame_index += 1
    finally:
        capture.release()

    return FrameExtractionResult(
        frame_paths=selected_paths,
        total_frames_seen=frame_index,
        selected_frame_count=len(selected_paths),
    )
