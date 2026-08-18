"""Independent, read-only audit of the full-room camera/image contract.

This module deliberately does not call the production scene loader, Gaussian
trainer, or DUSt3R inference.  It reads a completed geometry checkpoint
directly, uses DUSt3R's lightweight image loader as a preprocessing reference,
and writes all reports and visualizations to a separate output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageFont
from PIL.ImageOps import exif_transpose


DEFAULT_VIEWS = (0, 17, 18, 35, 36, 53, 54, 71, 72, 89, 90, 107, 108, 125, 126, 127)
FRAME_NUMBER = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)\.(?:jpe?g|png)$", re.IGNORECASE)


class AuditFailure(RuntimeError):
    """A structural invariant failed and the audit cannot be trusted."""


@dataclass(frozen=True)
class PixelTransform:
    crop_box: tuple[float, float, float, float]
    processed_size: tuple[int, int]
    final_size: tuple[int, int]
    scale_x: float
    scale_y: float


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_value), encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def parse_view_list(text: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("views must be comma-separated integers") from exc
    if not result or any(index < 0 for index in result):
        raise argparse.ArgumentTypeError("views must contain non-negative indices")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("views must be unique")
    return result


def parse_frame_number(name: str) -> int:
    match = FRAME_NUMBER.match(Path(name).name)
    if match is None:
        raise AuditFailure(f"Frame filename does not end in a numeric source index: {name}")
    return int(match.group("number"))


def assert_unique_chronology(frame_names: Sequence[str]) -> list[int]:
    numbers = [parse_frame_number(name) for name in frame_names]
    _require(len(numbers) == len(set(numbers)), "Source-frame chronology is not unique")
    _require(
        all(first < second for first, second in zip(numbers, numbers[1:])),
        "Source-frame chronology is not strictly increasing",
    )
    return numbers


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_pixel_transform(
    width: int,
    height: int,
    *,
    dust3r_size: int = 512,
    patch_size: int = 16,
    target_long_edge: int | None = None,
) -> PixelTransform:
    """Reconstruct the current crop contract without importing production helpers."""
    _require(width > 0 and height > 0, "Image dimensions must be positive")
    nominal_scale = dust3r_size / max(width, height)
    resized_width = round(width * nominal_scale)
    resized_height = round(height * nominal_scale)
    center_x, center_y = resized_width // 2, resized_height // 2
    half_width = ((2 * center_x) // patch_size) * patch_size / 2
    half_height = ((2 * center_y) // patch_size) * patch_size / 2
    if resized_width == resized_height:
        half_height = 3 * half_width / 4
    processed_size = (int(2 * half_width), int(2 * half_height))
    _require(min(processed_size) > 0, "Computed DUSt3R processed dimensions are empty")
    crop_resized = (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )
    crop_box = tuple(float(value / nominal_scale) for value in crop_resized)
    rounded_crop = tuple(round(value) for value in crop_box)
    crop_width = rounded_crop[2] - rounded_crop[0]
    crop_height = rounded_crop[3] - rounded_crop[1]
    _require(crop_width > 0 and crop_height > 0, "Rounded crop dimensions are empty")
    if target_long_edge is None:
        final_size = processed_size
    else:
        final_scale = min(1.0, target_long_edge / max(crop_width, crop_height))
        final_size = (round(crop_width * final_scale), round(crop_height * final_scale))
    return PixelTransform(
        crop_box=crop_box,
        processed_size=processed_size,
        final_size=final_size,
        scale_x=final_size[0] / processed_size[0],
        scale_y=final_size[1] / processed_size[1],
    )


def preprocess_current_path(
    path: Path, *, dust3r_size: int, target_long_edge: int | None
) -> tuple[np.ndarray, PixelTransform, dict[str, Any]]:
    with Image.open(path) as opened:
        raw_size = opened.size
        orientation = opened.getexif().get(ExifTags.Base.Orientation)
        oriented = exif_transpose(opened).convert("RGB")
    transform = independent_pixel_transform(
        *oriented.size,
        dust3r_size=dust3r_size,
        target_long_edge=target_long_edge,
    )
    cropped = oriented.crop(tuple(round(value) for value in transform.crop_box))
    resized = cropped.resize(transform.final_size, Image.Resampling.LANCZOS)
    metadata = {
        "raw_size": list(raw_size),
        "exif_orientation": orientation,
        "oriented_size": list(oriented.size),
    }
    return np.asarray(resized, dtype=np.float32) / 255.0, transform, metadata


def _tensor_to_rgb(value: Any, true_shape: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 4:
        _require(array.shape[0] == 1, f"Unexpected DUSt3R image batch shape {array.shape}")
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    _require(array.ndim == 3 and array.shape[2] == 3, f"Unexpected DUSt3R RGB shape {array.shape}")
    shape = np.asarray(true_shape).reshape(-1)
    _require(len(shape) >= 2, f"Invalid DUSt3R true_shape: {shape}")
    height, width = int(shape[-2]), int(shape[-1])
    array = array[:height, :width]
    # Official DUSt3R load_images uses ImgNorm, mapping RGB [0,1] to [-1,1].
    if float(np.nanmin(array)) < -0.01:
        array = (array + 1.0) / 2.0
    return np.clip(array.astype(np.float32), 0.0, 1.0)


def load_dust3r_reference(frame_paths: Sequence[Path], dust3r_size: int) -> list[dict[str, Any]]:
    try:
        from dust3r.utils.image import load_images
    except ImportError as exc:
        raise AuditFailure(
            "DUSt3R is unavailable. Set PYTHONPATH to the preserved DUSt3R checkout; "
            "the audit will not substitute its own preprocessing as the reference."
        ) from exc
    records = load_images([str(path) for path in frame_paths], size=dust3r_size)
    _require(len(records) == len(frame_paths), "DUSt3R loader returned the wrong image count")
    result = []
    for expected_index, (path, record) in enumerate(zip(frame_paths, records)):
        _require("img" in record and "true_shape" in record, "DUSt3R record lacks img/true_shape")
        reported_index = int(record.get("idx", expected_index))
        _require(
            reported_index == expected_index,
            f"DUSt3R image index mismatch at row {expected_index}: got {reported_index}",
        )
        instance = str(record.get("instance", ""))
        identity_evidence = "idx_only"
        if instance:
            # Official DUSt3R commonly sets instance=str(idx), while some forks
            # preserve a path. Validate whichever identity form is actually exposed.
            if instance.isdigit():
                _require(
                    int(instance) == expected_index,
                    f"DUSt3R numeric instance mismatch at row {expected_index}: {instance}",
                )
                identity_evidence = "numeric_instance"
            else:
                _require(
                    Path(instance).name == path.name,
                    f"DUSt3R image identity mismatch at row {expected_index}: {instance} != {path.name}",
                )
                identity_evidence = "path_instance"
        rgb = _tensor_to_rgb(record["img"], record["true_shape"])
        result.append(
            {
                "index": reported_index,
                "instance": instance,
                "identity_evidence": identity_evidence,
                "true_shape": list(rgb.shape[:2]),
                "rgb": rgb,
            }
        )
    return result


def validate_intrinsics(K: np.ndarray, width: int, height: int, index: int) -> None:
    _require(K.shape == (3, 3), f"View {index}: intrinsics must be 3x3, got {K.shape}")
    _require(np.isfinite(K).all(), f"View {index}: intrinsics contain non-finite values")
    _require(K[0, 0] > 0 and K[1, 1] > 0, f"View {index}: focal lengths must be positive")
    _require(abs(float(K[2, 2]) - 1.0) < 1e-4, f"View {index}: K[2,2] must be 1")
    _require(abs(float(K[2, 0])) < 1e-5 and abs(float(K[2, 1])) < 1e-5, f"View {index}: invalid K last row")
    _require(abs(float(K[0, 1])) < 1e-3 and abs(float(K[1, 0])) < 1e-3, f"View {index}: unsupported intrinsics skew")
    _require(0 <= K[0, 2] < width and 0 <= K[1, 2] < height, f"View {index}: principal point is out of bounds")


def validate_pose(pose: np.ndarray, index: int) -> dict[str, float]:
    _require(pose.shape == (4, 4), f"View {index}: pose must be 4x4, got {pose.shape}")
    _require(np.isfinite(pose).all(), f"View {index}: pose contains non-finite values")
    _require(np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5), f"View {index}: invalid homogeneous pose row")
    rotation = pose[:3, :3]
    determinant = float(np.linalg.det(rotation))
    orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    _require(abs(determinant - 1.0) < 1e-3, f"View {index}: rotation determinant is {determinant:.6g}, not +1")
    _require(orthogonality < 1e-3, f"View {index}: rotation is not orthonormal ({orthogonality:.6g})")
    inverse = np.linalg.inv(pose)
    _require(np.allclose(inverse @ pose, np.eye(4), atol=1e-4), f"View {index}: pose inverse is inconsistent")
    return {"rotation_determinant": determinant, "rotation_orthogonality_error": orthogonality}


def convention_score(
    points: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    viewmat: np.ndarray,
    *,
    pixel_offset: float = 0.0,
    sample_limit: int = 50_000,
) -> dict[str, float | None]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) > sample_limit:
        chosen = np.linspace(0, len(xs) - 1, sample_limit, dtype=int)
        ys, xs = ys[chosen], xs[chosen]
    world = points[ys, xs]
    homogeneous = np.concatenate((world, np.ones((len(world), 1), dtype=world.dtype)), axis=1)
    camera = (viewmat @ homogeneous.T).T[:, :3]
    finite = np.isfinite(camera).all(axis=1)
    positive = finite & (camera[:, 2] > 1e-6)
    positive_ratio = float(positive.mean()) if len(positive) else 0.0
    if not positive.any():
        return {"positive_depth_ratio": positive_ratio, "median_pixels": None, "p95_pixels": None}
    projected = (K @ camera[positive].T).T
    projected = projected[:, :2] / projected[:, 2:3]
    expected = np.stack((xs[positive] + pixel_offset, ys[positive] + pixel_offset), axis=1)
    errors = np.linalg.norm(projected - expected, axis=1)
    return {
        "positive_depth_ratio": positive_ratio,
        "median_pixels": float(np.median(errors)),
        "p95_pixels": float(np.percentile(errors, 95)),
    }


def validate_pose_convention(
    points: np.ndarray, mask: np.ndarray, K: np.ndarray, pose: np.ndarray, index: int
) -> dict[str, Any]:
    hypotheses = {
        "stored_c2w_inverted_integer_center": convention_score(points, mask, K, np.linalg.inv(pose), pixel_offset=0.0),
        "stored_c2w_inverted_half_center": convention_score(points, mask, K, np.linalg.inv(pose), pixel_offset=0.5),
        "stored_w2c_direct_integer_center": convention_score(points, mask, K, pose, pixel_offset=0.0),
    }
    expected = hypotheses["stored_c2w_inverted_integer_center"]
    possible = (
        expected["positive_depth_ratio"] >= 0.5
        and expected["median_pixels"] is not None
        and expected["median_pixels"] <= 2.0
        and expected["p95_pixels"] <= 10.0
    )
    _require(possible, f"View {index}: saved pose cannot satisfy the documented C2W convention")
    return hypotheses


def compare_rgb(reference: np.ndarray, candidate: np.ndarray, index: int) -> dict[str, float]:
    _require(reference.shape == candidate.shape, f"View {index}: DUSt3R/current RGB shapes differ: {reference.shape} vs {candidate.shape}")
    difference = np.abs(reference - candidate)
    reference_u8 = (reference * 255).astype(np.uint8)
    candidate_u8 = (candidate * 255).astype(np.uint8)
    ref_edges = cv2.Canny(cv2.cvtColor(reference_u8, cv2.COLOR_RGB2GRAY), 80, 160)
    candidate_edges = cv2.Canny(cv2.cvtColor(candidate_u8, cv2.COLOR_RGB2GRAY), 80, 160)
    edge_disagreement = float(np.mean((ref_edges > 0) != (candidate_edges > 0)))
    return {
        "mae": float(difference.mean()),
        "p95_absolute_error": float(np.percentile(difference, 95)),
        "maximum_absolute_error": float(difference.max()),
        "edge_disagreement_fraction": edge_disagreement,
    }


def _normalize_depth(points: np.ndarray, mask: np.ndarray, viewmat: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points.reshape(-1, 3), np.ones((points.size // 3, 1), dtype=points.dtype)), axis=1)
    depth = (viewmat @ homogeneous.T).T[:, 2].reshape(mask.shape)
    valid = mask & np.isfinite(depth) & (depth > 0)
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], [2, 98])
        normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[valid] = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)[valid]
    image[~mask] = (255, 0, 255)
    return image


def _project_overlay(
    base_rgb: np.ndarray,
    world_points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    viewmat: np.ndarray,
    *,
    sample_limit: int = 30_000,
) -> np.ndarray:
    height, width = base_rgb.shape[:2]
    if len(world_points) > sample_limit:
        chosen = np.linspace(0, len(world_points) - 1, sample_limit, dtype=int)
        world_points, colors = world_points[chosen], colors[chosen]
    homogeneous = np.concatenate((world_points, np.ones((len(world_points), 1), dtype=world_points.dtype)), axis=1)
    camera = (viewmat @ homogeneous.T).T[:, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    projected = (K @ camera[valid].T).T
    xy = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
    colors = colors[valid]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    overlay = (np.clip(base_rgb, 0, 1) * 255).astype(np.uint8).copy()
    xy, colors = xy[inside], colors[inside]
    overlay[xy[:, 1], xy[:, 0]] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    return overlay


def _edge_overlay(rgb: np.ndarray, depth_rgb: np.ndarray) -> np.ndarray:
    base = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgb_edges = cv2.Canny(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    depth_edges = cv2.Canny(cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2GRAY), 60, 120) > 0
    result = base.copy()
    result[rgb_edges] = (0, 255, 0)
    result[depth_edges] = (255, 0, 0)
    result[rgb_edges & depth_edges] = (255, 255, 0)
    return result


def _panel(image: np.ndarray, title: str, size: tuple[int, int] = (480, 320)) -> Image.Image:
    source = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    source.thumbnail((size[0], size[1] - 30), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "white")
    panel.paste(source, ((size[0] - source.width) // 2, 28 + (size[1] - 30 - source.height) // 2))
    ImageDraw.Draw(panel).text((8, 7), title, fill="black", font=ImageFont.load_default())
    return panel


def save_montage(
    output_path: Path,
    *,
    raw: np.ndarray,
    dust3r_rgb: np.ndarray,
    current_rgb: np.ndarray,
    mask: np.ndarray,
    depth_rgb: np.ndarray,
    same_projection: np.ndarray,
    neighbor_projection: np.ndarray,
    details: Sequence[str],
) -> None:
    difference = np.abs(dust3r_rgb - current_rgb).mean(axis=2)
    heat = cv2.applyColorMap((np.clip(difference * 8, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    mask_overlay = (current_rgb * 255).astype(np.uint8).copy()
    mask_overlay[~mask] = (255, 0, 255)
    edge_overlay = _edge_overlay(current_rgb, depth_rgb)
    panels = [
        _panel(raw, "Raw oriented JPEG"),
        _panel((dust3r_rgb * 255).astype(np.uint8), "DUSt3R load_images RGB"),
        _panel((current_rgb * 255).astype(np.uint8), "Independent current-path RGB"),
        _panel(heat, "Absolute RGB difference (8x)"),
        _panel(mask_overlay, "Mask overlay"),
        _panel(depth_rgb, "Robust camera depth"),
        _panel(same_projection, "Same-view point projection"),
        _panel(neighbor_projection, "Neighbor-to-current projection"),
        _panel(edge_overlay, "RGB edges green / depth edges red"),
    ]
    text_panel = Image.new("RGB", (480, 320), "white")
    draw = ImageDraw.Draw(text_panel)
    y = 8
    for line in details:
        draw.text((8, y), line[:78], fill="black", font=ImageFont.load_default())
        y += 15
    panels.append(text_panel)
    sheet = Image.new("RGB", (480 * 2, 320 * 5), "white")
    for position, panel in enumerate(panels):
        sheet.paste(panel, ((position % 2) * 480, (position // 2) * 320))
    sheet.save(output_path, quality=94)


def _load_manifest_frames(graph_path: Path) -> list[str]:
    _require(graph_path.exists(), f"Missing pair graph: {graph_path}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    _require(isinstance(graph.get("frames"), list), "Pair graph has no frames list")
    return [Path(value).name for value in graph["frames"]]


def _indexed_files(directory: Path, prefix: str) -> list[Path]:
    return sorted(directory.glob(f"{prefix}_*.npy"))


def _assert_indexed_files(paths: Sequence[Path], prefix: str, count: int) -> None:
    expected = [f"{prefix}_{index:04d}.npy" for index in range(count)]
    actual = [path.name for path in paths]
    _require(actual == expected, f"{prefix} files are missing, duplicated, stale, or non-contiguous")


def audit_registration(
    run_dir: Path,
    output_dir: Path,
    *,
    selected_views: Sequence[int] = DEFAULT_VIEWS,
    dust3r_size: int = 512,
    target_long_edge: int = 1440,
) -> dict[str, Any]:
    _require(run_dir.resolve() != output_dir.resolve(), "Audit output directory must differ from the reconstruction directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = run_dir / "geometry"
    frames_dir = run_dir / "frames"
    marker_path = geometry / "COMPLETE.json"
    cameras_path = geometry / "cameras.npz"
    _require(marker_path.exists(), f"Missing completion marker: {marker_path}")
    _require(cameras_path.exists(), f"Missing camera archive: {cameras_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    with np.load(cameras_path) as camera_data:
        for key in ("poses", "intrinsics", "frame_names"):
            _require(key in camera_data, f"Camera archive is missing {key}")
        poses = camera_data["poses"].astype(np.float64)
        intrinsics = camera_data["intrinsics"].astype(np.float64)
        frame_names = [str(value) for value in camera_data["frame_names"]]
    count = len(frame_names)
    _require(count > 0, "Camera archive contains no views")
    _require(len(poses) == count and len(intrinsics) == count, "Camera array counts do not match frame_names")
    _require(int(marker.get("views", -1)) == count, "Completion-marker view count does not match cameras")
    _require(int(marker.get("point_maps", -1)) == count, "Completion-marker point-map count does not match cameras")
    frame_paths = [frames_dir / name for name in frame_names]
    _require(all(path.is_file() for path in frame_paths), "One or more camera frame files are missing")
    directory_names = [path.name for path in sorted(frames_dir.glob("*.jpg"))]
    _require(directory_names == frame_names, "Sorted frame directory identities/order do not match cameras.npz")
    graph_names = _load_manifest_frames(run_dir / "pair_graph.json")
    _require(graph_names == frame_names, "Pair-graph frame identities/order do not match cameras.npz")
    chronology = assert_unique_chronology(frame_names)
    point_files = _indexed_files(geometry, "points")
    mask_files = _indexed_files(geometry, "mask")
    _require(len(point_files) == count and len(mask_files) == count, "Point/mask counts do not match camera count")
    _assert_indexed_files(point_files, "points", count)
    _assert_indexed_files(mask_files, "mask", count)
    _require(all(0 <= index < count for index in selected_views), "A requested representative view is out of range")

    dust3r_records = load_dust3r_reference(frame_paths, dust3r_size)
    report: dict[str, Any] = {
        "status": "running",
        "run_directory": str(run_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "view_count": count,
        "selected_views": list(selected_views),
        "dust3r_size": dust3r_size,
        "target_long_edge": target_long_edge,
        "global_assertions": {
            "counts_match": True,
            "frame_identities_match": True,
            "chronology_unique_and_increasing": True,
            "indexed_geometry_files_contiguous": True,
        },
        "views": [],
    }

    # Validate every view structurally; only representative views get expensive imagery.
    cached: dict[int, dict[str, Any]] = {}
    for index in range(count):
        points = np.load(point_files[index], mmap_mode="r")
        mask = np.load(mask_files[index], mmap_mode="r")
        _require(points.ndim == 3 and points.shape[2] == 3, f"View {index}: point map must be HxWx3")
        _require(mask.ndim == 2, f"View {index}: mask must be HxW")
        _require(points.shape[:2] == mask.shape, f"View {index}: point/mask shapes differ")
        _require(mask.dtype == np.bool_, f"View {index}: mask dtype must be bool, got {mask.dtype}")
        _require(mask.any(), f"View {index}: mask contains no valid pixels")
        validate_intrinsics(intrinsics[index], mask.shape[1], mask.shape[0], index)
        pose_validation = validate_pose(poses[index], index)
        current_rgb, transform, image_metadata = preprocess_current_path(
            frame_paths[index], dust3r_size=dust3r_size, target_long_edge=None
        )
        reference_rgb = dust3r_records[index]["rgb"]
        _require(current_rgb.shape[:2] == mask.shape, f"View {index}: current RGB/point/mask shapes differ")
        _require(reference_rgb.shape[:2] == mask.shape, f"View {index}: DUSt3R RGB/point/mask shapes differ")
        comparison = compare_rgb(reference_rgb, current_rgb, index)
        convention = validate_pose_convention(points, mask, intrinsics[index], poses[index], index)
        training_transform = independent_pixel_transform(
            image_metadata["oriented_size"][0],
            image_metadata["oriented_size"][1],
            dust3r_size=dust3r_size,
            target_long_edge=target_long_edge,
        )
        scaled_K = intrinsics[index].copy()
        scaled_K[0, (0, 2)] *= training_transform.scale_x
        scaled_K[1, (1, 2)] *= training_transform.scale_y
        validate_intrinsics(scaled_K, *training_transform.final_size, index)
        view_report = {
            "original_view_index": index,
            "frame_name": frame_names[index],
            "source_frame_number": chronology[index],
            "file_size": frame_paths[index].stat().st_size,
            "sha256": file_sha256(frame_paths[index]),
            **image_metadata,
            "dust3r_index": dust3r_records[index]["index"],
            "dust3r_instance": dust3r_records[index]["instance"],
            "dust3r_identity_evidence": dust3r_records[index]["identity_evidence"],
            "dust3r_true_shape": dust3r_records[index]["true_shape"],
            "point_shape": list(points.shape),
            "mask_shape": list(mask.shape),
            "valid_mask_fraction": float(mask.mean()),
            "crop_box_original": list(transform.crop_box),
            "processed_size": list(transform.processed_size),
            "training_final_size": list(training_transform.final_size),
            "training_scale_x": training_transform.scale_x,
            "training_scale_y": training_transform.scale_y,
            "intrinsics": intrinsics[index].tolist(),
            "scaled_training_intrinsics": scaled_K.tolist(),
            "pose": poses[index].tolist(),
            "pose_validation": pose_validation,
            "pose_hypotheses": convention,
            "preprocessing_comparison": comparison,
            "assertions": {
                "identities_match": True,
                "rgb_point_mask_shapes_match": True,
                "intrinsics_valid": True,
                "pose_valid": True,
                "documented_pose_convention_possible": True,
            },
        }
        report["views"].append(view_report)
        if index in selected_views or index - 1 in selected_views or index + 1 in selected_views:
            cached[index] = {
                "points": np.asarray(points),
                "mask": np.asarray(mask),
                "current_rgb": current_rgb,
                "reference_rgb": reference_rgb,
                "scaled_K": scaled_K,
                "training_transform": training_transform,
            }

    for index in selected_views:
        item = cached[index]
        training_rgb, _, _ = preprocess_current_path(
            frame_paths[index], dust3r_size=dust3r_size, target_long_edge=target_long_edge
        )
        geometry_rgb = item["current_rgb"]
        points, mask = item["points"], item["mask"]
        viewmat = np.linalg.inv(poses[index])
        colors = geometry_rgb[mask]
        same_projection = _project_overlay(geometry_rgb, points[mask], colors, intrinsics[index], viewmat)
        neighbor_index = index - 1 if index > 0 else index + 1
        neighbor = cached[neighbor_index]
        neighbor_projection = _project_overlay(
            geometry_rgb,
            neighbor["points"][neighbor["mask"]],
            neighbor["current_rgb"][neighbor["mask"]],
            intrinsics[index],
            viewmat,
        )
        depth_rgb = _normalize_depth(points, mask, viewmat)
        with Image.open(frame_paths[index]) as opened:
            raw = np.asarray(exif_transpose(opened).convert("RGB"))
        view_report = report["views"][index]
        details = [
            f"view={index} frame={frame_names[index]} source={chronology[index]}",
            f"raw/oriented={view_report['raw_size']}/{view_report['oriented_size']}",
            f"processed={view_report['processed_size']} training={view_report['training_final_size']}",
            f"crop={','.join(f'{v:.3f}' for v in view_report['crop_box_original'])}",
            f"scale=({view_report['training_scale_x']:.6f},{view_report['training_scale_y']:.6f})",
            f"mask_valid={view_report['valid_mask_fraction']:.4f}",
            f"rgb_mae={view_report['preprocessing_comparison']['mae']:.6f}",
            f"rgb_p95={view_report['preprocessing_comparison']['p95_absolute_error']:.6f}",
            f"reproj={view_report['pose_hypotheses']['stored_c2w_inverted_integer_center']}",
            f"K={np.array2string(intrinsics[index], precision=2, separator=',')}",
            "assertions=PASS",
        ]
        save_montage(
            output_dir / f"montage_view_{index:03d}.jpg",
            raw=raw,
            dust3r_rgb=item["reference_rgb"],
            current_rgb=geometry_rgb,
            mask=mask,
            depth_rgb=depth_rgb,
            same_projection=same_projection,
            neighbor_projection=neighbor_projection,
            details=details,
        )

    # A compact identity strip makes discontinuities at requested boundaries obvious.
    strip_images = []
    for index in selected_views:
        with Image.open(frame_paths[index]) as opened:
            rgb = np.asarray(exif_transpose(opened).convert("RGB"))
        strip_images.append(_panel(rgb, f"{index}: {frame_names[index]}", size=(240, 170)))
    columns = 4
    rows = math.ceil(len(strip_images) / columns)
    strip = Image.new("RGB", (columns * 240, rows * 170), "white")
    for position, panel in enumerate(strip_images):
        strip.paste(panel, ((position % columns) * 240, (position // columns) * 170))
    strip.save(output_dir / "chronology_boundaries.jpg", quality=94)

    report["status"] = "passed"
    summary = {
        "status": "passed",
        "view_count": count,
        "selected_views": list(selected_views),
        "maximum_preprocessing_mae": max(item["preprocessing_comparison"]["mae"] for item in report["views"]),
        "maximum_reprojection_median_pixels": max(
            item["pose_hypotheses"]["stored_c2w_inverted_integer_center"]["median_pixels"]
            for item in report["views"]
        ),
        "report": "registration_report.json",
    }
    _write_json(output_dir / "registration_report.json", report)
    _write_json(output_dir / "registration_summary.json", summary)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", type=parse_view_list, default=list(DEFAULT_VIEWS))
    parser.add_argument("--dust3r-size", type=int, default=512)
    parser.add_argument("--target-long-edge", type=int, default=1440)
    args = parser.parse_args(argv)
    try:
        audit_registration(
            args.run_dir,
            args.output_dir,
            selected_views=args.views,
            dust3r_size=args.dust3r_size,
            target_long_edge=args.target_long_edge,
        )
    except Exception as exc:  # Write a clear failure artifact and return non-zero.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output_dir / "registration_failure.json", failure)
        print(f"REGISTRATION AUDIT FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"Registration audit passed. Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
