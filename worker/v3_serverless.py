"""RunPod Serverless adapter for the existing V3 VGGT -> gsplat pipeline.

The reconstruction implementation remains in experiments/run_v3_vggt.py. This
module only moves artifacts across presigned URLs, reports product progress, and
packages reusable geometry so a training failure does not rerun VGGT.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Callable

import requests

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/project"))
VGGT_ROOT = Path(os.getenv("VGGT_ROOT", "/opt/vggt"))
GSPLAT_ROOT = Path(os.getenv("GSPLAT_ROOT", "/opt/gsplat"))
VGGT_PYTHON = Path(os.getenv("VGGT_PYTHON", "/opt/venvs/vggt/bin/python"))
GSPLAT_PYTHON = Path(os.getenv("GSPLAT_PYTHON", "/opt/venvs/gsplat/bin/python"))
DOWNLOAD_TIMEOUT = (20, 300)
UPLOAD_TIMEOUT = (20, 1800)


def _require_string(payload: dict, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required input: {name}")
    return value


def validate_payload(payload: dict) -> dict:
    frame_urls = payload.get("frame_urls")
    if not isinstance(frame_urls, list) or len(frame_urls) < 2 or not all(
        isinstance(url, str) and url for url in frame_urls
    ):
        raise ValueError("frame_urls must contain at least two presigned URLs")
    quality = payload.get("quality_profile", "high")
    if quality not in {"baseline", "high"}:
        raise ValueError("quality_profile must be baseline or high")
    return {
        "job_id": _require_string(payload, "job_id"),
        "frame_urls": frame_urls,
        "scene_upload_url": _require_string(payload, "scene_upload_url"),
        "cameras_upload_url": _require_string(payload, "cameras_upload_url"),
        "geometry_upload_url": _require_string(payload, "geometry_upload_url"),
        "geometry_download_url": payload.get("geometry_download_url"),
        "quality_profile": quality,
    }


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with destination.open("wb") as target:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    target.write(chunk)


def _upload(url: str, source: Path, content_type: str) -> None:
    with source.open("rb") as body:
        response = requests.put(
            url,
            data=body,
            headers={"Content-Type": content_type},
            timeout=UPLOAD_TIMEOUT,
        )
    response.raise_for_status()


def _run(command: list[str], cwd: Path) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _geometry_is_complete(scene_dir: Path) -> bool:
    candidates = (scene_dir / "sparse", scene_dir / "sparse" / "0")
    required = ("cameras.bin", "images.bin", "points3D.bin")
    return any(all((candidate / name).is_file() for name in required) for candidate in candidates)


def _pack_geometry(scene_dir: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(scene_dir / "sparse", arcname="sparse")


def _restore_geometry(url: str | None, scene_dir: Path) -> bool:
    if not url:
        return False
    archive = scene_dir.parent / "geometry.tar.gz"
    try:
        _download(url, archive)
        with tarfile.open(archive, "r:gz") as bundle:
            destination = scene_dir.resolve()
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if destination not in target.parents and target != destination:
                    raise tarfile.TarError("Geometry archive contains an unsafe path")
            bundle.extractall(scene_dir)
        return _geometry_is_complete(scene_dir)
    except (requests.RequestException, tarfile.TarError, OSError):
        shutil.rmtree(scene_dir / "sparse", ignore_errors=True)
        return False


def _latest_ply(result_dir: Path) -> Path:
    exports = list((result_dir / "ply").glob("point_cloud_*.ply"))
    if not exports:
        raise RuntimeError("V3 training completed without a PLY export")
    return max(exports, key=lambda path: int(path.stem.rsplit("_", 1)[1]))


def _metrics(result_dir: Path) -> dict:
    candidates = list((result_dir / "stats").glob("val_step*.json"))
    if not candidates:
        return {}
    latest = max(candidates, key=lambda path: int(path.stem.removeprefix("val_step")))
    return json.loads(latest.read_text(encoding="utf-8"))


def reconstruct(payload: dict, progress: Callable[[str, int, str], None]) -> dict:
    request = validate_payload(payload)
    with tempfile.TemporaryDirectory(prefix=f"scene-{request['job_id']}-") as directory:
        root = Path(directory)
        scene_dir = root / "scene"
        images_dir = scene_dir / "images"
        result_dir = root / "result"
        images_dir.mkdir(parents=True)

        progress("preparing_frames", 28, "Downloading prepared frames")
        for index, url in enumerate(request["frame_urls"]):
            _download(url, images_dir / f"frame_{index:06d}.jpg")

        geometry_reused = _restore_geometry(request["geometry_download_url"], scene_dir)
        if geometry_reused:
            progress("vggt_geometry", 52, "Reusing verified VGGT geometry")
        else:
            progress("vggt_geometry", 35, "Estimating cameras and room geometry")
            _run(
                [
                    str(VGGT_PYTHON),
                    str(PROJECT_ROOT / "experiments" / "run_v3_vggt.py"),
                    "--scene-dir", str(scene_dir),
                    "--result-dir", str(result_dir),
                    "--vggt-root", str(VGGT_ROOT),
                    "--gsplat-root", str(GSPLAT_ROOT),
                    "--vggt-python", str(VGGT_PYTHON),
                    "--gsplat-python", str(GSPLAT_PYTHON),
                    "--stage", "geometry",
                    "--quality-profile", request["quality_profile"],
                ],
                PROJECT_ROOT,
            )
            archive = root / "geometry.tar.gz"
            _pack_geometry(scene_dir, archive)
            _upload(request["geometry_upload_url"], archive, "application/gzip")
            progress("vggt_geometry", 55, "VGGT geometry saved")

        progress("gaussian_optimization", 58, "Optimizing Gaussian appearance")
        _run(
            [
                str(GSPLAT_PYTHON),
                str(PROJECT_ROOT / "experiments" / "run_v3_vggt.py"),
                "--scene-dir", str(scene_dir),
                "--result-dir", str(result_dir),
                "--vggt-root", str(VGGT_ROOT),
                "--gsplat-root", str(GSPLAT_ROOT),
                "--vggt-python", str(VGGT_PYTHON),
                "--gsplat-python", str(GSPLAT_PYTHON),
                "--stage", "train",
                "--quality-profile", request["quality_profile"],
            ],
            PROJECT_ROOT,
        )

        progress("finalizing", 92, "Packaging viewer artifacts")
        cameras_path = result_dir / "scene_cameras.json"
        factor = "1" if request["quality_profile"] == "high" else "2"
        _run(
            [
                str(GSPLAT_PYTHON),
                str(PROJECT_ROOT / "experiments" / "export_gsplat_cameras.py"),
                "--gsplat-repo", str(GSPLAT_ROOT),
                "--data-dir", str(scene_dir),
                "--output", str(cameras_path),
                "--factor", factor,
            ],
            PROJECT_ROOT,
        )
        final_ply = _latest_ply(result_dir)
        _upload(request["scene_upload_url"], final_ply, "application/octet-stream")
        _upload(request["cameras_upload_url"], cameras_path, "application/json")
        return {
            "stage": "complete",
            "progress": 100,
            "geometry_reused": geometry_reused,
            "num_frames": len(request["frame_urls"]),
            "metrics": _metrics(result_dir),
        }


def handler(job: dict) -> dict:
    import runpod

    def report(stage: str, progress: int, detail: str) -> None:
        runpod.serverless.progress_update(
            job,
            json.dumps({"stage": stage, "progress": progress, "detail": detail}),
        )

    return reconstruct(job.get("input") or {}, report)


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
