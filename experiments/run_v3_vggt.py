"""Run the reproducible V3 experiment: VGGT geometry -> official gsplat trainer.

This is intentionally an experiment runner, not the production worker.  It uses
the official repositories as external dependencies and retains every artifact
needed to audit a run instead of copying back only the final PLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")
QUALITY_PROFILES = {
    "baseline": {"data_factor": 2, "max_steps": 7_000, "pose_opt": False, "antialiased": False},
    # Per-camera appearance models are excluded: those corrections are not
    # represented in the portable PLY consumed by the browser viewer.
    "high": {"data_factor": 1, "max_steps": 30_000, "pose_opt": True, "antialiased": True},
}


def resolve_quality_profile(args: argparse.Namespace) -> dict[str, object]:
    config = dict(QUALITY_PROFILES[args.quality_profile])
    for name in ("data_factor", "max_steps", "pose_opt", "antialiased"):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
    return config


def image_manifest(images_dir: Path) -> list[dict[str, object]]:
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No input images found in {images_dir}")
    unexpected = sorted(path.name for path in images_dir.iterdir() if path.is_file() and path not in images)
    if unexpected:
        raise RuntimeError(
            f"The VGGT image directory must contain only images; found: {', '.join(unexpected)}"
        )
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in images
    ]


def run(command: list[str], cwd: Path) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def verify_colmap(scene_dir: Path) -> Path:
    candidates = [scene_dir / "sparse", scene_dir / "sparse" / "0"]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in COLMAP_FILES):
            return candidate
    raise RuntimeError(f"VGGT did not produce a complete COLMAP model under {scene_dir / 'sparse'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--gsplat-root", type=Path, required=True)
    parser.add_argument("--vggt-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gsplat-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage", choices=("geometry", "train", "all"), default="all")
    parser.add_argument("--quality-profile", choices=tuple(QUALITY_PROFILES), default="baseline")
    parser.add_argument("--use-ba", action="store_true")
    parser.add_argument("--query-frame-num", type=int, default=16)
    parser.add_argument("--max-query-pts", type=int, default=4096)
    parser.add_argument("--data-factor", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--pose-opt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--antialiased", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality = resolve_quality_profile(args)
    scene_dir = args.scene_dir.resolve()
    result_dir = args.result_dir.resolve()
    vggt_root = args.vggt_root.resolve()
    gsplat_root = args.gsplat_root.resolve()
    inputs = image_manifest(scene_dir / "images")
    result_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = result_dir / ".stages"
    stage_dir.mkdir(exist_ok=True)

    geometry_command = [str(args.vggt_python), "demo_colmap.py", f"--scene_dir={scene_dir}"]
    if args.use_ba:
        geometry_command.extend(
            [
                "--use_ba",
                f"--query_frame_num={args.query_frame_num}",
                f"--max_query_pts={args.max_query_pts}",
            ]
        )
    training_command = [
        str(args.gsplat_python),
        "examples/simple_trainer.py",
        "default",
        "--data-factor",
        str(quality["data_factor"]),
        "--data-dir",
        str(scene_dir),
        "--result-dir",
        str(result_dir),
        "--max-steps",
        str(quality["max_steps"]),
        "--save-ply",
        "--disable-viewer",
        "--disable-video",
    ]
    if quality["pose_opt"]:
        training_command.append("--pose-opt")
    if quality["antialiased"]:
        training_command.append("--antialiased")

    manifest = {
        "pipeline": "V3 VGGT -> COLMAP interchange -> gsplat",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "use_bundle_adjustment": args.use_ba,
        "quality_profile": args.quality_profile,
        "quality_config": quality,
        "input_count": len(inputs),
        "inputs": inputs,
        "geometry_command": geometry_command,
        "training_command": training_command,
        "status": "prepared",
    }
    manifest_path = result_dir / "v3_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.stage in {"geometry", "all"}:
        try:
            existing_sparse = verify_colmap(scene_dir)
        except RuntimeError:
            existing_sparse = None
        if existing_sparse and not args.force_geometry:
            print(f"Reusing verified VGGT geometry: {existing_sparse}")
        else:
            try:
                run(geometry_command, vggt_root)
            except Exception:
                (stage_dir / "geometry.failed").write_text(datetime.now(timezone.utc).isoformat())
                raise
    sparse_dir = verify_colmap(scene_dir)
    (stage_dir / "geometry.ok").write_text(datetime.now(timezone.utc).isoformat())
    manifest["colmap_sparse_dir"] = str(sparse_dir)
    manifest["status"] = "geometry_verified"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.stage in {"train", "all"}:
        completed_plys = sorted((result_dir / "ply").glob("point_cloud_*.ply"))
        if completed_plys and not args.force_training:
            print(f"Reusing completed training export: {completed_plys[-1]}")
        else:
            try:
                run(training_command, gsplat_root)
            except Exception:
                (stage_dir / "training.failed").write_text(datetime.now(timezone.utc).isoformat())
                raise
        completed_plys = sorted((result_dir / "ply").glob("point_cloud_*.ply"))
        if not completed_plys:
            raise RuntimeError("gsplat exited without a final PLY export")
        (stage_dir / "training.ok").write_text(datetime.now(timezone.utc).isoformat())
        manifest["final_ply"] = str(completed_plys[-1])
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
