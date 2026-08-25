"""Execute the canonical V3 pipeline in a prepared workspace.

Both local NVIDIA and temporary RunPod executors call this file. It deliberately
contains no provider logic: it only invokes the existing VGGT/gsplat runner and
normalizes its outputs for the application.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path


def emit(stage: str, progress: int, detail: str) -> None:
    print("SCENE_PROGRESS " + json.dumps({"stage": stage, "progress": progress, "detail": detail}), flush=True)


def run(command: list[str], cwd: Path) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def latest_numeric(directory: Path, pattern: str) -> Path:
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise RuntimeError(f"No output matched {directory / pattern}")
    return max(candidates, key=lambda path: int(path.stem.rsplit("_", 1)[1].removeprefix("step")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--gsplat-root", type=Path, required=True)
    parser.add_argument("--vggt-python", type=Path, required=True)
    parser.add_argument("--gsplat-python", type=Path, required=True)
    parser.add_argument("--quality-profile", choices=("baseline", "high"), default="high")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    scene_dir = args.scene_dir.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    base_command = [
        str(args.vggt_python), str(project_root / "experiments" / "run_v3_vggt.py"),
        "--scene-dir", str(scene_dir), "--result-dir", str(result_dir),
        "--vggt-root", str(args.vggt_root), "--gsplat-root", str(args.gsplat_root),
        "--vggt-python", str(args.vggt_python), "--gsplat-python", str(args.gsplat_python),
        "--quality-profile", args.quality_profile,
    ]
    pipeline_timings = {}
    geometry_started = time.monotonic()
    emit("vggt_geometry", 35, "Estimating cameras and room geometry")
    run(base_command + ["--stage", "geometry"], project_root)
    pipeline_timings["vggt_seconds"] = round(time.monotonic() - geometry_started, 3)

    gsplat_started = time.monotonic()
    emit("gaussian_optimization", 58, "Optimizing Gaussian appearance")
    run(base_command + ["--stage", "train"], project_root)
    pipeline_timings["gsplat_seconds"] = round(time.monotonic() - gsplat_started, 3)
    emit("gaussian_optimization", 88, "Gaussian optimization complete")
    cameras = result_dir / "scene_cameras.json"
    run([
        str(args.gsplat_python), str(project_root / "experiments" / "export_gsplat_cameras.py"),
        "--gsplat-repo", str(args.gsplat_root), "--data-dir", str(scene_dir),
        "--output", str(cameras), "--factor", "1" if args.quality_profile == "high" else "2",
    ], project_root)

    emit("finalizing", 94, "Packaging local viewer artifacts")
    final_ply = latest_numeric(result_dir / "ply", "point_cloud_*.ply")
    shutil.copy2(final_ply, result_dir / "scene.ply")
    with tarfile.open(result_dir / "vggt_geometry.tar.gz", "w:gz") as archive:
        archive.add(scene_dir / "sparse", arcname="sparse")
    stats = list((result_dir / "stats").glob("val_step*.json"))
    metrics = json.loads(max(stats, key=lambda p: int(p.stem.removeprefix("val_step"))).read_text()) if stats else {}
    metrics["timings"] = pipeline_timings
    (result_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    emit("finalizing", 98, "Artifacts ready for retrieval")


if __name__ == "__main__":
    main()
