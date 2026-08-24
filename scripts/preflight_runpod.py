"""CPU-only gate that must pass before paid GPU startup."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


REQUIRED_FILES = (
    "Dockerfile",
    ".dockerignore",
    "bootstrap/versions.txt",
    "bootstrap/requirements-vggt.txt",
    "bootstrap/requirements-gsplat.txt",
    "bootstrap/start.sh",
    "bootstrap/verify_env.sh",
    "bootstrap/verify_cuda.py",
    "bootstrap/run_gpu_job.sh",
    "experiments/run_v3_vggt.py",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"PREFLIGHT ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()

    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        fail(f"required files missing: {', '.join(missing)}")

    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    forbidden = ("pip install gsplat\n", "apt-get upgrade", ":latest")
    bad = [token for token in forbidden if token in dockerfile]
    if bad:
        fail(f"Dockerfile contains unpinned/undesired patterns: {bad}")
    if not re.search(r"gsplat-1\.5\.3%2Bpt24cu124-cp310", dockerfile):
        fail("Dockerfile is not pinned to the precompiled gsplat wheel")

    report: dict[str, object] = {
        "project_root": str(root),
        "required_files": list(REQUIRED_FILES),
        "versions_sha256": sha256(root / "bootstrap/versions.txt"),
        "container": "pinned Python 3.10 / PyTorch 2.4.1 cu124 / gsplat 1.5.3 wheel",
        "cache_strategy": {
            "models": "/workspace/models",
            "huggingface": "/workspace/cache/huggingface",
            "torch_models": "/workspace/cache/torch",
            "torch_extensions": "/workspace/cache/torch_extensions",
            "pip": "/workspace/cache/pip",
        },
    }

    if args.scene_dir:
        scene = args.scene_dir.resolve()
        images_dir = scene / "images"
        if not images_dir.is_dir():
            fail(f"scene image directory missing: {images_dir}")
        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) < 2:
            fail(f"need at least two prepared images, found {len(images)}")
        unexpected = [p.name for p in images_dir.iterdir() if p.is_file() and p not in images]
        if unexpected:
            fail(f"non-image files in input directory: {unexpected}")
        report["input"] = {
            "scene_dir": str(scene),
            "images_dir": str(images_dir),
            "image_count": len(images),
            "images": [
                {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in images
            ],
        }

    if args.output_dir:
        output = args.output_dir.resolve()
        if output.exists() and any(output.iterdir()):
            report["resume_output"] = str(output)
            report["resume_required"] = True
        else:
            report["new_output"] = str(output)

    manifest_out = args.manifest_out or root / "bootstrap" / "preflight-report.json"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("LOCAL PREFLIGHT PASSED — SAFE TO START GPU")
    print(f"manifest: {manifest_out}")
    print("GPU-required task: VGGT inference, gsplat CUDA training/evaluation only")
    print("Prepared outside RunPod: container, pins, scripts, inputs, configuration, validation")
    print("Expected GPU runtime: set per experiment before launch")
    print("Expected GPU utilization: high during inference/training; stop after artifact flush")
    print(f"Input: {args.scene_dir.resolve() if args.scene_dir else 'not supplied (infrastructure-only preflight)'}")
    print(f"Output: {args.output_dir.resolve() if args.output_dir else 'not supplied'}")


if __name__ == "__main__":
    main()
