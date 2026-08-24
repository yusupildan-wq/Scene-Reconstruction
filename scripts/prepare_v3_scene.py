"""Prepare a deterministic, manifest-backed VGGT scene without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def select_evenly(paths: list[Path], count: int) -> list[Path]:
    if count >= len(paths):
        return paths
    indices = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-frames", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--copy", action="store_true", help="Copy instead of hard-linking files")
    args = parser.parse_args()

    source = args.source_frames.resolve()
    destination = args.scene_dir.resolve() / "images"
    frames = sorted(path for path in source.iterdir() if path.suffix.lower() in SUFFIXES)
    if len(frames) < 2:
        raise SystemExit(f"Need at least two source frames, found {len(frames)}")
    if args.count < 2:
        raise SystemExit("--count must be at least 2")
    selected = select_evenly(frames, args.count)

    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty scene: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    records = []
    for index, frame in enumerate(selected):
        target = destination / f"frame_{index:04d}{frame.suffix.lower()}"
        if args.copy:
            shutil.copy2(frame, target)
        else:
            try:
                os.link(frame, target)
            except OSError:
                shutil.copy2(frame, target)
        records.append(
            {
                "output": target.name,
                "source": str(frame),
                "bytes": target.stat().st_size,
                "sha256": digest(target),
            }
        )

    manifest = {
        "source_frames": str(source),
        "scene_dir": str(args.scene_dir.resolve()),
        "source_count": len(frames),
        "selected_count": len(selected),
        "selection": "deterministic_even_spacing_in_sorted_source_order",
        "images": records,
    }
    (args.scene_dir.resolve() / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(selected)} images in {destination}")


if __name__ == "__main__":
    main()
