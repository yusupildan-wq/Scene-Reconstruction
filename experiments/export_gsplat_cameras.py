"""Export gsplat's normalized training cameras for the browser viewer."""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsplat-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=2)
    args = parser.parse_args()

    sys.path.insert(0, str(args.gsplat_repo / "examples"))
    from datasets.colmap import Parser  # type: ignore

    scene = Parser(str(args.data_dir), factor=args.factor, normalize=True)
    payload = {
        "coordinate_convention": "opencv",
        "world_up": "z",
        "frames": [
            {
                "file_path": image_name,
                "transform_matrix": matrix.tolist(),
            }
            for image_name, matrix in zip(scene.image_names, scene.camtoworlds)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Exported {len(scene.camtoworlds)} normalized cameras to {args.output}")


if __name__ == "__main__":
    main()
