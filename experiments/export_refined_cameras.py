"""Apply learned six-degree-of-freedom pose corrections to viewer camera JSON."""

import argparse
import json
from pathlib import Path

import numpy as np


CV_TO_THREE = np.diag([1.0, -1.0, -1.0, 1.0])


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-12:
        return np.eye(3) + skew(rotation_vector)
    axis = rotation_vector / angle
    cross = skew(axis)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.base.read_text(encoding="utf-8"))
    with np.load(args.state) as state:
        deltas = state["camera_pose_deltas"]

    camera_key = "camera_to_world_matrices" if "camera_to_world_matrices" in payload else "camera_poses"
    poses = payload[camera_key]
    if len(poses) != len(deltas):
        raise ValueError(f"Camera count mismatch: JSON has {len(poses)}, state has {len(deltas)}")

    refined = []
    for flattened, delta in zip(poses, deltas):
        camera_to_world_three = np.asarray(flattened).reshape(4, 4, order="F")
        world_to_camera = np.linalg.inv(camera_to_world_three @ CV_TO_THREE)

        correction = np.eye(4)
        correction[:3, :3] = rotation_matrix(delta[:3])
        correction[:3, 3] = delta[3:]
        refined_world_to_camera = correction @ world_to_camera
        refined_camera_to_world_three = np.linalg.inv(refined_world_to_camera) @ CV_TO_THREE
        refined.append(refined_camera_to_world_three.flatten(order="F").tolist())

    payload[camera_key] = refined
    payload["pose_source"] = "best_scene_camera_state.npz learned pose corrections"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(refined)} refined cameras to {args.output}")


if __name__ == "__main__":
    main()
