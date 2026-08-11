"""Renders a small synthetic multi-view scene with EXACTLY known camera poses.

Purpose: give COLMAP something to reconstruct where we already know the right
answer, so we can check its estimated poses against ground truth instead of just
eyeballing the result. Not meant to look like a real room -- it's a controlled
test case, the same way you'd use a known input to unit-test a function.
"""

import json
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).parent / "data" / "synthetic_scene"
IMAGES_DIR = OUT_DIR / "images"
IMG_W, IMG_H = 640, 480
FOCAL = 500.0
CX, CY = IMG_W / 2, IMG_H / 2
N_POINTS = 400
N_CAMERAS = 16
ORBIT_RADIUS = 4.0
ORBIT_HEIGHT = 0.4

rng = np.random.default_rng(42)


def look_at_rotation(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera rotation for a camera at camera_pos looking at target.

    Builds an orthonormal camera basis (right, up, forward) and returns it as the
    rotation matrix that expresses world-frame vectors in the camera's frame.
    """
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # Camera looks down +Z in this convention; rows are the camera's basis vectors
    # expressed in world coordinates, which is exactly the world->camera rotation.
    return np.stack([right, -true_up, forward], axis=0)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # A "room" of colored point-markers scattered through a 3D box, not on a
    # single plane -- a fully planar scene is a degenerate case for SfM's
    # essential-matrix step, so we deliberately spread points across depth.
    points_3d = rng.uniform(low=[-2.0, -1.5, -2.0], high=[2.0, 1.5, 2.0], size=(N_POINTS, 3))
    # SIFT (COLMAP's feature detector) works on grayscale intensity GRADIENTS, not
    # color -- a flat-filled colored circle has almost no internal gradient
    # structure and looks nearly identical to same-sized circles of a different
    # hue once converted to grayscale. Give each point a real fixed noise texture
    # patch instead, pasted identically wherever it's visible, so it actually has
    # distinctive local structure to match on -- mimicking a real textured surface.
    patch_size = 14
    textures = rng.integers(30, 255, size=(N_POINTS, patch_size, patch_size, 3), dtype=np.uint8)

    K = np.array([[FOCAL, 0, CX], [0, FOCAL, CY], [0, 0, 1]])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    cameras = []
    for i in range(N_CAMERAS):
        angle = 2 * np.pi * i / N_CAMERAS
        cam_pos = np.array(
            [ORBIT_RADIUS * np.cos(angle), ORBIT_HEIGHT, ORBIT_RADIUS * np.sin(angle)]
        )
        R = look_at_rotation(cam_pos, target, up)
        t = -R @ cam_pos  # standard convention: X_cam = R @ X_world + t

        image = np.full((IMG_H, IMG_W, 3), 20, dtype=np.uint8)  # near-black background
        points_cam = (R @ points_3d.T).T + t
        in_front = points_cam[:, 2] > 0.1
        proj = (K @ points_cam[in_front].T).T
        proj_2d = proj[:, :2] / proj[:, 2:3]

        half = patch_size // 2
        for (x, y), texture in zip(proj_2d, textures[in_front]):
            xi, yi = int(x), int(y)
            x0, y0 = xi - half, yi - half
            x1, y1 = x0 + patch_size, y0 + patch_size
            if x0 < 0 or y0 < 0 or x1 > IMG_W or y1 > IMG_H:
                continue
            image[y0:y1, x0:x1] = texture

        filename = f"cam_{i:02d}.jpg"
        cv2.imwrite(str(IMAGES_DIR / filename), image)
        cameras.append(
            {
                "filename": filename,
                "position_world": cam_pos.tolist(),
                "rotation_world_to_cam": R.tolist(),
                "translation": t.tolist(),
            }
        )

    ground_truth = {
        "intrinsics": {"fx": FOCAL, "fy": FOCAL, "cx": CX, "cy": CY, "width": IMG_W, "height": IMG_H},
        "cameras": cameras,
        "n_points_3d": N_POINTS,
    }
    (OUT_DIR / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))
    print(f"Wrote {N_CAMERAS} images to {IMAGES_DIR}")
    print(f"Wrote ground truth to {OUT_DIR / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
