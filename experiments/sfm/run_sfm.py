"""Runs COLMAP's real SfM pipeline (feature extraction -> matching -> incremental
reconstruction) against our synthetic images, then compares its ESTIMATED camera
positions against the GROUND TRUTH positions we used to render them.

This is the actual point of using a synthetic scene: SfM only recovers geometry up
to an unknown global scale/rotation/translation (there's no way to know "how big"
or "which way is up" from photos alone), so we can't just diff positions directly --
we align COLMAP's estimated cameras onto the ground-truth ones first (a similarity
transform: rotate + scale + translate), then measure the leftover error.
"""

import json
from pathlib import Path

import numpy as np
import pycolmap

SCENE_DIR = Path(__file__).parent / "data" / "synthetic_scene"
IMAGES_DIR = SCENE_DIR / "images"
DATABASE_PATH = SCENE_DIR / "database.db"
SPARSE_DIR = SCENE_DIR / "sparse"


def main() -> None:
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting features (SIFT keypoints per image)...")
    pycolmap.extract_features(str(DATABASE_PATH), str(IMAGES_DIR))

    print("Matching features exhaustively (every image pair)...")
    pycolmap.match_exhaustive(str(DATABASE_PATH))

    print("Running incremental SfM (pose estimation + triangulation + bundle adjustment)...")
    reconstructions = pycolmap.incremental_mapping(
        str(DATABASE_PATH), str(IMAGES_DIR), str(SPARSE_DIR)
    )

    if not reconstructions:
        print("SfM failed to register any cameras.")
        return

    # incremental_mapping can produce multiple disconnected reconstructions if not
    # all images could be linked together; take the one that registered the most.
    best_id = max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
    recon = reconstructions[best_id]
    print(f"\nRegistered {recon.num_reg_images()} / {len(list(IMAGES_DIR.glob('*.jpg')))} images")
    print(f"Triangulated {recon.num_points3D()} 3D points")
    print(f"Mean reprojection error: {recon.compute_mean_reprojection_error():.3f} px")

    ground_truth = json.loads((SCENE_DIR / "ground_truth.json").read_text())
    gt_by_name = {c["filename"]: np.array(c["position_world"]) for c in ground_truth["cameras"]}

    estimated, truth = [], []
    for image in recon.images.values():
        if image.has_pose and image.name in gt_by_name:
            estimated.append(np.array(image.projection_center()))
            truth.append(gt_by_name[image.name])
    estimated = np.array(estimated)
    truth = np.array(truth)

    # Align estimated camera centers onto ground truth with a similarity transform
    # (Umeyama's method), since SfM recovers shape only up to unknown scale/pose.
    def umeyama(src, dst):
        src_mean, dst_mean = src.mean(0), dst.mean(0)
        src_c, dst_c = src - src_mean, dst - dst_mean
        cov = (dst_c.T @ src_c) / len(src)
        U, S, Vt = np.linalg.svd(cov)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = U @ Vt
        var_src = (src_c**2).sum() / len(src)
        scale = S.sum() / var_src
        t = dst_mean - scale * R @ src_mean
        return R, scale, t

    R, s, t = umeyama(estimated, truth)
    aligned = (s * (R @ estimated.T).T) + t
    errors = np.linalg.norm(aligned - truth, axis=1)

    print(f"\nCompared {len(estimated)} registered cameras against ground truth")
    print(f"Mean camera position error after alignment: {errors.mean():.4f} (scene units)")
    print(f"Max camera position error after alignment: {errors.max():.4f} (scene units)")


if __name__ == "__main__":
    main()
