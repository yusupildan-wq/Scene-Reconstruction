"""Run the expensive full-room geometry stage without notebook state.

This is the reproducible RunPod entrypoint. It checkpoints selected source
frames, the exact pair graph, aligned cameras/intrinsics and every dense point
map before Gaussian training starts. A stopped pod can therefore resume from
the geometry directory instead of decoding the video and aligning the room
again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.pipeline import extract_frames
from experiments.reconstruction_graph import (
    COLAB_PROFILE,
    PHOTOREAL_PROFILE,
    ReconstructionProfile,
    build_pair_graph,
    dust3r_pairs,
)


def _to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _geometry_is_complete(output_dir: Path, expected_views: int) -> bool:
    camera_file = output_dir / "geometry" / "cameras.npz"
    point_files = list((output_dir / "geometry").glob("points_*.npy"))
    mask_files = list((output_dir / "geometry").glob("mask_*.npy"))
    return camera_file.exists() and len(point_files) == expected_views and len(mask_files) == expected_views


def save_aligned_geometry(scene, frame_paths: list[Path], output_dir: Path) -> None:
    geometry_dir = output_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    poses = [_to_numpy(value).astype(np.float32) for value in scene.get_im_poses()]
    intrinsics = [_to_numpy(value).astype(np.float32) for value in scene.get_intrinsics()]
    points = [_to_numpy(value).astype(np.float32) for value in scene.get_pts3d()]
    masks = [_to_numpy(value).astype(bool) for value in scene.get_masks()]
    np.savez(
        geometry_dir / "cameras.npz",
        poses=np.stack(poses),
        intrinsics=np.stack(intrinsics),
        frame_names=np.asarray([path.name for path in frame_paths]),
    )
    for index, (point_map, mask) in enumerate(zip(points, masks)):
        np.save(geometry_dir / f"points_{index:04d}.npy", point_map)
        np.save(geometry_dir / f"mask_{index:04d}.npy", mask)
    (geometry_dir / "COMPLETE.json").write_text(
        json.dumps({"views": len(frame_paths), "point_maps": len(points)}, indent=2),
        encoding="utf-8",
    )


def run_geometry(
    video_path: Path,
    output_dir: Path,
    profile: ReconstructionProfile,
    *,
    force: bool = False,
) -> None:
    os.environ.setdefault("MAX_JOBS", "2")
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(frames_dir.glob("*.jpg"))
    if existing_frames and not force:
        frame_paths = existing_frames
        print(f"Reusing {len(frame_paths)} checkpointed frames")
    else:
        result = extract_frames(
            video_path,
            frames_dir,
            max_selected_frames=profile.max_frames,
        )
        frame_paths = result.frame_paths
        print(f"Selected {len(frame_paths)} sharp distributed frames")

    graph_path = output_dir / "pair_graph.json"
    if graph_path.exists() and not force:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        print(f"Reusing checkpointed graph with {len(graph['combined_pairs'])} pairs")
    else:
        graph = build_pair_graph(frame_paths, profile, graph_path)
        print(
            f"Built {len(graph['temporal_pairs'])} temporal + "
            f"{len(graph['verified_loop_closures'])} verified loop edges"
        )

    if _geometry_is_complete(output_dir, len(frame_paths)) and not force:
        print(f"Aligned geometry is already complete at {output_dir / 'geometry'}")
        return

    # Imported only after all CPU checkpoints exist. RunPod must have the
    # official DUSt3R repository installed or on PYTHONPATH.
    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("The geometry stage requires a CUDA GPU")
    model = AsymmetricCroCo3DStereo.from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)
    images = load_images(
        [str(path) for path in frame_paths], size=profile.dust3r_image_size
    )
    pairs = dust3r_pairs(images, graph)
    print(f"Running dense inference for {len(pairs)} directed pairs")
    output = inference(pairs, model, device, batch_size=1)
    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
    )
    final_loss = scene.compute_global_alignment(
        init="mst",
        niter=profile.alignment_iterations,
        schedule="cosine",
        lr=0.01,
    )
    print(f"Global alignment final loss: {final_loss}")
    save_aligned_geometry(scene, frame_paths, output_dir)
    print(f"Checkpointed aligned geometry at {output_dir / 'geometry'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("run_artifacts/full-room"))
    parser.add_argument(
        "--profile", choices=("colab", "photoreal"), default="photoreal"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    profile = PHOTOREAL_PROFILE if args.profile == "photoreal" else COLAB_PROFILE
    run_geometry(args.video, args.output, profile, force=args.force)


if __name__ == "__main__":
    main()
