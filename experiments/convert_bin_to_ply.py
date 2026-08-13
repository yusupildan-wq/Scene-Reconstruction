"""Convert a saved Gaussian scene (.bin or training .npz) into a standard .ply.

One-off local utility, not part of the pipeline -- dust3r_scene.bin already has
everything needed (the Colab export cell writes quats; our browser viewer's
loadBinaryScene just skips those bytes), so this runs entirely on CPU with no
GPU/Colab involved.

The .ply convention stores Gaussians in their raw/unconstrained form (pre-exp
scale, pre-sigmoid opacity, SH0-encoded color) -- the opposite of what
runner.py returns (already activated) -- so this inverts those activations
back before writing.
"""

import argparse
import struct
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814  # fixed constant for the standard SH0 color encoding
SCENE_SCALE_FACTOR = 100.0
MAX_AXIS_RATIO = 20.0


def load_bin(path: Path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 0)[0]
    offset = 4
    means = np.frombuffer(data, dtype=np.float32, count=n * 3, offset=offset).reshape(n, 3)
    offset += n * 3 * 4
    quats = np.frombuffer(data, dtype=np.float32, count=n * 4, offset=offset).reshape(n, 4)
    offset += n * 4 * 4
    scales = np.frombuffer(data, dtype=np.float32, count=n * 3, offset=offset).reshape(n, 3)
    offset += n * 3 * 4
    opacities = np.frombuffer(data, dtype=np.float32, count=n, offset=offset)
    offset += n * 4
    colors = np.frombuffer(data, dtype=np.float32, count=n * 3, offset=offset).reshape(n, 3)
    return n, means, quats, scales, opacities, colors


def write_binary_ply(path: Path, vertices: np.ndarray) -> None:
    property_names = vertices.dtype.names or ()
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}"]
    header.extend(f"property float {name}" for name in property_names)
    header.extend(["end_header", ""])
    with path.open("wb") as stream:
        stream.write("\n".join(header).encode("ascii"))
        stream.write(vertices.tobytes())


def convert_npz(npz_path: Path, ply_path: Path) -> None:
    with np.load(npz_path) as scene:
        means = scene["means"].astype(np.float32)
        quats = scene["quats"].astype(np.float32)
        scales = scene["scales"].astype(np.float32)
        opacities = scene["opacities"].astype(np.float32)
        colors = scene["colors"].astype(np.float32)
    _write_scene(ply_path, means, quats, scales, opacities, colors)


def _write_scene(ply_path, means, quats, scales, opacities, colors):
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(means)

    # DUSt3R's coordinate frame is arbitrary/canonical, not real-world meters
    # -- this scene's whole bounding box is ~0.89 units, so individual
    # Gaussian scales are on the order of 1e-4 to 1e-3. Squaring values that
    # small while projecting the 3D covariance to 2D screen space (part of
    # this renderer's shader) almost certainly underflows float32 precision,
    # which is consistent with the renderer's own accessor reporting scale
    # (0,0,0) for a real trained Gaussian. Rescaling positions and scales
    # together by the same constant preserves the scene's shape exactly
    # (it's a uniform scale, not a distortion) while moving it into a unit
    # range this renderer's math was actually designed for.
    means = means * SCENE_SCALE_FACTOR
    scales = scales * SCENE_SCALE_FACTOR

    # Extreme anisotropy produces the screen-spanning needle artifacts seen
    # from novel viewpoints. Keep surface-like ellipsoids, but cap pathological
    # long axes relative to each Gaussian's smallest axis.
    minimum_axis = np.maximum(scales.min(axis=1, keepdims=True), 1e-8)
    scales = np.minimum(scales, minimum_axis * MAX_AXIS_RATIO)

    log_scales = np.log(np.clip(scales, 1e-8, None))
    logit_opacities = np.log(np.clip(opacities, 1e-6, 1 - 1e-6) / (1 - np.clip(opacities, 1e-6, 1 - 1e-6)))
    f_dc = (colors - 0.5) / SH_C0
    normals = np.zeros_like(means)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    vertices = np.empty(n, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertices["nx"], vertices["ny"], vertices["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    vertices["opacity"] = logit_opacities
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = log_scales[:, 0], log_scales[:, 1], log_scales[:, 2]
    # Our quats are (w,x,y,z) -- runner.py initializes identity as quats[:,0]=1,
    # matching gsplat's documented wxyz convention. But this renderer's .ply
    # parser reads rot_0..rot_3 as (x,y,z,w) (verified directly in its source:
    # tempRotation.set(rot_0, rot_1, rot_2, rot_3) is THREE.Quaternion's
    # (x,y,z,w) constructor order) -- write in the order it actually expects,
    # not the order our own data happens to be labeled in.
    vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = (
        quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]
    )

    write_binary_ply(ply_path, vertices)
    size_mb = ply_path.stat().st_size / 1e6
    print(f"Wrote {ply_path} ({size_mb:.1f} MB)")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.npz:
        ply_path = args.output or args.npz.with_suffix(".ply")
        convert_npz(args.npz, ply_path)
        return

    bin_path = root / "frontend" / "public" / "dust3r_scene.bin"
    ply_path = args.output or root / "frontend" / "public" / "dust3r_scene.ply"
    n, means, quats, scales, opacities, colors = load_bin(bin_path)
    print(f"Loaded {n} Gaussians from {bin_path}")
    _write_scene(ply_path, means, quats, scales, opacities, colors)


if __name__ == "__main__":
    main()
