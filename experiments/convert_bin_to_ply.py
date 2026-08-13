"""Convert our custom dust3r_scene.bin into a standard 3D Gaussian Splatting .ply.

One-off local utility, not part of the pipeline -- dust3r_scene.bin already has
everything needed (the Colab export cell writes quats; our browser viewer's
loadBinaryScene just skips those bytes), so this runs entirely on CPU with no
GPU/Colab involved.

The .ply convention stores Gaussians in their raw/unconstrained form (pre-exp
scale, pre-sigmoid opacity, SH0-encoded color) -- the opposite of what
runner.py returns (already activated) -- so this inverts those activations
back before writing.
"""

import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814  # fixed constant for the standard SH0 color encoding
SCENE_SCALE_FACTOR = 100.0


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


def main():
    root = Path(__file__).resolve().parent.parent
    bin_path = root / "frontend" / "public" / "dust3r_scene.bin"
    ply_path = root / "frontend" / "public" / "dust3r_scene.ply"

    n, means, quats, scales, opacities, colors = load_bin(bin_path)
    print(f"Loaded {n} Gaussians from {bin_path}")

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

    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(ply_path)
    size_mb = ply_path.stat().st_size / 1e6
    print(f"Wrote {ply_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
