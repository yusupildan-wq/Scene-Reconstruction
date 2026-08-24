# V3: VGGT geometry and Gaussian Splatting

V3 replaces COLMAP pose/geometry estimation with VGGT while retaining Gaussian
Splatting for appearance optimization. V1 remains the control. The retired V2
PLY is not an input to this experiment.

Infrastructure is governed by `RUNPOD_SETUP.md` and `bootstrap/README.md`.
Run the local preflight before paid compute; normal RunPod startup performs no
installation, cloning, downloads, or general setup.

## Baseline experiment

1. Put 64-100 clean, time-ordered frames in `<scene>/images/`. The folder must
   contain only input images.
2. Install the official `facebookresearch/vggt` repository and obtain the
   appropriately licensed checkpoint.
3. Install the official `nerfstudio-project/gsplat` repository and its CUDA
   dependencies.
4. Run feed-forward VGGT first, without bundle adjustment:

   ```text
   python experiments/run_v3_vggt.py --scene-dir <scene> --result-dir <result> --vggt-root <vggt> --gsplat-root <gsplat>
   ```

5. Run a second geometry experiment with `--use-ba`. Never overwrite the first
   result directory.

The runner records exact input hashes and commands in `v3_run_manifest.json`,
verifies the COLMAP interchange model, and leaves the complete gsplat result in
the selected output directory.

## Required gates

- Inspect predicted cameras and the confidence-filtered point cloud before
  Gaussian training.
- Compare feed-forward cameras against the optional bundle-adjusted cameras.
- Reject flipped, discontinuous, duplicated, or collapsed camera trajectories.
- Retain exact-camera RGB renders and held-out metrics.
- Export a browser camera sidecar in the same normalized coordinate system as
  the final PLY.

Window and dynamic-region masking is a separate A/B experiment. The first run
must establish an unmasked baseline so masking cannot hide an unrelated camera
or coordinate-convention failure.
