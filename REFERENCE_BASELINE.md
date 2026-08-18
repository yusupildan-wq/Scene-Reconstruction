# External reference baseline: official COLMAP + 3D Gaussian Splatting

## Why this exists

Every diagnosis so far (registration audit, cross-view consistency checks,
the 16-run ablation) has been about *our own* pipeline: our DUSt3R
integration, our fusion code, our trainer. That's thorough, but it can't rule
out a subtle bug somewhere in our own code that none of those tests happen to
catch. The fastest way to find out whether the room itself is reconstructible
at all is to run a completely independent, unmodified, widely-used pipeline
against the exact same video and just look at the result.

This is not a redesign of the project. It's one isolated test, in its own
branch/worktree/output directory, that doesn't touch anything already built.

## What's being introduced (per the "teach new tech" rule)

**COLMAP (the standalone `colmap` CLI, not the `pycolmap` Python bindings
already used elsewhere in this project)** — a mature, open-source
Structure-from-Motion + Multi-View Stereo program (from ETH Zurich /
Microsoft-affiliated researchers). Same underlying engine `pycolmap` wraps,
but invoked as a real command-line program rather than through Python.
Input: a folder of images. Output: camera poses, intrinsics, and a sparse
3D point cloud, written to disk as COLMAP's own binary format. Not a neural
network — classical feature matching (SIFT) + bundle adjustment.

**The official `graphdeco-inria/gaussian-splatting` repository** — the actual
reference implementation from the paper that introduced 3D Gaussian
Splatting. Installed via `git clone` (with submodules for the CUDA
rasterizer and a k-nearest-neighbors helper). Input: a COLMAP-format scene
(images + poses/intrinsics/points, exactly what the step above produces).
Output: a trained `.ply` Gaussian scene plus rendered images. Not
pretrained/downloaded weights -- like our own trainer, it optimizes a fresh
set of Gaussians from scratch for this one specific scene, per-scene, no
generalization across scenes.

**License, since this affects what happens if it wins**: this repo is
released under INRIA's own non-commercial research license, not something
permissively licensed like MIT/Apache. Using it for this benchmark test
(your own research/learning use, not distributing it) is fine. If it turns
out to meaningfully outperform our own pipeline and we want to actually ship
that quality, the *algorithm* isn't licensed -- only their specific code is
-- so the real path would be porting the same ideas into our already
Apache-2.0-licensed `gsplat`-based trainer, not shipping their code directly.
Flagging this now so it's not a surprise later, not as a reason to avoid
running the test.

## Verified, not guessed

The exact commands below were checked against the real source of
`convert.py`, `train.py`, and `arguments/__init__.py` in the official repo
(not assumed from memory) -- flag names, defaults (30,000 iterations,
`OPENCV` camera model, exhaustive matching), and short-flag aliases (`-s`,
`-m`, `-r`) are all confirmed as of 2026-08-18.

## Run this on the RunPod pod (same environment as the rest of this project)

```bash
# 1. Clone the reference implementation, isolated from everything else
cd /workspace
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting

# 2. Install just what's needed on top of the RunPod PyTorch container
#    (skip the repo's own conda env -- redundant with what's already here)
pip install -q plyfile tqdm
pip install -q submodules/diff-gaussian-rasterization submodules/simple-knn

# 3. Install the standalone COLMAP CLI (system package -- separate from the
#    pycolmap bindings already used elsewhere in this project)
apt-get update && apt-get install -y colmap

# 4. Extract frames from the SAME source video already on the volume, using
#    plain fps-based sampling -- the standard/recommended approach for this
#    pipeline, deliberately NOT our own blur-filtered frame selection, so
#    this test stays uncontaminated by any of our own preprocessing code.
#    fps=2 on a ~125s video gives ~250 frames; COLMAP (unlike DUSt3R) isn't
#    memory-bound by frame count the same way, so this is a safe starting
#    point, not a tight constraint.
mkdir -p /workspace/reference-baseline/input
ffmpeg -i /workspace/IMG_1705.mov -qscale:v 2 -vf "fps=2" \
  /workspace/reference-baseline/input/frame_%04d.jpg

# 5. Their own COLMAP wrapper -- feature extraction, exhaustive matching,
#    incremental bundle adjustment, then undistortion into pinhole images.
#    This is the slowest step; expect real time here (tens of minutes),
#    not a quick command.
cd /workspace/gaussian-splatting
python convert.py -s /workspace/reference-baseline

# 6. Their own official trainer, completely unmodified, default settings
#    (30,000 iterations -- this will take a while; that's expected and is
#    the actual point of using their real defaults, not a shortcut).
python train.py -s /workspace/reference-baseline -m /workspace/reference-baseline/output

# 7. Render the result so there's something to actually look at
python render.py -m /workspace/reference-baseline/output
```

## What to do with the result

`render.py` writes rendered images under
`/workspace/reference-baseline/output/train/ours_30000/renders/` (and a
matching `gt/` folder with the real photos for the same views) -- open a few
of those side by side with the real photo first, before looking at any
number. That visual comparison is the actual answer to "can a proven
pipeline get a sharp room from our exact footage."

If it's sharp: pull `point_cloud.ply` from
`/workspace/reference-baseline/output/point_cloud/iteration_30000/` and
compare it against our current best full-room result
(`/workspace/full-room-photoreal-v2/gaussians_step8000.ply`, 16.69dB) both
visually and, once that's confirmed worth pursuing, on the same held-out
camera views for a real PSNR/SSIM number -- worth writing that comparison
script only after the visual answer is known, not before.

If it's still blurry on our exact footage: that's strong evidence the
capture itself (motion blur, exposure changes, weak parallax on some
segments -- all flagged as open hypotheses in `PROJECT_STATE.md`) is the
real limiting factor, not any of our own pipeline code, and the next real
lever is a better capture, not another pipeline change.
