# External reference baselines

## Why this exists

Every diagnosis so far (registration audit, cross-view consistency checks,
the 16-run ablation) has been about *our own* pipeline: our DUSt3R
integration, our fusion code, our trainer. That's thorough, but it can't rule
out a subtle bug somewhere in our own code that none of those tests happen to
catch. The fastest way to find out whether the room itself is reconstructible
at all is to run completely independent, unmodified, widely-used pipelines
against the exact same video and just look at the result.

This is not a redesign of the project. Three isolated tests, each in its own
output directory under `/workspace/reference-baseline/`, none of which touch
the existing DUSt3R pipeline, the four experimental worktrees, or main.

## The three paths, and why each one

**Path A — official `graphdeco-inria/gaussian-splatting`** (COLMAP incremental
+ the actual paper's reference trainer). The single most standard,
most-cited benchmark in this field. If this can't get a sharp room from our
footage, that's very strong evidence the capture itself is the limit.

**Path B — Nerfstudio's `splatfacto`** (COLMAP incremental, run via
`ns-process-data`, + a mature, actively-maintained trainer built on the SAME
`gsplat` library this project already uses). The most turnkey of the three
(one command does frame extraction + COLMAP), and — unlike Path A —
Apache-2.0 licensed, so if it wins it can actually ship in the product later,
not just serve as a benchmark. This is the fastest path to a first result.

**Path C — COLMAP's own global mapper** (same feature extraction/matching as
A and B, but a *global* reconstruction solver instead of *incremental*).
This is the most surgically targeted test of all three: your 16-run
experiment found geometric disagreement getting *worse* as more views were
added, and incremental SfM (what A and B both use) registers cameras one at
a time, which is exactly the kind of process that can accumulate drift as
more images join. A global solver fits all cameras jointly from the whole
view graph at once instead, and is generally more robust to that specific
failure mode. GLOMAP (the standalone project this used to require) is now
deprecated -- its approach was merged directly into COLMAP 4.1.1+, so this
doesn't need a separate tool, just a newer COLMAP and a different mapper flag.

Run B first if you want the fastest possible first look at a room. Run C
specifically if you want the most direct test of the actual failure mode
already diagnosed. A is the most authoritative reference point but the
slowest to set up.

## Licenses, since this affects what happens if one wins

- Path A (INRIA gaussian-splatting): non-commercial research license.
  Fine for this benchmark; if it wins, port the *ideas* into our own
  Apache-2.0 `gsplat`-based trainer rather than shipping their code.
- Path B (Nerfstudio/splatfacto): Apache 2.0. Could be adopted directly.
- Path C (COLMAP): BSD-family, permissive either way -- this only changes
  the camera/geometry step, not the trainer, so pair it with our existing
  trainer or Path B's trainer, either is fine license-wise.

## Verified, not guessed

Path A's commands were checked against the real source of `convert.py`,
`train.py`, and `arguments/__init__.py`. Path B's commands were checked
against Nerfstudio's real docs and `method_configs.py` (confirming
`splatfacto` is a real registered method) and its license file (confirmed
Apache 2.0). Path C's claim about COLMAP absorbing GLOMAP was checked
against GLOMAP's own README and COLMAP's real CLI source
(`src/colmap/exe/sfm.cc`), which confirmed a `--mapper {incremental,
hierarchical, global}` option on `automatic_reconstructor`. What's genuinely
**not yet verified**: whether the COLMAP version `apt-get install colmap`
pulls on the RunPod image is new enough (4.1.1+) to have this flag at all --
that's a real unknown, flagged rather than assumed, and Path C includes a
version check for exactly this reason.

---

## Path A: official COLMAP + reference 3D Gaussian Splatting

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
mkdir -p /workspace/reference-baseline/path-a/input
ffmpeg -i /workspace/IMG_1705.mov -qscale:v 2 -vf "fps=2" \
  /workspace/reference-baseline/path-a/input/frame_%04d.jpg

# 5. Their own COLMAP wrapper -- feature extraction, exhaustive matching,
#    incremental bundle adjustment, then undistortion into pinhole images.
#    This is the slowest step; expect real time here (tens of minutes),
#    not a quick command.
cd /workspace/gaussian-splatting
python convert.py -s /workspace/reference-baseline/path-a

# 6. Their own official trainer, completely unmodified, default settings
#    (30,000 iterations -- this will take a while; that's expected and is
#    the actual point of using their real defaults, not a shortcut).
python train.py -s /workspace/reference-baseline/path-a \
  -m /workspace/reference-baseline/path-a/output

# 7. Render the result so there's something to actually look at
python render.py -m /workspace/reference-baseline/path-a/output
```

Renders land in
`/workspace/reference-baseline/path-a/output/train/ours_30000/renders/`
(with matching `gt/` real photos for the same views). The trained scene is
`/workspace/reference-baseline/path-a/output/point_cloud/iteration_30000/point_cloud.ply`.

---

## Path B: Nerfstudio `splatfacto` (fastest first result)

```bash
# 1. Install (this pulls in its own gsplat/torch-compatible extras)
pip install -q nerfstudio

# 2. One command: frame extraction (ffmpeg) + COLMAP, from the video directly
ns-process-data video \
  --data /workspace/IMG_1705.mov \
  --output-dir /workspace/reference-baseline/path-b

# 3. Train -- defaults, unmodified
ns-train splatfacto --data /workspace/reference-baseline/path-b

# 4. Render a comparison video/images from the trained checkpoint once
#    training finishes (path printed at the end of ns-train's output, looks
#    like outputs/.../splatfacto/<timestamp>/config.yml)
ns-render camera-path --load-config <config.yml path from step 3> \
  --output-path /workspace/reference-baseline/path-b/render.mp4
```

Nerfstudio also has a built-in web viewer (`ns-train` prints a URL) if you'd
rather look around the scene live during/after training than wait for an
export -- genuinely the fastest way to eyeball whether this is working.

---

## Path C: COLMAP's global mapper (most targeted test)

```bash
# 1. Check the installed COLMAP version FIRST -- global mapper needs 4.1.1+
apt-get update && apt-get install -y colmap
colmap -h | head -5   # look for a version string
# If this is older than 4.1.1, either build from source (see colmap.github.io/install)
# or skip Path C -- it isn't worth burning time on an unsupported version.

# 2. Reuse the same frames Path A already extracted (or re-extract if running
#    this alone -- same ffmpeg command as Path A, step 4)
mkdir -p /workspace/reference-baseline/path-c
cp -r /workspace/reference-baseline/path-a/input /workspace/reference-baseline/path-c/input

# 3. One-shot automatic reconstruction with the GLOBAL mapper instead of
#    incremental. --dense 0 since we only need sparse cameras+points for
#    Gaussian Splatting, not a dense mesh.
colmap automatic_reconstructor \
  --workspace_path /workspace/reference-baseline/path-c \
  --image_path /workspace/reference-baseline/path-c/input \
  --data_type video \
  --quality high \
  --mapper global \
  --dense 0

# 4. Feed the resulting sparse/ + input/ into our OWN existing trainer (not
#    a new one) -- this isolates the test to "does a global camera solver
#    fix it" specifically, without also changing the trainer at the same
#    time. Reuses worker/runner.py's colmap_reconstruction_to_scene, already
#    proven working -- see run_colmap_sfm's pattern in worker/runner.py for
#    the exact call shape if wiring this into a script.
```

Path C's step 4 deliberately does NOT specify exact code yet -- confirm step
3 actually produces a usable `sparse/0` first (this is the least-verified
path of the three), then wire it into our existing trainer once that's
confirmed, rather than writing untested glue code against an unverified
output format.

---

## What counts as an answer

Open a rendered image next to the real photo for the same view, for
whichever path finishes first. That visual comparison is the actual
question, not a metric.

**If any path is visibly sharp**: strong evidence the capture is fine and
something in our own pipeline (geometry, fusion, or trainer -- Path C vs. A
vs. B tells you which) is the fixable bottleneck. Pull that path's `.ply`
and compare it against our current best full-room result
(`/workspace/full-room-photoreal-v2/gaussians_step8000.ply`, 16.69dB) both
visually and, once that's worth pursuing, on the same held-out camera views
for a real PSNR/SSIM number.

**If all three are still blurry on our exact footage**: that's strong
evidence the capture itself (motion blur, exposure changes, weak parallax on
some segments -- open hypotheses already in `PROJECT_STATE.md`) is the real
limiting factor, not any of our own pipeline code, and the next real lever
is a better capture, not another pipeline change.
