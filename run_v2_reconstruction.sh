#!/usr/bin/env bash
# V2 reconstruction: video -> COLMAP -> splatfacto (MCMC) -> exported .ply + renders.
# See V2_ARCHITECTURE.md for what this replaces and why.
#
# Run on the RunPod pod. Writes everything to /workspace/v2-reconstruction/,
# isolated from all existing V1/Codex outputs. Safe to re-run: each step
# checks for its own output before redoing work, and the whole thing is
# nohup-safe (see the bottom) so an SSH/agent disconnect doesn't kill it.
set -euo pipefail

# Real failure hit on the third run: COLMAP's apt-packaged build tries to
# initialize a Qt GUI context even for pure CLI feature extraction/matching,
# and this pod is headless (no X display) -- crashed with "qt.qpa.xcb: could
# not connect to display". QT_QPA_PLATFORM=offscreen forces Qt to use a
# headless-safe rendering backend instead of trying to reach a real display.
# Standard, well-known fix for exactly this error on headless Linux servers.
export QT_QPA_PLATFORM=offscreen

# Real failure hit on the seventh run: training crashed with
# "AssertionError: duplicate template name" deep inside
# torch._inductor.kernel.flex_attention, triggered by torch._dynamo.optimize
# during splatfacto startup -- a torch.compile/Triton internal bug (a
# module's kernel-template registration ran twice in the same process), not
# something caused by this script's own flags. torch.compile is a
# performance optimization, not required for correctness, so disabling it
# entirely is the standard, well-known workaround for this class of
# torch-inductor bug -- forces eager-mode execution instead.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

OUT=/workspace/v2-reconstruction
VIDEO=/workspace/IMG_1705.mov
mkdir -p "$OUT"

echo "=== [1/4] Install nerfstudio (pulls in gsplat + a compatible torch/CUDA build) ==="
if ! command -v ns-train >/dev/null 2>&1; then
  # This pod's base image has some packages (e.g. blinker) installed via apt,
  # which leaves no pip RECORD file -- pip refuses to upgrade them by default
  # ("Cannot uninstall blinker ... no RECORD file was found"). --ignore-installed
  # tells pip to shadow the apt-installed version with its own instead of
  # trying to uninstall it first. Real failure hit on the first attempt, not
  # a preemptive guess.
  pip install -q --ignore-installed blinker nerfstudio
else
  echo "already installed, skipping"
fi

echo "=== [1.5/4] Install COLMAP CLI + ffmpeg if missing ==="
# Real failure hit on the second run: ns-process-data needs the standalone
# `colmap` command, which nerfstudio does NOT install alongside itself
# (confirmed by nerfstudio's own docs -- "COLMAP install issues are common").
# ffmpeg checked defensively too, to avoid a third failure cycle for the
# same reason -- most RunPod PyTorch images have it, but not guaranteed.
if ! command -v colmap >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq colmap ffmpeg
else
  echo "already installed, skipping"
fi

echo "=== [2/4] Frame extraction + COLMAP (feature matching, incremental SfM) ==="
# Real failure hit on the fourth run: COLMAP's GPU-accelerated SIFT uses
# OpenGL (not just CUDA) for the GPU context, which needs a real or virtual
# display -- this pod is fully headless (confirmed: "Check failed:
# context_.create()" in opengl_utils.cc). --no-gpu (verified real field,
# colmap_converter_to_nerfstudio_dataset.py's `gpu: bool = True`) falls back
# to CPU-based SIFT extraction/matching -- slower, but has no display
# dependency at all, so it just works instead of needing to rig up a
# virtual framebuffer for marginal speed gain.
if [ ! -f "$OUT/data/colmap/sparse/0/cameras.bin" ]; then
  # Real failure hit on the fifth run: process got silently "Killed" (the
  # OOM killer, confirmed: this container's real cgroup memory limit is
  # ~125GB, not the 1TB the host reports) partway through feature
  # extraction. Root cause: nerfstudio's run_colmap() has no thread-count
  # flag at all, and COLMAP's SIFT extraction auto-detects thread count
  # from nproc, which reported this HOST's 256 cores -- 256 parallel
  # CPU-side SIFT workers on 1920x1080 frames blew past the container's
  # actual budget. taskset restricts CPU affinity for the whole process
  # tree; COLMAP's thread auto-detection respects that, so this caps
  # parallelism (and peak memory) without needing to patch nerfstudio.
  # Sixth run: taskset alone got much further (129/313 files vs. 8/313
  # before) but still eventually got OOM-killed -- roughly proportional to
  # the thread-count reduction, suggesting memory pressure scales with
  # total images processed, not just instantaneous parallelism (root cause
  # inside COLMAP's own feature extractor, not something nerfstudio or this
  # script controls directly). Rather than keep chasing the exact mechanism,
  # applying two independent safety margins: fewer total frames
  # (num_frames_target, real default 300) and a tighter CPU cap.
  taskset -c 0-7 ns-process-data video --data "$VIDEO" --output-dir "$OUT/data" --no-gpu --num-frames-target 150
else
  echo "COLMAP output already present, skipping"
fi

echo "=== [3/4] Train splatfacto ==="
# --pipeline.model.strategy mcmc dropped: the pip-published nerfstudio release
# installed here predates MCMC-strategy support (confirmed via the real tyro
# error -- "Unrecognized options: --pipeline.model.strategy, mcmc" -- not
# guessed). That flag only exists on nerfstudio's unreleased GitHub main
# branch, which is not what `pip install nerfstudio` gives us. Falling back to
# splatfacto's actual installed default (its original split/clone/prune
# DefaultStrategy) rather than chase exact-version flag compatibility further.
ns-train splatfacto \
  --data "$OUT/data" \
  --output-dir "$OUT/training" \
  --viewer.quit-on-train-completion True

echo "=== [4/4] Export a standard .ply + render images for visual inspection ==="
CONFIG=$(find "$OUT/training" -name config.yml | sort | tail -1)
echo "Using config: $CONFIG"

ns-export gaussian-splat \
  --load-config "$CONFIG" \
  --output-dir "$OUT/export"

ns-render dataset \
  --load-config "$CONFIG" \
  --output-path "$OUT/renders" \
  --split train

echo "=== DONE ==="
echo "Exported scene: $OUT/export/splat.ply"
echo "Renders (compare against real photos in the same folder): $OUT/renders/"
echo "Pull $OUT/export/splat.ply to compare against V1's"
echo "experiments/artifacts/2026-08-17-full-room-8000/dust3r_scene.ply (16.69dB, preserved in the V1 checkpoint)."

# To run this resilient to disconnects:
#   nohup bash run_v2_reconstruction.sh > /workspace/v2-reconstruction.log 2>&1 &
#   tail -f /workspace/v2-reconstruction.log
# Ctrl+C on tail only stops the tail, not the background job.
