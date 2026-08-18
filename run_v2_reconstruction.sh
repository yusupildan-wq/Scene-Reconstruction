#!/usr/bin/env bash
# V2 reconstruction: video -> COLMAP -> splatfacto (MCMC) -> exported .ply + renders.
# See V2_ARCHITECTURE.md for what this replaces and why.
#
# Run on the RunPod pod. Writes everything to /workspace/v2-reconstruction/,
# isolated from all existing V1/Codex outputs. Safe to re-run: each step
# checks for its own output before redoing work, and the whole thing is
# nohup-safe (see the bottom) so an SSH/agent disconnect doesn't kill it.
set -euo pipefail

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

echo "=== [2/4] Frame extraction + COLMAP (feature matching, incremental SfM) ==="
if [ ! -f "$OUT/data/colmap/sparse/0/cameras.bin" ]; then
  ns-process-data video --data "$VIDEO" --output-dir "$OUT/data"
else
  echo "COLMAP output already present, skipping"
fi

echo "=== [3/4] Train splatfacto with MCMC densification ==="
# Defaults otherwise -- deliberately not hand-tuning hyperparameters for the
# first real run, to test the mature implementation as intended before
# customizing anything.
ns-train splatfacto \
  --data "$OUT/data" \
  --output-dir "$OUT/training" \
  --pipeline.model.strategy mcmc \
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
