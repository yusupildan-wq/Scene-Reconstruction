#!/usr/bin/env bash
set -Eeuo pipefail

# nvidia-smi talks to NVML, which can report the GPU before the CUDA driver
# API is ready to create a context. On a freshly attached RunPod GPU this
# shows up as: nvidia-smi succeeds immediately, then the very first
# `torch.cuda.is_available()` call in the container fails with
# "CUDA initialization: CUDA unknown error ... Setting the available devices
# to be zero". That message's env-var hint is PyTorch boilerplate for any
# unexpected cudaGetDeviceCount() failure, not evidence CUDA_VISIBLE_DEVICES
# was actually touched. This is a documented cold-start race on GPU rental
# hosts, not a configuration bug, so poll for readiness with a bounded budget
# instead of a blind sleep or a single one-shot check.

python_bin="${1:?usage: wait_for_cuda.sh <python-executable>}"
max_attempts="${CUDA_READY_MAX_ATTEMPTS:-60}"
poll_seconds="${CUDA_READY_POLL_SECONDS:-5}"

echo "--- CUDA/NVIDIA environment diagnostics ---"
env | grep -E 'CUDA|NVIDIA' || echo "(no CUDA/NVIDIA env vars set)"
ls -l /dev/nvidia* 2>&1 || echo "(no /dev/nvidia* devices found)"
nvidia-smi || true
echo "--------------------------------------------"

ready=0
for attempt in $(seq 1 "$max_attempts"); do
  if "$python_bin" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    ready=1
    break
  fi
  echo "CUDA not ready yet (attempt $attempt/$max_attempts); retrying in ${poll_seconds}s"
  sleep "$poll_seconds"
done

"$python_bin" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'

if [[ "$ready" -ne 1 ]]; then
  echo "ERROR: CUDA did not become available after $max_attempts attempts (~$((max_attempts * poll_seconds))s)" >&2
  exit 1
fi
