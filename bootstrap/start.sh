#!/usr/bin/env bash
set -Eeuo pipefail

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
export MODEL_ROOT="${MODEL_ROOT:-$WORKSPACE_ROOT/models}"
export DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE_ROOT/datasets}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$WORKSPACE_ROOT/experiments}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$WORKSPACE_ROOT/checkpoints}"
export HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$WORKSPACE_ROOT/cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORKSPACE_ROOT/cache/torch_extensions}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE_ROOT/cache/pip}"
export PROJECT_ROOT="${PROJECT_ROOT:-/opt/project}"
export VGGT_ROOT="${VGGT_ROOT:-/opt/vggt}"
export GSPLAT_ROOT="${GSPLAT_ROOT:-/opt/gsplat}"

"$PROJECT_ROOT/bootstrap/verify_env.sh"

latest_checkpoint="$(find "$EXPERIMENT_ROOT" "$CHECKPOINT_ROOT" -type f \( -name '*.pt' -o -name '*.ckpt' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
echo "RUNPOD READY"
echo "GPU: healthy"
echo "PyTorch CUDA: healthy"
echo "VGGT: healthy"
echo "gsplat CUDA: healthy"
echo "models: cached"
echo "experiment state: ${latest_checkpoint:-none detected}"
echo "READY FOR GPU WORK"
