#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
MODEL_ROOT="${MODEL_ROOT:-$WORKSPACE_ROOT/models}"
STATE_ROOT="${STATE_ROOT:-$WORKSPACE_ROOT/bootstrap/state}"
PROJECT_ROOT="${PROJECT_ROOT:-/opt/project}"
VGGT_PYTHON="${VGGT_PYTHON:-/opt/venvs/vggt/bin/python}"
GSPLAT_PYTHON="${GSPLAT_PYTHON:-/opt/venvs/gsplat/bin/python}"
GSPLAT_ROOT="${GSPLAT_ROOT:-/opt/gsplat}"
TORCH_HOME="${TORCH_HOME:-$WORKSPACE_ROOT/cache/torch}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-$TORCH_HOME/hub/checkpoints/model.pt}"

mkdir -p "$STATE_ROOT" "$MODEL_ROOT" \
  "$WORKSPACE_ROOT/datasets" "$WORKSPACE_ROOT/experiments" \
  "$WORKSPACE_ROOT/checkpoints" "$WORKSPACE_ROOT/logs" \
  "$WORKSPACE_ROOT/manifests" "$WORKSPACE_ROOT/cache/huggingface/hub" \
  "$WORKSPACE_ROOT/cache/torch/hub/checkpoints" \
  "$WORKSPACE_ROOT/cache/torch_extensions" "$WORKSPACE_ROOT/cache/pip"
test -w "$WORKSPACE_ROOT"

nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
touch "$STATE_ROOT/01_runtime.ok"

"$VGGT_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
touch "$STATE_ROOT/02_torch.ok"

"$VGGT_PYTHON" -c 'from vggt.models.vggt import VGGT; print("VGGT import: healthy")'
touch "$STATE_ROOT/03_vggt.ok"

"$VGGT_PYTHON" -c 'import pycolmap; assert pycolmap.__version__ == "3.10.0"; print("VGGT pycolmap: healthy")'
touch "$STATE_ROOT/04_pycolmap.ok"

"$GSPLAT_PYTHON" -c 'import gsplat; print("gsplat", gsplat.__version__)'
touch "$STATE_ROOT/05_gsplat_python.ok"

TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORKSPACE_ROOT/cache/torch_extensions}" \
  "$GSPLAT_PYTHON" "$PROJECT_ROOT/bootstrap/verify_cuda.py"
touch "$STATE_ROOT/06_gsplat_cuda.ok"

"$GSPLAT_PYTHON" "$GSPLAT_ROOT/examples/simple_trainer.py" default --help >/dev/null
touch "$STATE_ROOT/07_trainer.ok"

if [[ ! -s "$VGGT_CHECKPOINT" ]]; then
  echo "ERROR: persistent VGGT checkpoint missing: $VGGT_CHECKPOINT" >&2
  exit 1
fi
touch "$STATE_ROOT/08_full_smoke_test.ok"

echo "ENVIRONMENT HEALTHY — ZERO INSTALLATION REQUIRED"
