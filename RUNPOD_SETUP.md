# RunPod GPU execution workflow

RunPod is used only for VGGT inference, gsplat CUDA validation, training,
rendering, evaluation, and GPU-intensive refinement. Repository setup,
dependency resolution, frame preparation, manifests, viewer work, and image
builds happen before the pod starts.

## 1. Local preflight (GPU off)

Prepare the exact input scene locally if needed:

```powershell
python scripts/prepare_v3_scene.py `
  --source-frames <extracted-frames> --scene-dir <prepared-scene> --count 96
```

```powershell
python scripts/preflight_runpod.py `
  --scene-dir <prepared-scene> `
  --output-dir <planned-persistent-output> `
  --manifest-out bootstrap/preflight-report.json
```

Do not start paid compute unless it prints
`LOCAL PREFLIGHT PASSED — SAFE TO START GPU`.

Record before launch: the GPU-required task, preparation already completed,
expected runtime/utilization, exact input, and exact output.

Seed the VGGT checkpoint from a CPU/free machine or migrate the already
downloaded verified file into the mounted volume. The cache destination is:

```text
/workspace/cache/torch/hub/checkpoints/model.pt
```

`scripts/cache_vggt_checkpoint.py` performs atomic download, size validation,
optional SHA-256 validation, and reuse of an existing valid file.

## 2. Interchangeable pod

Use the pinned image in `bootstrap/README.md`, attach the existing network
volume at `/workspace`, and use any validated compatible GPU. Current minimum:
compute capability 8.0 and 24 GB VRAM; 48 GB is recommended for large scenes.

Startup must finish with `READY FOR GPU WORK`. If normal startup invokes pip,
apt, git clone, a model download, or gsplat compilation, stop and repair the
image/cache outside RunPod.

## 3. GPU stage

```bash
/opt/project/bootstrap/run_gpu_job.sh v3-room-7k \
  /opt/venvs/vggt/bin/python /opt/project/experiments/run_v3_vggt.py \
  --scene-dir /workspace/datasets/v3-room/scene \
  --result-dir /workspace/experiments/v3-room-7k \
  --vggt-root /opt/vggt --gsplat-root /opt/gsplat \
  --vggt-python /opt/venvs/vggt/bin/python \
  --gsplat-python /opt/venvs/gsplat/bin/python \
  --stage all --data-factor 2 --max-steps 7000
```

Valid VGGT geometry and completed PLY exports are reused automatically. Never
use `--force-geometry` or `--force-training` without evidence that the existing
artifact is invalid.

For the high-quality A/B pass, preserve the existing 7k result and use a new
output directory with `--stage train --quality-profile high`. This means full
resolution, 30k steps, pose refinement, and antialiased rasterization. Budget
approximately the official 30k single-GPU training class (tens of minutes), then
stop the pod immediately after verification and artifact export.

## 4. Completion and shutdown

Verify the job exit file contains `0`, the run manifest says `complete`, and
the PLY/checkpoints/logs live under `/workspace`. Stop the pod immediately.
Review and copy results from persistent storage/local tooling.

The successful existing V3 geometry and 7,000-step output remain valid. Do not
rerun VGGT or overwrite them; later experiments consume those artifacts and
write to a new result directory.
