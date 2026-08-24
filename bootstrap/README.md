# Pinned GPU runtime

RunPod is an execution target only. Build and publish the root `Dockerfile`
locally or with `.github/workflows/gpu-image.yml`; never build it on a paid GPU.

## Runtime contract

- Ubuntu 22.04, Python 3.10, CUDA 12.4
- PyTorch 2.4.1/cu124 shared by two isolated virtual environments
- VGGT pinned to `a288dd0`; its environment uses pycolmap 3.10.0
- gsplat 1.5.3 pinned to `937e299`; installed from its official precompiled
  `pt24cu124/cp310/linux_x86_64` wheel
- Compiled compute capabilities: 8.0, 8.6, 8.9, 9.0
- Minimum target: compute capability 8.0 and 24 GB VRAM

The environments are separate because VGGT's exporter and gsplat v1.5.3's
dataset parser require incompatible pycolmap APIs.

## Persistent volume

Mount durable storage at `/workspace`:

```text
/workspace/cache/torch/hub/checkpoints/model.pt
/workspace/datasets/<scene>/images
/workspace/experiments/<run>
/workspace/checkpoints
/workspace/cache/huggingface
/workspace/cache/torch_extensions
/workspace/cache/pip
/workspace/bootstrap/state
/workspace/logs
/workspace/manifests
```

Never leave reusable assets under `/root/.cache` or `/tmp`. For the old V3 pod,
copy the verified checkpoint from `/root/.cache/torch/hub/checkpoints/model.pt`
to `/workspace/cache/torch/hub/checkpoints/model.pt`, verify byte count/hash,
then retire it. `TORCH_HOME=/workspace/cache/torch` makes the official VGGT
loader reuse it without downloading.

## Startup and jobs

The image runs `bootstrap/start.sh`. Healthy startup performs no installation,
clone, download, or source build. It verifies the mount, GPU, PyTorch CUDA,
both environments, persistent model, and a real gsplat CUDA rasterization.

Run long work with `bootstrap/run_gpu_job.sh`; it writes persistent log, PID,
start time, and exit code, and survives SSH disconnection.

Build/publish outside RunPod:

```bash
docker build -t scene-reconstruction-gpu:v3-pt24-cu124-gsplat153 .
docker push <registry>/scene-reconstruction-gpu:v3-pt24-cu124-gsplat153
```

After the first healthy GPU validation, save complete freezes under
`/workspace/manifests/` and review them locally before updating the checked-in
working freeze.
