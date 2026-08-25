# Scene Reconstruction

Turn a room video into an interactive 3D Gaussian Splat scene.

## Pipeline

```text
video upload → frame selection → VGGT geometry → gsplat optimization → WebGL viewer
```

The React frontend provides upload, preview, progress, retry, and scene viewing. The FastAPI backend stores resumable jobs and artifacts. RunPod is used only for CUDA-required VGGT and gsplat work; development and preprocessing run locally.

## Local development

```bash
docker compose -f infra/docker-compose.yml up --build
```

- Frontend: http://localhost:5173
- API: http://localhost:8000
- Experiments: http://localhost:5173/?experiments=1

Local mode uses the existing V3 reconstruction and does not start paid GPU compute.

## Repository

```text
frontend/     React upload flow and Gaussian viewer
backend/      FastAPI jobs, storage, and orchestration
experiments/  V1/V2/V3 reconstruction experiments
bootstrap/    Pinned GPU environment and verification
scripts/      Local preparation and RunPod preflight
```

See [RUNPOD_SETUP.md](RUNPOD_SETUP.md) for GPU execution and [experiments/V3_VGGT.md](experiments/V3_VGGT.md) for the current pipeline.
