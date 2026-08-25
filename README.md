# Scene Reconstruction

Turn a room video into a local interactive Gaussian Splat scene using the existing V3 VGGT + gsplat pipeline.

## Run it

```powershell
Copy-Item backend/.env.example backend/.env
# Add your RUNPOD_API_KEY to backend/.env
docker compose -f infra/docker-compose.yml up --build
```

Open http://localhost:5173, choose a compute mode, and drop in a video. Uploads, frames, geometry, PLYs, camera metadata, and job state remain under `backend/data` and the local PostgreSQL volume.

## Compute modes

- **RunPod:** provisions a temporary GPU pod, transfers the prepared scene over SSH, runs V3, downloads the results, and terminates the pod automatically. No S3/R2 or Serverless endpoint is required.
- **Local NVIDIA GPU:** validates the NVIDIA driver, CUDA-enabled PyTorch, VRAM, VGGT, and gsplat before running the exact same workspace command locally. Configure the local paths in `backend/.env` and run the backend natively so it can access the host GPU.

RunPod is used only for temporary CUDA compute. See [RUNPOD_SETUP.md](RUNPOD_SETUP.md) for configuration and safety details.
