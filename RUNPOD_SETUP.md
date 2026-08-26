# RunPod mode

RunPod is temporary GPU compute only. All permanent files remain in `backend/data`.

1. Copy `backend/.env.example` to `backend/.env`.
2. Set `RUNPOD_API_KEY` to a RunPod key with pod read/write permissions.
3. Start the application with Docker Compose and choose **RunPod** before uploading.

For each job the backend:

1. Selects frames locally.
2. Generates a dedicated SSH key locally.
3. Creates one temporary pod from the pinned GPU image.
4. Transfers the prepared scene directly with SCP.
5. Runs the shared V3 workspace runner remotely.
6. Downloads the PLY, cameras, metrics, and reusable VGGT geometry.
7. Terminates the pod in a `finally` cleanup path.

No Serverless endpoint, network volume, S3/R2 bucket, public backend, or manual SSH work is required. A terminated process or machine shutdown can prevent cleanup; if that happens, delete the pod shown in the job record from the RunPod console.

The pinned GPU image keeps the CUDA/VGGT/gsplat runtime compact. The VGGT checkpoint is verified and downloaded after boot into the temporary workspace; baking the 5 GB checkpoint into the image was tested and rejected because a cold RunPod host could not finish pulling the oversized image within 40 minutes.

The default remote GPU is the 24 GB RTX 4090 used by the golden IMG_1705 run. CUDA readiness is polled for up to five minutes after container boot because RunPod can expose NVML before the CUDA driver accepts PyTorch contexts.
