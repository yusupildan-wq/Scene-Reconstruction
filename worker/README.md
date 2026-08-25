# GPU worker

`v3_serverless.py` is the RunPod Serverless adapter for the existing V3 pipeline.
It downloads locally prepared frames, runs the unchanged VGGT and gsplat scripts,
uploads the PLY/camera metadata, and reports progress to the API. VGGT geometry is
stored separately and reused when a retry only needs to repeat training.

The pinned root `Dockerfile` supports both modes:

- `MODE_TO_RUN=serverless`: automatic product jobs through the RunPod endpoint.
- `MODE_TO_RUN=pod` (default): the existing interactive pod workflow.

RunPod is only used for CUDA work. Upload handling, frame selection, job state,
artifact storage, and the viewer stay in the application stack.
