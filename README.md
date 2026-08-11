# Scene Reconstruction — AI 2D/Video → Interactive 3D Environment

Turns ordinary phone video / multi-image captures into an interactively explorable 3D
scene, using classical Structure-from-Motion for camera/geometry recovery and 3D
Gaussian Splatting for the renderable representation.

## V0 pipeline

```
PHONE VIDEO
    -> upload (backend, local disk / S3-compatible storage)
    -> Job row created in Postgres (status = pending)
    -> frame extraction + blur/redundancy filtering        [classical CV, runs on backend host, CPU]
    -> dispatch to GPU worker (RunPod)                      [infra]
    -> COLMAP: feature matching, SfM (poses + sparse cloud) [classical CV / geometry, GPU-accelerated matching]
    -> Gaussian Splatting training (per-scene optimization) [gradient-based optimization, NOT a generalizing
                                                               trained network -- see "ML classification" below]
    -> trained scene artifact uploaded to storage
    -> Job row updated (status = complete)
    -> frontend polls job status, loads WebGL Gaussian Splat viewer
```

## Why this stack

- **No local CUDA GPU** (dev machine has an AMD Radeon 880M, not NVIDIA) — COLMAP's GPU
  matching and Gaussian Splatting training are CUDA-only in every practical
  implementation, so the GPU stage *must* run remotely. This is a hard constraint, not
  a preference, and it shapes the whole job architecture: the backend and GPU worker
  are separate services from day one, not a "V2 split out the worker" refactor later.
- **RunPod (on-demand rental)**, not a persistent cloud VM — training a scene takes
  minutes, so pay-per-second serverless GPU execution fits usage much better than an
  always-on instance.
- **Storage is behind an interface** (`backend/app/storage.py`) with a local-disk
  implementation for dev and an S3-compatible implementation for anything the RunPod
  worker needs to read/write — this isn't speculative abstraction, it's required
  because the worker runs on a different machine than local dev storage.
- **FastAPI + Postgres + Docker + React**, matching prior project experience — moved
  through quickly, no re-explanation of the basics. New parts of these tools (async
  job lifecycles, GPU container builds) get taught when we hit them.

## Component classification

| Component | Category | Why |
|---|---|---|
| Feature matching, SfM, bundle adjustment (COLMAP) | Classical CV / geometry | Deterministic geometric optimization (nonlinear least squares), no learned generalizing weights |
| Gaussian Splatting training | Gradient-based optimization, per-scene | Uses backprop + Adam like ML training, but "overfits" to one scene by design — not a trained model that generalizes to new scenes |
| Frame filtering (blur/redundancy detection) | Classical CV | Deterministic heuristics on pixel statistics |
| Job orchestration, upload, storage, DB | Backend engineering | Standard SWE |
| GPU worker container, RunPod dispatch | Infra | Deployment/execution environment, not an algorithmic contribution |
| Gaussian Splat WebGL viewer | Graphics | Real-time rendering of the trained representation |

No component in V0 is a pretrained neural network used for inference — that enters in
V1/V2 (monocular depth, feed-forward reconstruction) and will be labeled explicitly
when it does, per the categories: trained-from-scratch / fine-tuned / pretrained-as-is
/ external API.

## Repo layout

```
backend/   FastAPI app, Postgres models, storage abstraction, job API
worker/    GPU worker: COLMAP + Gaussian Splatting training, built for RunPod
frontend/  React + Vite upload UI, job status, WebGL viewer
infra/     docker-compose for local dev (postgres + backend + frontend)
```

## Status

V0 in progress. Backend job/project model and local dev environment are real and
runnable. The GPU worker (`worker/`) is scaffolded but **not yet implemented** — it
requires a RunPod account/API key, which is a manual step outside this repo. Job
dispatch currently raises `NotImplementedError` at that boundary rather than faking
a result.
