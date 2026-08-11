# GPU worker (not implemented yet)

Runs on RunPod. Given a video already uploaded to S3-compatible storage, this
container is meant to:

1. Download the video (`runner.py` receives the storage URL in the job payload).
2. Run COLMAP: feature extraction -> feature matching -> sparse SfM (poses + sparse
   point cloud) -> optionally dense MVS.
3. Run Gaussian Splatting training using the SfM output as initialization, against
   the original frames as supervision (photometric loss).
4. Upload the trained scene + evaluation metrics (held-out PSNR/SSIM/LPIPS, training
   time, GPU memory) back to storage.
5. Report status back so the backend can update the Job row (via RunPod's job
   result payload, which the backend polls -- see `backend/app/dispatch.py`).

## Why this isn't built yet

Needs a RunPod account + API key (manual signup, not something that can be
automated here) and a decision on which Gaussian Splatting implementation to build
on (leaning INRIA reference / `gsplat` -- not committed yet). `runner.py` is a
skeleton showing the intended shape of the pipeline; every stage currently raises
`NotImplementedError`.

## Base image plan

CUDA-enabled base (`nvidia/cuda:12.x-devel`) with COLMAP built/installed and the
chosen Gaussian Splatting implementation's Python + CUDA extension dependencies.
Left unpinned until the RunPod GPU type (and its CUDA version) is chosen, since the
CUDA toolkit version in the image must match what's available on the pod.
