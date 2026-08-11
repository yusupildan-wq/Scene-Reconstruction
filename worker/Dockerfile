# UNVERIFIED -- cannot build or run this locally (no CUDA GPU on the dev machine),
# so this has never actually been built. Written to be correct based on what we
# validated works outside Docker (pycolmap, gsplat's real API), but treat as a
# first draft until it's actually built and run on a RunPod GPU. CUDA version
# should match whatever GPU type gets chosen on RunPod.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker

# pycolmap ships COLMAP itself as a prebuilt wheel -- no need to build COLMAP or
# its dependencies (Ceres Solver, Eigen, ...) from source, unlike the original
# plan when this Dockerfile was first stubbed out.
# --index-url replaces the DEFAULT index for the whole command, not just for the
# package it's next to -- torch needs PyTorch's CUDA-specific index, everything
# else needs regular PyPI, so these have to be two separate installs.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir pycolmap gsplat numpy scipy requests pillow runpod

COPY runner.py ./

CMD ["python3", "runner.py"]
