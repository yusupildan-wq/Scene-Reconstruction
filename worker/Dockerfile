# STUB -- not buildable yet. CUDA version is left unpinned until the RunPod GPU
# type is chosen, since it must match the pod's driver. See README.md.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git cmake build-essential \
    # TODO: COLMAP build dependencies (Ceres Solver, Eigen, etc.) or install a
    # prebuilt COLMAP binary instead of building from source.
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker
COPY runner.py ./
RUN pip3 install --no-cache-dir runpod
# TODO: install the chosen Gaussian Splatting implementation and its CUDA extensions.

CMD ["python3", "runner.py"]
