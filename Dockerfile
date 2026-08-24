FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG VGGT_COMMIT=a288dd0f14786c93483e45524328726ab7b1b4ce
ARG GSPLAT_COMMIT=937e29912570c372bed6747a5c9bf85fed877bae

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    WORKSPACE_ROOT=/workspace \
    MODEL_ROOT=/workspace/models \
    DATASET_ROOT=/workspace/datasets \
    EXPERIMENT_ROOT=/workspace/experiments \
    CHECKPOINT_ROOT=/workspace/checkpoints \
    HF_HOME=/workspace/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface/hub \
    TORCH_HOME=/workspace/cache/torch \
    TORCH_EXTENSIONS_DIR=/workspace/cache/torch_extensions \
    PIP_CACHE_DIR=/workspace/cache/pip

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv python3-pip git ca-certificates ffmpeg \
      build-essential ninja-build libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1" && \
    python3.10 -m pip install \
      torch==2.4.1 torchvision==0.19.1 \
      --index-url https://download.pytorch.org/whl/cu124

RUN git clone https://github.com/facebookresearch/vggt.git /opt/vggt && \
    git -C /opt/vggt checkout "$VGGT_COMMIT" && \
    git clone https://github.com/nerfstudio-project/gsplat.git /opt/gsplat && \
    git -C /opt/gsplat checkout "$GSPLAT_COMMIT"

COPY bootstrap/requirements-vggt.txt /opt/bootstrap/requirements-vggt.txt
COPY bootstrap/requirements-gsplat.txt /opt/bootstrap/requirements-gsplat.txt

RUN python3.10 -m venv --system-site-packages /opt/venvs/vggt && \
    /opt/venvs/vggt/bin/pip install -r /opt/bootstrap/requirements-vggt.txt && \
    /opt/venvs/vggt/bin/pip install --no-deps -e /opt/vggt && \
    python3.10 -m venv --system-site-packages /opt/venvs/gsplat && \
    /opt/venvs/gsplat/bin/pip install -r /opt/bootstrap/requirements-gsplat.txt && \
    /opt/venvs/gsplat/bin/pip install --no-deps \
      "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"

COPY . /opt/project
RUN chmod +x /opt/project/bootstrap/*.sh
WORKDIR /opt/project
CMD ["bash", "/opt/project/bootstrap/start.sh"]
