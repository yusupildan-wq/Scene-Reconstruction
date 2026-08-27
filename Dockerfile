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
      python3.10 python3.10-dev python3.10-venv python3-pip git ca-certificates ffmpeg openssh-server \
      build-essential ninja-build libgl1 libglib2.0-0 && \
    mkdir -p /run/sshd && \
    rm -rf /var/lib/apt/lists/*

# --no-cache-dir everywhere below: PIP_CACHE_DIR is set as an image-wide ENV above,
# so every pip install here would otherwise leave a second, unpacked copy of every
# downloaded wheel sitting in /workspace/cache/pip -- measured at 2.4GB of pure waste
# on the current published image, never read by anything at runtime. This changes
# nothing about which packages or versions get installed.
RUN python3.10 -m pip install --no-cache-dir --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1" && \
    python3.10 -m pip install --no-cache-dir \
      torch==2.4.1 torchvision==0.19.1 \
      --index-url https://download.pytorch.org/whl/cu124

# .git metadata (history, packed objects) is measured at 230MB combined across both
# clones and is never read after checkout -- vggt is later pip installed with -e
# (editable), which imports directly from this directory's *source files*, so only
# .git itself is safe to remove here, not the checked-out tree.
RUN git clone https://github.com/facebookresearch/vggt.git /opt/vggt && \
    git -C /opt/vggt checkout "$VGGT_COMMIT" && \
    rm -rf /opt/vggt/.git && \
    git clone https://github.com/nerfstudio-project/gsplat.git /opt/gsplat && \
    git -C /opt/gsplat checkout "$GSPLAT_COMMIT" && \
    rm -rf /opt/gsplat/.git

COPY bootstrap/requirements-vggt.txt /opt/bootstrap/requirements-vggt.txt
COPY bootstrap/requirements-gsplat.txt /opt/bootstrap/requirements-gsplat.txt

RUN python3.10 -m venv --system-site-packages /opt/venvs/vggt && \
    /opt/venvs/vggt/bin/pip install --no-cache-dir -r /opt/bootstrap/requirements-vggt.txt && \
    /opt/venvs/vggt/bin/pip install --no-cache-dir --no-deps -e /opt/vggt && \
    python3.10 -m venv --system-site-packages /opt/venvs/gsplat && \
    /opt/venvs/gsplat/bin/pip install --no-cache-dir -r /opt/bootstrap/requirements-gsplat.txt && \
    /opt/venvs/gsplat/bin/pip install --no-cache-dir --no-deps \
      "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"

COPY bootstrap /opt/project/bootstrap
COPY scripts/execute_v3_workspace.py scripts/cache_vggt_checkpoint.py /opt/project/scripts/
COPY experiments/run_v3_vggt.py experiments/export_gsplat_cameras.py /opt/project/experiments/
RUN chmod +x /opt/project/bootstrap/*.sh
WORKDIR /opt/project
CMD ["/opt/project/bootstrap/entrypoint.sh"]
