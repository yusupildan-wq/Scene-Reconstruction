from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://scene:scene@localhost:5432/scene"
    storage_local_path: str = "./data/storage"

    runpod_api_key: str | None = None
    runpod_image: str = "ghcr.io/yusupildan-wq/scene-reconstruction-gpu:v3-pt24-cu124-gsplat153"
    # RTX 4090 first (the golden GPU); same-quality (>=24GB, Ampere/Ada) fallbacks
    # let RunPod provision a working GPU when 4090 Community Cloud capacity is briefly
    # exhausted, instead of failing pod creation outright.
    runpod_gpu_type_ids: str = "NVIDIA GeForce RTX 4090,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A6000"
    runpod_cloud_type: str = "COMMUNITY"
    # A healthy compact image normally exposes its network within a few minutes.
    # Do not pay for a bad Community Cloud host for forty minutes.
    runpod_startup_timeout_seconds: int = 600
    runpod_ssh_ready_timeout_seconds: int = 300
    runpod_poll_interval_seconds: int = 8
    # Observed twice in a row: SSH drops right after the reconstruction pipeline
    # launches (tar extraction + checkpoint download + two venvs starting up all at
    # once) and stays unreachable for the *entire* 120s grace period, both times --
    # not a brief blip. That's consistent with the host being transiently overloaded
    # by its own launch, not a dead pod. 300s gives it real room to come back before
    # we pay to kill and lose the run; re-reading the pod's current IP/port on each
    # failed attempt (see _await_reconstruction) covers the case where RunPod also
    # reassigns the endpoint underneath us.
    runpod_reconnect_grace_seconds: int = 300
    runpod_container_disk_gb: int = 80
    runpod_volume_gb: int = 20
    ssh_private_key_path: str = "./data/runpod_ed25519"

    local_project_root: str = ".."
    local_vggt_root: str = "../external/vggt"
    local_gsplat_root: str = "../external/gsplat"
    local_vggt_python: str = "../.venvs/vggt/Scripts/python.exe"
    local_gsplat_python: str = "../.venvs/gsplat/Scripts/python.exe"
    minimum_vram_gb: float = 24.0
    reconstruction_quality_profile: str = "high"


settings = Settings()
