from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://scene:scene@localhost:5432/scene"
    storage_local_path: str = "./data/storage"

    runpod_api_key: str | None = None
    runpod_image: str = "ghcr.io/yusupildan-wq/scene-reconstruction-gpu:v3-pt24-cu124-gsplat153"
    runpod_gpu_type_ids: str = "NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A6000,NVIDIA GeForce RTX 4090"
    runpod_cloud_type: str = "COMMUNITY"
    runpod_startup_timeout_seconds: int = 1200
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
