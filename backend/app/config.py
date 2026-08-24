from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://scene:scene@localhost:5432/scene"

    # "local" stores uploads/artifacts on disk under storage_local_path -- only
    # reachable by processes on this machine. "s3" stores them in an S3-compatible
    # bucket (e.g. Cloudflare R2) reachable by the remote GPU worker. The GPU worker
    # can never use "local" storage since it runs on a different machine (RunPod).
    storage_backend: str = "local"
    storage_local_path: str = "./data/storage"

    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "auto"

    runpod_api_key: str | None = None
    runpod_endpoint_id: str | None = None
    gpu_backend: str = "local"  # local simulates GPU completion without paid compute
    local_demo_scene_ply: str = "../frontend/public/v3_scene.ply"
    local_demo_cameras_json: str = "../frontend/public/v3_scene_cameras.json"
    runpod_poll_seconds: float = 3.0
    reconstruction_quality_profile: str = "high"


settings = Settings()
