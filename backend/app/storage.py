"""Storage abstraction.

Two implementations exist because two different machines need access to the same
files: local dev (this machine) and the GPU worker (a RunPod instance). "local"
writes to disk and is only valid when the reader is a process on this machine --
it cannot be used once a job is dispatched to the remote worker. "s3" writes to an
S3-compatible bucket (e.g. Cloudflare R2) reachable from anywhere, which is what
production job dispatch requires.
"""

from __future__ import annotations

import abc
from pathlib import Path

import boto3

from app.config import settings


class Storage(abc.ABC):
    @abc.abstractmethod
    def save(self, key: str, data: bytes) -> None: ...

    @abc.abstractmethod
    def read(self, key: str) -> bytes: ...

    @abc.abstractmethod
    def url_for(self, key: str) -> str:
        """A URL the GPU worker (a different machine) can use to fetch/put this key.

        Raises for the local backend, since a local path is not reachable remotely.
        """


class LocalStorage(Storage):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = self.base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, key: str, data: bytes) -> None:
        self._path(key).write_bytes(data)

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def url_for(self, key: str) -> str:
        raise RuntimeError(
            "LocalStorage keys are not reachable by a remote GPU worker. "
            "Switch STORAGE_BACKEND=s3 before dispatching jobs to RunPod."
        )


class S3Storage(Storage):
    def __init__(self):
        if not (settings.s3_bucket and settings.s3_access_key_id and settings.s3_secret_access_key):
            raise RuntimeError("S3 storage requested but S3 credentials are not configured")
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
        )

    def save(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def read(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def url_for(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=3600
        )


def get_storage() -> Storage:
    if settings.storage_backend == "s3":
        return S3Storage()
    return LocalStorage(settings.storage_local_path)
