"""Local permanent artifact storage used by every compute provider."""
from __future__ import annotations

import shutil
from pathlib import Path

from app.config import settings


class LocalStorage:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = self.base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, key: str, data: bytes) -> None:
        self._path(key).write_bytes(data)

    def save_fileobj(self, key: str, fileobj) -> None:
        with self._path(key).open("wb") as target:
            shutil.copyfileobj(fileobj, target, length=1024 * 1024)

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def iter_bytes(self, key: str, chunk_size: int = 1024 * 1024):
        with self._path(key).open("rb") as source:
            while chunk := source.read(chunk_size):
                yield chunk


def get_storage() -> LocalStorage:
    if settings.storage_backend != "local":
        raise RuntimeError("Only local permanent storage is supported")
    return LocalStorage(settings.storage_local_path)
