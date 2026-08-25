"""Local permanent artifact storage used by every compute provider."""
from __future__ import annotations

import shutil
from pathlib import Path

from app.config import settings


class LocalStorage:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        path = self.base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, key: str, data: bytes) -> None:
        self.path(key).write_bytes(data)

    def save_fileobj(self, key: str, fileobj) -> None:
        with self.path(key).open("wb") as target:
            shutil.copyfileobj(fileobj, target, length=1024 * 1024)

    def read(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()


def get_storage() -> LocalStorage:
    return LocalStorage(settings.storage_local_path)
