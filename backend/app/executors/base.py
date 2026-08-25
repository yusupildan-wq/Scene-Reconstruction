from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

ProgressReporter = Callable[[str, int, str], None]


@dataclass(frozen=True)
class ProviderCapability:
    available: bool
    detail: str
    vram_gb: float | None = None


@dataclass(frozen=True)
class ExecutionRequest:
    job_id: str
    scene_dir: Path
    result_dir: Path
    quality_profile: str


@dataclass(frozen=True)
class ExecutionResult:
    scene_ply: Path
    cameras_json: Path
    geometry_archive: Path
    metrics_json: Path
    provider_job_id: str | None = None


class Executor(Protocol):
    def validate(self) -> ProviderCapability: ...
    def execute(self, request: ExecutionRequest, report: ProgressReporter) -> ExecutionResult: ...
