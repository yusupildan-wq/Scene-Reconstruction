from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.config import settings
from app.executors.base import ExecutionRequest, ExecutionResult, ProviderCapability, ProgressReporter
from app.executors.process import run_streaming


class LocalNvidiaExecutor:
    def _paths(self) -> tuple[Path, Path, Path, Path, Path]:
        values = (
            settings.local_project_root, settings.local_vggt_root, settings.local_gsplat_root,
            settings.local_vggt_python, settings.local_gsplat_python,
        )
        return tuple(Path(value).resolve() for value in values)  # type: ignore[return-value]

    def validate(self) -> ProviderCapability:
        project, vggt, gsplat, vggt_python, gsplat_python = self._paths()
        if not shutil.which("nvidia-smi"):
            return ProviderCapability(False, "NVIDIA driver tools were not found. Use RunPod.")
        required = (project / "experiments" / "run_v3_vggt.py", vggt / "demo_colmap.py",
                    gsplat / "examples" / "simple_trainer.py", vggt_python, gsplat_python)
        if any(not path.exists() for path in required):
            return ProviderCapability(False, "Local VGGT/gsplat is incomplete. Use RunPod or configure the local paths.")
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=15,
            )
            vram = max(float(line.strip()) for line in query.stdout.splitlines() if line.strip()) / 1024
            check = subprocess.run([
                str(vggt_python), "-c",
                "import json,torch,vggt; print(json.dumps({'cuda':torch.cuda.is_available(),'version':torch.version.cuda}))",
            ], capture_output=True, text=True, check=True, timeout=30)
            torch_state = json.loads(check.stdout.splitlines()[-1])
            subprocess.run([str(gsplat_python), "-c", "import torch,gsplat; assert torch.cuda.is_available()"],
                           capture_output=True, text=True, check=True, timeout=30)
            if not torch_state["cuda"]:
                return ProviderCapability(False, "PyTorch cannot access CUDA. Use RunPod.", vram)
            if not torch_state.get("version"):
                return ProviderCapability(False, "PyTorch has no CUDA runtime. Use RunPod.", vram)
            if vram < settings.minimum_vram_gb:
                return ProviderCapability(False, f"Local GPU has {vram:.0f} GB VRAM; {settings.minimum_vram_gb:.0f} GB is required. Use RunPod.", vram)
            return ProviderCapability(True, f"Local CUDA {torch_state['version']} is ready ({vram:.0f} GB VRAM).", vram)
        except (subprocess.SubprocessError, ValueError, json.JSONDecodeError) as error:
            return ProviderCapability(False, f"Local CUDA validation failed: {error}. Use RunPod.")

    def execute(self, request: ExecutionRequest, report: ProgressReporter) -> ExecutionResult:
        capability = self.validate()
        if not capability.available:
            raise RuntimeError(capability.detail)
        project, vggt, gsplat, vggt_python, gsplat_python = self._paths()
        command = [
            sys.executable, str(project / "scripts" / "execute_v3_workspace.py"),
            "--project-root", str(project), "--scene-dir", str(request.scene_dir),
            "--result-dir", str(request.result_dir), "--vggt-root", str(vggt),
            "--gsplat-root", str(gsplat), "--vggt-python", str(vggt_python),
            "--gsplat-python", str(gsplat_python), "--quality-profile", request.quality_profile,
        ]
        run_streaming(command, project, report)
        return ExecutionResult(
            request.result_dir / "scene.ply", request.result_dir / "scene_cameras.json",
            request.result_dir / "vggt_geometry.tar.gz", request.result_dir / "metrics.json",
        )
