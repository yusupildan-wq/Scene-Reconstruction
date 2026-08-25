from __future__ import annotations

import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import requests

from app.config import settings
from app.executors.base import ExecutionRequest, ExecutionResult, ProviderCapability, ProgressReporter
from app.executors.process import run_streaming

API = "https://rest.runpod.io/v1"


class RunPodExecutor:
    def __init__(self, on_pod_created=None):
        self.on_pod_created = on_pod_created

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {settings.runpod_api_key}", "Content-Type": "application/json"}

    def validate(self) -> ProviderCapability:
        if not settings.runpod_api_key:
            return ProviderCapability(False, "Set RUNPOD_API_KEY in backend/.env to use RunPod.")
        if not shutil.which("ssh") or not shutil.which("scp") or not shutil.which("ssh-keygen"):
            return ProviderCapability(False, "OpenSSH client tools (ssh, scp, ssh-keygen) are required.")
        try:
            response = requests.get(f"{API}/pods", headers=self.headers, timeout=20)
            response.raise_for_status()
            return ProviderCapability(True, "RunPod is configured; a temporary pod will be created only after upload.")
        except requests.RequestException as error:
            return ProviderCapability(False, f"RunPod API validation failed: {error}")

    def _ensure_key(self) -> tuple[Path, str]:
        private_key = Path(settings.ssh_private_key_path).resolve()
        public_key = private_key.with_suffix(private_key.suffix + ".pub")
        private_key.parent.mkdir(parents=True, exist_ok=True)
        if not private_key.is_file() or not public_key.is_file():
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)], check=True)
        return private_key, public_key.read_text(encoding="utf-8").strip()

    def _create_pod(self, public_key: str, job_id: str) -> dict:
        body = {
            "name": f"scene-reconstruction-{job_id[:8]}", "imageName": settings.runpod_image,
            "computeType": "GPU", "gpuCount": 1,
            "gpuTypeIds": [item.strip() for item in settings.runpod_gpu_type_ids.split(",") if item.strip()],
            "gpuTypePriority": "availability", "cloudType": settings.runpod_cloud_type,
            "containerDiskInGb": settings.runpod_container_disk_gb, "volumeInGb": settings.runpod_volume_gb,
            "ports": ["22/tcp"], "supportPublicIp": True,
            "env": {"PUBLIC_KEY": public_key, "AUTOMATED_JOB": "1"},
        }
        response = requests.post(f"{API}/pods", headers=self.headers, json=body, timeout=60)
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"RunPod could not provision a GPU: {detail}")
        return response.json()

    def _get_pod(self, pod_id: str) -> dict:
        response = requests.get(f"{API}/pods/{pod_id}", headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def _terminate(self, pod_id: str) -> None:
        response = requests.delete(f"{API}/pods/{pod_id}", headers=self.headers, timeout=30)
        if response.status_code not in {200, 202, 204, 404}:
            response.raise_for_status()

    def terminate_existing(self, pod_id: str) -> None:
        """Best-effort cleanup used when the local backend resumes after a crash."""
        self._terminate(pod_id)

    def _wait_for_ssh(self, pod_id: str, private_key: Path, known_hosts: Path, report: ProgressReporter) -> tuple[str, int]:
        deadline = time.monotonic() + settings.runpod_startup_timeout_seconds
        while time.monotonic() < deadline:
            pod = self._get_pod(pod_id)
            ip = pod.get("publicIp")
            mappings = pod.get("portMappings") or {}
            port = mappings.get("22") or mappings.get(22)
            if ip and port:
                command = self._ssh_base(str(ip), int(port), private_key, known_hosts) + ["true"]
                if subprocess.run(command, capture_output=True, timeout=20).returncode == 0:
                    return str(ip), int(port)
            report("vggt_geometry", 32, "Waiting for temporary RunPod GPU")
            time.sleep(5)
        raise TimeoutError("RunPod did not become reachable before the startup timeout")

    @staticmethod
    def _ssh_base(host: str, port: int, key: Path, known_hosts: Path) -> list[str]:
        return ["ssh", "-p", str(port), "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}", f"root@{host}"]

    @staticmethod
    def _scp_base(port: int, key: Path, known_hosts: Path) -> list[str]:
        return ["scp", "-P", str(port), "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}"]

    def execute(self, request: ExecutionRequest, report: ProgressReporter) -> ExecutionResult:
        capability = self.validate()
        if not capability.available:
            raise RuntimeError(capability.detail)
        private_key, public_key = self._ensure_key()
        transfer_dir = request.result_dir.parent
        known_hosts = transfer_dir / "runpod_known_hosts"
        input_archive = transfer_dir / "runpod_input.tar.gz"
        with tarfile.open(input_archive, "w:gz") as archive:
            archive.add(request.scene_dir, arcname="scene")

        pod_id: str | None = None
        try:
            report("vggt_geometry", 30, "Provisioning temporary RunPod GPU")
            pod = self._create_pod(public_key, request.job_id)
            pod_id = str(pod["id"])
            if self.on_pod_created:
                self.on_pod_created(pod_id)
            host, port = self._wait_for_ssh(pod_id, private_key, known_hosts, report)
            report("vggt_geometry", 33, "Transferring prepared frames to RunPod")
            subprocess.run(self._scp_base(port, private_key, known_hosts) + [str(input_archive), f"root@{host}:/workspace/input.tar.gz"], check=True)

            remote = (
                "rm -rf /workspace/job && mkdir -p /workspace/job && "
                "tar -xzf /workspace/input.tar.gz -C /workspace/job && "
                "/opt/venvs/vggt/bin/python /opt/project/scripts/cache_vggt_checkpoint.py "
                "--output /workspace/cache/torch/hub/checkpoints/model.pt && "
                "/opt/project/bootstrap/verify_env.sh && "
                "/opt/venvs/gsplat/bin/python /opt/project/scripts/execute_v3_workspace.py "
                "--project-root /opt/project --scene-dir /workspace/job/scene --result-dir /workspace/job/result "
                "--vggt-root /opt/vggt --gsplat-root /opt/gsplat --vggt-python /opt/venvs/vggt/bin/python "
                f"--gsplat-python /opt/venvs/gsplat/bin/python --quality-profile {request.quality_profile} && "
                "tar -czf /workspace/output.tar.gz -C /workspace/job/result "
                "scene.ply scene_cameras.json vggt_geometry.tar.gz metrics.json"
            )
            report("vggt_geometry", 34, "Preparing the remote VGGT model")
            try:
                run_streaming(self._ssh_base(host, port, private_key, known_hosts) + [remote], Path.cwd(), report)
            except Exception:
                # Preserve completed VGGT geometry even when gsplat fails so a
                # retry does not pay for geometry a second time.
                request.result_dir.mkdir(parents=True, exist_ok=True)
                geometry = request.result_dir / "vggt_geometry.tar.gz"
                subprocess.run(
                    self._ssh_base(host, port, private_key, known_hosts) +
                    ["tar -czf /workspace/geometry-on-failure.tar.gz -C /workspace/job/scene sparse"],
                    capture_output=True,
                )
                subprocess.run(
                    self._scp_base(port, private_key, known_hosts) +
                    [f"root@{host}:/workspace/geometry-on-failure.tar.gz", str(geometry)],
                    capture_output=True,
                )
                raise
            report("finalizing", 96, "Downloading completed scene")
            output_archive = transfer_dir / "runpod_output.tar.gz"
            subprocess.run(self._scp_base(port, private_key, known_hosts) + [f"root@{host}:/workspace/output.tar.gz", str(output_archive)], check=True)
            request.result_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(output_archive, "r:gz") as archive:
                archive.extractall(request.result_dir)
            return ExecutionResult(
                request.result_dir / "scene.ply", request.result_dir / "scene_cameras.json",
                request.result_dir / "vggt_geometry.tar.gz", request.result_dir / "metrics.json", pod_id,
            )
        finally:
            if pod_id:
                try:
                    report("finalizing", 99, "Terminating temporary RunPod GPU")
                    self._terminate(pod_id)
                except Exception as cleanup_error:
                    raise RuntimeError(f"RunPod cleanup failed for pod {pod_id}: {cleanup_error}") from cleanup_error
