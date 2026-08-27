from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests

from app.config import settings
from app.executors.base import ExecutionRequest, ExecutionResult, ProviderCapability, ProgressReporter
from app.executors.process import PREFIX

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

    # RunPod's `desiredStatus` reports what the pod is meant to be doing, not
    # whether its container has actually finished starting: it stays RUNNING for
    # the whole scheduling + image-pull + boot sequence, and only flips to one of
    # these two once the pod has actually died. That makes it the signal for a
    # real provisioning failure, distinct from a pod that is still legitimately
    # pulling a multi-gigabyte CUDA image.
    _FAILED_POD_STATUSES = {"EXITED", "TERMINATED"}

    # Paths written by bootstrap/run_gpu_job.sh for the detached "reconstruct" job.
    _REMOTE_LOG = "/workspace/logs/reconstruct.log"
    _REMOTE_EXIT = "/workspace/bootstrap/state/jobs/reconstruct.exit"
    _EXIT_MARKER = "---RECONSTRUCT_EXIT---"

    def _wait_for_ssh(self, pod_id: str, private_key: Path, known_hosts: Path, report: ProgressReporter) -> tuple[str, int]:
        provisioning_deadline = time.monotonic() + settings.runpod_startup_timeout_seconds
        ssh_deadline: float | None = None
        network_ready = False
        while True:
            now = time.monotonic()
            if not network_ready and now >= provisioning_deadline:
                raise TimeoutError(
                    "RunPod pod did not finish provisioning (scheduling or image pull) within "
                    f"{settings.runpod_startup_timeout_seconds}s"
                )
            if network_ready and now >= ssh_deadline:
                raise TimeoutError(
                    "RunPod pod network became reachable but SSH never came up within "
                    f"{settings.runpod_ssh_ready_timeout_seconds}s; the pod likely failed to start correctly"
                )
            try:
                pod = self._get_pod(pod_id)
            except requests.RequestException as error:
                # RunPod's own API is occasionally flaky (a transient 404/5xx on a pod
                # that is still very much alive), independent of the pod's own boot
                # progress. Treat that as "still waiting", not as proof the pod died --
                # only desiredStatus itself is a real failure signal.
                report("vggt_geometry", 32, f"RunPod status check failed transiently ({error}); retrying")
                time.sleep(5)
                continue
            status = pod.get("desiredStatus")
            if status in self._FAILED_POD_STATUSES:
                raise RuntimeError(f"RunPod pod exited during startup (status={status}) before it became reachable")
            ip = pod.get("publicIp")
            mappings = pod.get("portMappings") or {}
            port = mappings.get("22") or mappings.get(22)
            if ip and port:
                if not network_ready:
                    network_ready = True
                    ssh_deadline = time.monotonic() + settings.runpod_ssh_ready_timeout_seconds
                command = self._ssh_base(str(ip), int(port), private_key, known_hosts) + ["true"]
                if subprocess.run(command, capture_output=True, timeout=20).returncode == 0:
                    return str(ip), int(port)
                report("vggt_geometry", 32, "Pod network is up; waiting for SSH to come online")
            else:
                report("vggt_geometry", 32, "Waiting for temporary RunPod GPU to finish provisioning (image download can take a while)")
            time.sleep(5)

    # A degrading connection (not a clean drop) can otherwise sit silent for a
    # long time before either side notices; these make the SSH client itself
    # detect and give up on a dead session within ~30s instead of relying on
    # a stall to eventually surface on its own.
    _KEEPALIVE_OPTS = ["-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]

    @staticmethod
    def _ssh_base(host: str, port: int, key: Path, known_hosts: Path) -> list[str]:
        return ["ssh", "-p", str(port), "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                *RunPodExecutor._KEEPALIVE_OPTS,
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}", f"root@{host}"]

    @staticmethod
    def _scp_base(port: int, key: Path, known_hosts: Path) -> list[str]:
        return ["scp", "-P", str(port), "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                *RunPodExecutor._KEEPALIVE_OPTS,
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}"]

    def _launch_reconstruction(self, host: str, port: int, private_key: Path, known_hosts: Path, quality_profile: str) -> None:
        pipeline = (
            "export WORKSPACE_ROOT=/workspace "
            "HF_HOME=/workspace/cache/huggingface "
            "HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface/hub "
            "TORCH_EXTENSIONS_DIR=/workspace/cache/torch_extensions; "
            "if [ -s /opt/cache/torch/hub/checkpoints/model.pt ]; then "
            "export TORCH_HOME=/opt/cache/torch VGGT_CHECKPOINT=/opt/cache/torch/hub/checkpoints/model.pt; "
            "else export TORCH_HOME=/workspace/cache/torch VGGT_CHECKPOINT=/workspace/cache/torch/hub/checkpoints/model.pt; fi; "
            "rm -rf /workspace/job && mkdir -p /workspace/job && "
            "tar -xzf /workspace/input.tar.gz -C /workspace/job && "
            "/opt/venvs/vggt/bin/python /opt/project/scripts/cache_vggt_checkpoint.py "
            "--output $VGGT_CHECKPOINT && "
            "/opt/project/bootstrap/verify_env.sh && "
            "/opt/venvs/gsplat/bin/python /opt/project/scripts/execute_v3_workspace.py "
            "--project-root /opt/project --scene-dir /workspace/job/scene --result-dir /workspace/job/result "
            "--vggt-root /opt/vggt --gsplat-root /opt/gsplat --vggt-python /opt/venvs/vggt/bin/python "
            f"--gsplat-python /opt/venvs/gsplat/bin/python --quality-profile {quality_profile} && "
            "tar -czf /workspace/output.tar.gz -C /workspace/job/result "
            "scene.ply scene_cameras.json vggt_geometry.tar.gz metrics.json"
        )
        # run_gpu_job.sh nohup's this with stdin from /dev/null and writes a PID
        # file, a log file, and an exit-status file on the pod, so the pipeline
        # keeps running (and stays inspectable) even if this SSH session drops.
        # It also no-ops if the same job name is already running, so retrying
        # this launch call itself is safe.
        launch = f"/opt/project/bootstrap/run_gpu_job.sh reconstruct bash -c {shlex.quote(pipeline)}"
        command = self._ssh_base(host, port, private_key, known_hosts) + [launch]
        last_error: Exception | None = None
        for _ in range(3):
            try:
                subprocess.run(command, check=True, timeout=30)
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                last_error = error
                time.sleep(3)
        raise RuntimeError(f"Could not launch the reconstruction job on the RunPod pod: {last_error}")

    def _await_reconstruction(
        self, pod_id: str, host: str, port: int, private_key: Path, known_hosts: Path, report: ProgressReporter,
    ) -> dict[str, float]:
        # The trailing `; true` matters: without it, this remote command's own exit
        # status is whatever the last `cat` returned, and `cat` on the exit-status
        # file returns 1 (No such file) for the entire time the job is still running
        # (the file doesn't exist yet). That made every poll of an actually-healthy,
        # still-running job look identical to a dropped SSH connection -- the "lost
        # connection" path was firing on the very first poll after every launch, not
        # on any real network failure. Whether the job is actually done is decided
        # below from the parsed exit marker/text, not from this command's exit code.
        poll = f"cat {self._REMOTE_LOG} 2>/dev/null; echo {self._EXIT_MARKER}; cat {self._REMOTE_EXIT} 2>/dev/null; true"
        seen_lines = 0
        reconnect_deadline: float | None = None
        monitoring_started = time.monotonic()
        first_pipeline_progress: float | None = None
        last_failure: str | None = None
        while True:
            command = self._ssh_base(host, port, private_key, known_hosts) + [poll]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                connected = result.returncode == 0
                if not connected:
                    last_failure = (result.stderr or "").strip() or f"ssh exited with code {result.returncode}"
            except subprocess.TimeoutExpired:
                connected, result = False, None
                last_failure = "ssh connection attempt timed out after 30s"

            if not connected:
                # SSH itself failed to reach the pod. That alone doesn't tell us
                # whether the pod died or the connection just blipped, so ask
                # RunPod directly before giving up: a real EXITED/TERMINATED pod
                # fails fast here instead of waiting out the reconnect budget,
                # while an ambiguous "can't reach it right now" keeps retrying —
                # the reconstruction itself is unaffected either way, since it
                # runs detached from this SSH session. A transient failure to even
                # query RunPod's API is treated the same as "can't confirm it's dead"
                # rather than propagating and aborting the whole job.
                try:
                    pod = self._get_pod(pod_id)
                except requests.RequestException:
                    pod = {}
                if pod.get("desiredStatus") in self._FAILED_POD_STATUSES:
                    raise RuntimeError(
                        f"RunPod pod exited (status={pod.get('desiredStatus')}) while the reconstruction was running"
                    )
                # RunPod can remap a pod's public IP/SSH port under it (observed: a pod that
                # connected fine once then had every reconnect attempt fail against the same
                # address for the whole grace period). Re-read the pod's current endpoint on
                # every failed attempt instead of hammering a possibly-stale host:port for the
                # entire reconnect window.
                new_ip = pod.get("publicIp")
                new_mappings = pod.get("portMappings") or {}
                new_port = new_mappings.get("22") or new_mappings.get(22)
                if new_ip and new_port and (str(new_ip), int(new_port)) != (host, port):
                    report("vggt_geometry", 34, f"RunPod reassigned the pod's network endpoint; reconnecting to {new_ip}:{new_port}")
                    host, port = str(new_ip), int(new_port)
                if reconnect_deadline is None:
                    reconnect_deadline = time.monotonic() + settings.runpod_reconnect_grace_seconds
                    report("vggt_geometry", 34, "Lost connection to temporary RunPod GPU; reconnecting")
                elif time.monotonic() >= reconnect_deadline:
                    raise RuntimeError(
                        "Lost the SSH connection to the RunPod pod and could not reconnect within "
                        f"{settings.runpod_reconnect_grace_seconds}s; the pod may be unreachable "
                        f"(last error: {last_failure})"
                    )
                time.sleep(settings.runpod_poll_interval_seconds)
                continue

            reconnect_deadline = None
            log_text, _, exit_text = result.stdout.partition(self._EXIT_MARKER)
            lines = log_text.splitlines()
            for line in lines[seen_lines:]:
                print(line, flush=True)
                if line.startswith(PREFIX):
                    update = json.loads(line[len(PREFIX):])
                    if first_pipeline_progress is None:
                        first_pipeline_progress = time.monotonic()
                    report(update["stage"], int(update["progress"]), update["detail"])
            seen_lines = len(lines)

            exit_text = exit_text.strip()
            if exit_text:
                if int(exit_text.splitlines()[0]) != 0:
                    raise RuntimeError("Reconstruction command failed:\n" + "\n".join(lines[-10:]))
                completed = time.monotonic()
                return {
                    "remote_setup_seconds": round((first_pipeline_progress or completed) - monitoring_started, 3),
                    "remote_job_seconds": round(completed - monitoring_started, 3),
                }
            time.sleep(settings.runpod_poll_interval_seconds)

    def execute(self, request: ExecutionRequest, report: ProgressReporter) -> ExecutionResult:
        capability = self.validate()
        if not capability.available:
            raise RuntimeError(capability.detail)
        private_key, public_key = self._ensure_key()
        transfer_dir = request.result_dir.parent
        known_hosts = transfer_dir / "runpod_known_hosts"
        input_archive = transfer_dir / "runpod_input.tar.gz"
        packaging_started = time.monotonic()
        with tarfile.open(input_archive, "w:gz") as archive:
            archive.add(request.scene_dir, arcname="scene")
        timings: dict[str, float] = {
            "local_packaging_seconds": round(time.monotonic() - packaging_started, 3),
        }

        pod_id: str | None = None
        succeeded = False
        execution_result: ExecutionResult | None = None
        paid_started: float | None = None
        try:
            report("vggt_geometry", 30, "Provisioning temporary RunPod GPU")
            provisioning_started = time.monotonic()
            pod = self._create_pod(public_key, request.job_id)
            pod_id = str(pod["id"])
            paid_started = time.monotonic()
            if self.on_pod_created:
                self.on_pod_created(pod_id)
            host, port = self._wait_for_ssh(pod_id, private_key, known_hosts, report)
            timings["pod_provisioning_seconds"] = round(time.monotonic() - provisioning_started, 3)
            report("vggt_geometry", 33, "Transferring prepared frames to RunPod")
            input_transfer_started = time.monotonic()
            subprocess.run(
                self._scp_base(port, private_key, known_hosts) + [str(input_archive), f"root@{host}:/workspace/input.tar.gz"],
                check=True, timeout=1800,
            )
            timings["input_transfer_seconds"] = round(time.monotonic() - input_transfer_started, 3)

            report("vggt_geometry", 34, "Preparing the remote VGGT model")
            try:
                self._launch_reconstruction(host, port, private_key, known_hosts, request.quality_profile)
                timings.update(self._await_reconstruction(pod_id, host, port, private_key, known_hosts, report))
            except Exception:
                # Preserve completed VGGT geometry even when gsplat fails so a
                # retry does not pay for geometry a second time.
                request.result_dir.mkdir(parents=True, exist_ok=True)
                geometry = request.result_dir / "vggt_geometry.tar.gz"
                subprocess.run(
                    self._ssh_base(host, port, private_key, known_hosts) +
                    ["tar -czf /workspace/geometry-on-failure.tar.gz -C /workspace/job/scene sparse"],
                    capture_output=True, timeout=60,
                )
                subprocess.run(
                    self._scp_base(port, private_key, known_hosts) +
                    [f"root@{host}:/workspace/geometry-on-failure.tar.gz", str(geometry)],
                    capture_output=True, timeout=300,
                )
                raise
            report("finalizing", 96, "Downloading completed scene")
            output_archive = transfer_dir / "runpod_output.tar.gz"
            retrieval_started = time.monotonic()
            subprocess.run(
                self._scp_base(port, private_key, known_hosts) + [f"root@{host}:/workspace/output.tar.gz", str(output_archive)],
                check=True, timeout=1800,
            )
            request.result_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(output_archive, "r:gz") as archive:
                archive.extractall(request.result_dir)
            timings["artifact_retrieval_seconds"] = round(time.monotonic() - retrieval_started, 3)
            succeeded = True
            execution_result = ExecutionResult(
                request.result_dir / "scene.ply", request.result_dir / "scene_cameras.json",
                request.result_dir / "vggt_geometry.tar.gz", request.result_dir / "metrics.json", pod_id,
            )
        finally:
            active_error = sys.exc_info()[0] is not None
            if pod_id:
                try:
                    # Only claim "finalizing"/99% once the run actually reached that
                    # point. Reporting it unconditionally here (including on early
                    # failures, e.g. an SSH-readiness timeout) fabricated progress
                    # that then stuck around as a stale high-water mark for a retry.
                    if succeeded:
                        report("finalizing", 99, "Terminating temporary RunPod GPU")
                    termination_started = time.monotonic()
                    self._terminate(pod_id)
                    timings["pod_termination_seconds"] = round(time.monotonic() - termination_started, 3)
                    if paid_started is not None:
                        timings["paid_runpod_seconds"] = round(time.monotonic() - paid_started, 3)
                except Exception as cleanup_error:
                    if active_error:
                        print(f"RunPod cleanup also failed for pod {pod_id}: {cleanup_error}", file=sys.stderr)
                    else:
                        raise RuntimeError(f"RunPod cleanup failed for pod {pod_id}: {cleanup_error}") from cleanup_error
            metrics_path = request.result_dir / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics["timings"] = {**dict(metrics.get("timings") or {}), **timings}
                metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        if execution_result is None:
            raise RuntimeError("RunPod reconstruction finished without an execution result")
        return execution_result
