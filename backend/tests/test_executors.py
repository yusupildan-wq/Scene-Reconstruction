import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config import settings
from app.executors.base import ExecutionRequest, ProviderCapability
from app.executors.local_nvidia import LocalNvidiaExecutor
from app.executors.runpod_pod import RunPodExecutor


class _FakeClock:
    """Deterministic stand-in for time.monotonic/time.sleep so wait-loop tests run instantly."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ExecutorTests(unittest.TestCase):
    def test_local_validation_recommends_runpod_without_nvidia(self):
        with patch("app.executors.local_nvidia.shutil.which", return_value=None):
            capability = LocalNvidiaExecutor().validate()
        self.assertFalse(capability.available)
        self.assertIn("RunPod", capability.detail)

    def test_local_executor_uses_shared_workspace_runner(self):
        executor = LocalNvidiaExecutor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = ExecutionRequest("job-id", root / "scene", root / "result", "baseline")
            request.result_dir.mkdir()
            for name in ("scene.ply", "scene_cameras.json", "vggt_geometry.tar.gz", "metrics.json"):
                (request.result_dir / name).write_bytes(b"output")
            fake_paths = (root, root / "vggt", root / "gsplat", root / "vggt-python", root / "gsplat-python")
            with patch.object(executor, "validate", return_value=ProviderCapability(True, "ready")), \
                 patch.object(executor, "_paths", return_value=fake_paths), \
                 patch("app.executors.local_nvidia.run_streaming") as run_local:
                executor.execute(request, lambda *_: None)
            self.assertTrue(any("execute_v3_workspace.py" in part for part in run_local.call_args.args[0]))

    def test_runpod_is_terminated_when_remote_execution_fails(self):
        executor = RunPodExecutor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene"
            (scene / "images").mkdir(parents=True)
            (scene / "images" / "frame.jpg").write_bytes(b"frame")
            request = ExecutionRequest("job-id", scene, root / "result", "baseline")
            with patch.object(executor, "validate", return_value=ProviderCapability(True, "ready")), \
                 patch.object(executor, "_ensure_key", return_value=(root / "key", "ssh-ed25519 test")), \
                 patch.object(executor, "_create_pod", return_value={"id": "pod-1"}), \
                 patch.object(executor, "_wait_for_ssh", return_value=("127.0.0.1", 2222)), \
                 patch("app.executors.runpod_pod.subprocess.run"), \
                 patch("app.executors.runpod_pod.run_streaming", side_effect=RuntimeError("remote failed")) as run_remote, \
                 patch.object(executor, "_terminate") as terminate:
                with self.assertRaisesRegex(RuntimeError, "remote failed"):
                    executor.execute(request, lambda *_: None)
                self.assertIn("execute_v3_workspace.py", run_remote.call_args.args[0][-1])
                terminate.assert_called_once_with("pod-1")

    def test_runpod_wait_for_ssh_fails_fast_when_pod_exits(self):
        """A pod RunPod itself reports as EXITED/TERMINATED is a real failure, not a slow cold start."""
        executor = RunPodExecutor()
        clock = _FakeClock()
        with patch("app.executors.runpod_pod.time.monotonic", side_effect=clock.monotonic), \
             patch("app.executors.runpod_pod.time.sleep", side_effect=clock.sleep), \
             patch.object(executor, "_get_pod", return_value={"desiredStatus": "EXITED"}) as get_pod:
            with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                executor._wait_for_ssh("pod-1", Path("key"), Path("known_hosts"), lambda *_: None)
        get_pod.assert_called_once()
        self.assertEqual(clock.now, 0.0)

    def test_runpod_wait_for_ssh_times_out_while_image_still_pulling(self):
        """No publicIp/portMappings yet and status still RUNNING means a legitimate slow image pull, not a stall."""
        executor = RunPodExecutor()
        clock = _FakeClock()
        pod = {"desiredStatus": "RUNNING", "publicIp": None, "portMappings": {}}
        with patch("app.executors.runpod_pod.time.monotonic", side_effect=clock.monotonic), \
             patch("app.executors.runpod_pod.time.sleep", side_effect=clock.sleep), \
             patch.object(executor, "_get_pod", return_value=pod), \
             patch.object(settings, "runpod_startup_timeout_seconds", 12):
            with self.assertRaisesRegex(TimeoutError, "provisioning \\(scheduling or image pull\\)"):
                executor._wait_for_ssh("pod-1", Path("key"), Path("known_hosts"), lambda *_: None)

    def test_runpod_wait_for_ssh_times_out_when_ssh_never_comes_up(self):
        """publicIp/portMappings present (container network is up) but sshd never answers is a real stall, not a pull."""
        executor = RunPodExecutor()
        clock = _FakeClock()
        pod = {"desiredStatus": "RUNNING", "publicIp": "1.2.3.4", "portMappings": {"22": 2222}}
        with patch("app.executors.runpod_pod.time.monotonic", side_effect=clock.monotonic), \
             patch("app.executors.runpod_pod.time.sleep", side_effect=clock.sleep), \
             patch.object(executor, "_get_pod", return_value=pod), \
             patch.object(settings, "runpod_ssh_ready_timeout_seconds", 12), \
             patch("app.executors.runpod_pod.subprocess.run", return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(TimeoutError, "SSH never came up"):
                executor._wait_for_ssh("pod-1", Path("key"), Path("known_hosts"), lambda *_: None)

    def test_runpod_wait_for_ssh_survives_a_slow_image_pull_then_connects(self):
        """A pod that stays RUNNING with no network for a while and then comes up should succeed, not time out."""
        executor = RunPodExecutor()
        clock = _FakeClock()
        pods = [
            {"desiredStatus": "RUNNING", "publicIp": None, "portMappings": {}},
            {"desiredStatus": "RUNNING", "publicIp": None, "portMappings": {}},
            {"desiredStatus": "RUNNING", "publicIp": "1.2.3.4", "portMappings": {"22": 2222}},
        ]
        with patch("app.executors.runpod_pod.time.monotonic", side_effect=clock.monotonic), \
             patch("app.executors.runpod_pod.time.sleep", side_effect=clock.sleep), \
             patch.object(executor, "_get_pod", side_effect=pods), \
             patch("app.executors.runpod_pod.subprocess.run", return_value=SimpleNamespace(returncode=0)):
            host, port = executor._wait_for_ssh("pod-1", Path("key"), Path("known_hosts"), lambda *_: None)
        self.assertEqual((host, port), ("1.2.3.4", 2222))

    def test_runpod_create_uses_rest_pod_image_and_direct_ssh(self):
        executor = RunPodExecutor()
        response = SimpleNamespace(ok=True, json=lambda: {"id": "pod-1"}, text="")
        with patch("app.executors.runpod_pod.requests.post", return_value=response) as post:
            executor._create_pod("ssh-ed25519 public", "job-id")
        payload = post.call_args.kwargs["json"]
        self.assertIn("imageName", payload)
        self.assertNotIn("image", payload)
        self.assertEqual(payload["ports"], ["22/tcp"])
        self.assertEqual(payload["env"]["AUTOMATED_JOB"], "1")


if __name__ == "__main__":
    unittest.main()
