import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

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
                 patch.object(executor, "_launch_reconstruction"), \
                 patch.object(executor, "_await_reconstruction", side_effect=RuntimeError("remote failed")), \
                 patch("app.executors.runpod_pod.subprocess.run"), \
                 patch.object(executor, "_terminate") as terminate:
                with self.assertRaisesRegex(RuntimeError, "remote failed"):
                    executor.execute(request, lambda *_: None)
                terminate.assert_called_once_with("pod-1")

    def test_runpod_launch_uses_detached_job_runner_and_shared_v3_command(self):
        """The shared V3 command must run byte-for-byte the same; only how it's launched changed
        (detached via run_gpu_job.sh, so it survives an SSH disconnect instead of dying with it)."""
        executor = RunPodExecutor()
        with patch("app.executors.runpod_pod.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            executor._launch_reconstruction("1.2.3.4", 2222, Path("key"), Path("known_hosts"), "baseline")
        remote_command = run.call_args.args[0][-1]
        self.assertIn("run_gpu_job.sh reconstruct bash -c", remote_command)
        self.assertIn("execute_v3_workspace.py", remote_command)
        self.assertIn("verify_env.sh", remote_command)
        self.assertIn("--quality-profile baseline", remote_command)
        self.assertIn("WORKSPACE_ROOT=/workspace", remote_command)
        self.assertIn("TORCH_HOME=/opt/cache/torch", remote_command)
        self.assertIn("HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface/hub", remote_command)
        self.assertIn("TORCH_EXTENSIONS_DIR=/workspace/cache/torch_extensions", remote_command)

    def test_runpod_await_reconnects_when_ssh_drops_but_job_continues(self):
        """A dropped SSH connection must not restart or abandon the job: it runs detached on the
        pod, so reconnecting and resuming monitoring is correct behavior, restarting is not."""
        executor = RunPodExecutor()
        progress_line = "SCENE_PROGRESS " + json.dumps({"stage": "gaussian_optimization", "progress": 60, "detail": "training"})
        success_stdout = f"some earlier log line\n{progress_line}\n" + RunPodExecutor._EXIT_MARKER + "\n0\n"
        responses = [
            SimpleNamespace(returncode=255, stdout="", stderr="Connection to 1.2.3.4 closed by remote host."),
            SimpleNamespace(returncode=0, stdout=success_stdout, stderr=""),
        ]
        reports: list[tuple[str, int, str]] = []
        with patch("app.executors.runpod_pod.subprocess.run", side_effect=responses), \
             patch("app.executors.runpod_pod.time.sleep"), \
             patch.object(executor, "_get_pod", return_value={"desiredStatus": "RUNNING"}) as get_pod:
            executor._await_reconstruction(
                "pod-1", "1.2.3.4", 2222, Path("key"), Path("known_hosts"),
                lambda stage, pct, detail: reports.append((stage, pct, detail)),
            )
        get_pod.assert_called_once()
        self.assertIn(("gaussian_optimization", 60, "training"), reports)

    def test_runpod_await_survives_transient_get_pod_error_during_reconnect(self):
        """RunPod's own status API can blip (e.g. a transient 404) independent of whether the
        pod is actually alive; that must not be treated as proof of a dead pod."""
        executor = RunPodExecutor()
        progress_line = "SCENE_PROGRESS " + json.dumps({"stage": "gaussian_optimization", "progress": 60, "detail": "training"})
        success_stdout = f"{progress_line}\n" + RunPodExecutor._EXIT_MARKER + "\n0\n"
        responses = [
            SimpleNamespace(returncode=255, stdout="", stderr="Connection reset by peer"),
            SimpleNamespace(returncode=0, stdout=success_stdout, stderr=""),
        ]
        reports: list[tuple[str, int, str]] = []
        with patch("app.executors.runpod_pod.subprocess.run", side_effect=responses), \
             patch("app.executors.runpod_pod.time.sleep"), \
             patch.object(executor, "_get_pod", side_effect=requests.RequestException("404")):
            executor._await_reconstruction(
                "pod-1", "1.2.3.4", 2222, Path("key"), Path("known_hosts"),
                lambda stage, pct, detail: reports.append((stage, pct, detail)),
            )
        self.assertIn(("gaussian_optimization", 60, "training"), reports)

    def test_runpod_wait_for_ssh_survives_transient_get_pod_error(self):
        """A transient failure to query RunPod's status API must not be treated as a dead pod;
        it should keep polling within the provisioning budget."""
        executor = RunPodExecutor()
        clock = _FakeClock()
        pods = [
            requests.RequestException("503"),
            {"desiredStatus": "RUNNING", "publicIp": "1.2.3.4", "portMappings": {"22": 2222}},
        ]
        with patch("app.executors.runpod_pod.time.monotonic", side_effect=clock.monotonic), \
             patch("app.executors.runpod_pod.time.sleep", side_effect=clock.sleep), \
             patch.object(executor, "_get_pod", side_effect=pods), \
             patch("app.executors.runpod_pod.subprocess.run", return_value=SimpleNamespace(returncode=0)):
            host, port = executor._wait_for_ssh("pod-1", Path("key"), Path("known_hosts"), lambda *_: None)
        self.assertEqual((host, port), ("1.2.3.4", 2222))

    def test_runpod_await_poll_command_does_not_fail_on_missing_exit_file(self):
        """The remote poll command must always exit 0 for a healthy SSH connection,
        even while the job is still running and its exit-status file doesn't exist
        yet -- `cat` on a missing file returns 1, and without a trailing `true` that
        became the whole command's exit status, making every poll of a perfectly
        healthy in-progress job look identical to a dropped connection."""
        executor = RunPodExecutor()
        with patch("app.executors.runpod_pod.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stdout=RunPodExecutor._EXIT_MARKER + "\n0\n", stderr="")) as run, \
             patch.object(executor, "_get_pod", return_value={"desiredStatus": "RUNNING"}):
            executor._await_reconstruction(
                "pod-1", "1.2.3.4", 2222, Path("key"), Path("known_hosts"), lambda *_: None,
            )
        remote_command = run.call_args.args[0][-1]
        self.assertTrue(remote_command.rstrip().endswith("true"))

    def test_runpod_await_reconnects_to_reassigned_endpoint(self):
        """RunPod can remap a pod's public IP/port under it (observed after a launch);
        the reconnect loop must pick up the new endpoint instead of retrying the stale
        one for the whole grace period."""
        executor = RunPodExecutor()
        progress_line = "SCENE_PROGRESS " + json.dumps({"stage": "gaussian_optimization", "progress": 60, "detail": "training"})
        success_stdout = f"{progress_line}\n" + RunPodExecutor._EXIT_MARKER + "\n0\n"
        responses = [
            SimpleNamespace(returncode=255, stdout="", stderr="Connection timed out"),
            SimpleNamespace(returncode=0, stdout=success_stdout, stderr=""),
        ]
        with patch("app.executors.runpod_pod.subprocess.run", side_effect=responses) as run, \
             patch("app.executors.runpod_pod.time.sleep"), \
             patch.object(executor, "_get_pod",
                           return_value={"desiredStatus": "RUNNING", "publicIp": "5.6.7.8", "portMappings": {"22": 9999}}):
            executor._await_reconstruction(
                "pod-1", "1.2.3.4", 2222, Path("key"), Path("known_hosts"), lambda *_: None,
            )
        second_command = run.call_args_list[1].args[0]
        self.assertTrue(any("5.6.7.8" in part for part in second_command))
        self.assertIn("9999", second_command)

    def test_runpod_await_fails_fast_when_pod_actually_exited(self):
        """If RunPod itself confirms the pod died, fail immediately rather than waiting out the
        reconnect grace period for a connection that will never come back."""
        executor = RunPodExecutor()
        with patch("app.executors.runpod_pod.subprocess.run",
                   return_value=SimpleNamespace(returncode=255, stdout="", stderr="Connection reset by peer")), \
             patch("app.executors.runpod_pod.time.sleep") as sleep_mock, \
             patch.object(executor, "_get_pod", return_value={"desiredStatus": "EXITED"}):
            with self.assertRaisesRegex(RuntimeError, "RunPod pod exited"):
                executor._await_reconstruction("pod-1", "1.2.3.4", 2222, Path("key"), Path("known_hosts"), lambda *_: None)
        sleep_mock.assert_not_called()

    def test_runpod_await_gives_up_after_reconnect_grace_period(self):
        """An SSH connection that never comes back, and a pod RunPod's API can't confirm is dead
        either, must still fail within a bounded budget instead of polling forever."""
        executor = RunPodExecutor()
        clock = _FakeClock()
        with patch("app.executors.runpod_pod.subprocess.run",
                   return_value=SimpleNamespace(returncode=255, stdout="", stderr="")), \
             patch("app.executors.runpod_pod.time.monotonic", side_effect=clock.monotonic), \
             patch("app.executors.runpod_pod.time.sleep", side_effect=clock.sleep), \
             patch.object(executor, "_get_pod", return_value={"desiredStatus": "RUNNING"}), \
             patch.object(settings, "runpod_reconnect_grace_seconds", 20), \
             patch.object(settings, "runpod_poll_interval_seconds", 5):
            with self.assertRaisesRegex(RuntimeError, "could not reconnect"):
                executor._await_reconstruction("pod-1", "1.2.3.4", 2222, Path("key"), Path("known_hosts"), lambda *_: None)

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

    def test_runpod_does_not_report_finalizing_progress_on_failure(self):
        """A failed run must not fabricate 'finalizing'/99% progress: orchestrator._set never
        lets progress_percent decrease within a run, so a false 99% here would stick around as
        a stale high-water mark and leak into a later retry's progress bar."""
        executor = RunPodExecutor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene"
            (scene / "images").mkdir(parents=True)
            (scene / "images" / "frame.jpg").write_bytes(b"frame")
            request = ExecutionRequest("job-id", scene, root / "result", "baseline")
            reports: list[tuple[str, int, str]] = []
            with patch.object(executor, "validate", return_value=ProviderCapability(True, "ready")), \
                 patch.object(executor, "_ensure_key", return_value=(root / "key", "ssh-ed25519 test")), \
                 patch.object(executor, "_create_pod", return_value={"id": "pod-1"}), \
                 patch.object(executor, "_wait_for_ssh", side_effect=TimeoutError("RunPod did not become reachable")), \
                 patch("app.executors.runpod_pod.subprocess.run"), \
                 patch.object(executor, "_terminate") as terminate:
                with self.assertRaises(TimeoutError):
                    executor.execute(request, lambda stage, pct, detail: reports.append((stage, pct, detail)))
            terminate.assert_called_once_with("pod-1")
            self.assertFalse(any(stage == "finalizing" for stage, _, _ in reports))

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
