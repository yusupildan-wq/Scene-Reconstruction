"""Regression test for the RunPod cold-start CUDA-readiness race.

nvidia-smi (NVML) can report the GPU before the CUDA driver API is ready to
create a context on a freshly attached RunPod GPU: the first
`torch.cuda.is_available()` call in the container fails with "CUDA
initialization: CUDA unknown error", then succeeds moments later with no
other change. bootstrap/wait_for_cuda.sh polls for readiness with a bounded
budget instead of a blind sleep or a single one-shot check; these tests drive
it against a fake `torch` module (no real GPU/CUDA involved) that fails a
configurable number of times before reporting ready.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "wait_for_cuda.sh"


def _find_bash() -> str | None:
    # On Windows, plain PATH lookup can resolve to the WSL launcher shim at
    # C:\Windows\System32\bash.exe, which fails without a WSL distro installed.
    # Prefer a real POSIX bash (Git for Windows) when present.
    for candidate in (r"C:\Program Files\Git\usr\bin\bash.exe", r"C:\Program Files\Git\bin\bash.exe"):
        if Path(candidate).is_file():
            return candidate
    return shutil.which("bash")


BASH = _find_bash()

FAKE_TORCH = textwrap.dedent(
    """
    import os
    from pathlib import Path

    __version__ = "2.4.1+cu124"

    class _Version:
        cuda = "12.4"

    version = _Version()
    _state_path = Path(os.environ["FAKE_CUDA_STATE_FILE"])
    _ready_after = int(os.environ.get("FAKE_CUDA_READY_AFTER", "0"))

    def _tick() -> int:
        count = int(_state_path.read_text()) if _state_path.exists() else 0
        count += 1
        _state_path.write_text(str(count))
        return count

    class cuda:
        @staticmethod
        def is_available():
            return 0 < _ready_after <= _tick()

        @staticmethod
        def device_count():
            return 1 if cuda.is_available() else 0

        @staticmethod
        def get_device_name():
            return "Fake GPU"
    """
)


@unittest.skipUnless(BASH, "a POSIX bash is required to run wait_for_cuda.sh")
class WaitForCudaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "torch.py").write_text(FAKE_TORCH, encoding="utf-8")
        self.state_file = self.tmp / "state"

    def _run(self, ready_after: int, max_attempts: int) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(self.tmp),
            "FAKE_CUDA_STATE_FILE": str(self.state_file),
            "FAKE_CUDA_READY_AFTER": str(ready_after),
            "CUDA_READY_MAX_ATTEMPTS": str(max_attempts),
            "CUDA_READY_POLL_SECONDS": "0",
        })
        return subprocess.run(
            [BASH, str(SCRIPT), sys.executable],
            capture_output=True, text=True, timeout=30, env=env,
        )

    def test_succeeds_after_transient_cuda_unavailability(self):
        """CUDA becomes available on the 3rd poll, well within budget: this must succeed, not fail fast."""
        result = self._run(ready_after=3, max_attempts=12)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA not ready yet (attempt 1/12)", result.stdout)
        self.assertIn("CUDA not ready yet (attempt 2/12)", result.stdout)

    def test_fails_clearly_after_exhausting_bounded_attempts(self):
        """CUDA that never becomes available must fail with a clear, bounded error, not hang or succeed silently."""
        result = self._run(ready_after=0, max_attempts=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA did not become available after 3 attempts", result.stderr)

    def test_prints_diagnostics_for_every_run(self):
        result = self._run(ready_after=1, max_attempts=12)
        self.assertIn("CUDA/NVIDIA environment diagnostics", result.stdout)


if __name__ == "__main__":
    unittest.main()
