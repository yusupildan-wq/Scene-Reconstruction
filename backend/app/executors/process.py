from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.executors.base import ProgressReporter

PREFIX = "SCENE_PROGRESS "


def run_streaming(command: list[str], cwd: Path, report: ProgressReporter) -> None:
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    tail: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-30:]
        if line.startswith(PREFIX):
            update = json.loads(line[len(PREFIX):])
            report(update["stage"], int(update["progress"]), update["detail"])
    if process.wait() != 0:
        raise RuntimeError("Reconstruction command failed:\n" + "\n".join(tail[-10:]))
