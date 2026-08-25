import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.orchestrator import _workspace
from app.pipeline import extract_frames
from app.storage import LocalStorage


class GoldenPreparationTests(unittest.TestCase):
    def test_default_frame_budget_matches_golden_run(self):
        self.assertEqual(inspect.signature(extract_frames).parameters["max_selected_frames"].default, 96)

    def test_workspace_contains_96_hashed_manifest_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            storage = LocalStorage(str(root / "storage"))
            frame_keys = []
            for index in range(96):
                key = f"frames/source_{index:04d}.jpg"
                storage.save(key, f"frame-{index}".encode())
                frame_keys.append(key)
            job = SimpleNamespace(id="golden-job", stage_artifacts={})
            with patch("app.orchestrator.SCRATCH_DIR", root / "scratch"):
                scene_dir, _ = _workspace(job, storage, frame_keys)
            manifest = json.loads((scene_dir / "input_manifest.json").read_text())
            self.assertEqual(manifest["selected_count"], 96)
            self.assertEqual(len(manifest["images"]), 96)
            self.assertEqual(len(list((scene_dir / "images").glob("*.jpg"))), 96)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["images"]))


if __name__ == "__main__":
    unittest.main()
