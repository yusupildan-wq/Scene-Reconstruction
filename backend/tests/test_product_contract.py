import io
import json
import tempfile
import unittest
from types import SimpleNamespace

from app.main import app
from app.orchestrator import _validate_viewer_artifacts
from app.storage import LocalStorage


class ProductContractTests(unittest.TestCase):
    def test_api_exposes_progress_retry_and_viewer_artifacts(self):
        schema = app.openapi()
        self.assertIn("/jobs/{job_id}/retry", schema["paths"])
        self.assertIn("/jobs/{job_id}/scene.ply", schema["paths"])
        fields = schema["components"]["schemas"]["JobOut"]["properties"]
        self.assertIn("progress_percent", fields)
        self.assertIn("scene_url", fields)

    def test_local_storage_streams_upload_and_reports_reusable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = LocalStorage(directory)
            storage.save_fileobj("job/input.mp4", io.BytesIO(b"video-data"))
            self.assertTrue(storage.exists("job/input.mp4"))
            self.assertEqual(storage.read("job/input.mp4"), b"video-data")

    def test_completed_scene_requires_portable_camera_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = LocalStorage(directory)
            job = SimpleNamespace(output_storage_key="output/scene.ply",
                                  camera_storage_key="output/scene_cameras.json")
            storage.save(job.output_storage_key, b"ply")
            with self.assertRaisesRegex(RuntimeError, "without camera metadata"):
                _validate_viewer_artifacts(job, storage)

            metadata = {
                "coordinate_convention": "opencv",
                "frames": [{"transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0],
                                                    [0, 0, 1, 0], [0, 0, 0, 1]]}],
            }
            storage.save(job.camera_storage_key, json.dumps(metadata).encode())
            _validate_viewer_artifacts(job, storage)


if __name__ == "__main__":
    unittest.main()
