import io
import tempfile
import unittest

from app.main import app
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


if __name__ == "__main__":
    unittest.main()
