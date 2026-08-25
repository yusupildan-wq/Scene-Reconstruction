import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import v3_serverless as worker


class V3ServerlessContractTests(unittest.TestCase):
    def test_reconstruction_runs_existing_stages_and_uploads_all_artifacts(self):
        uploads: list[tuple[str, str]] = []
        updates: list[tuple[str, int, str]] = []

        def download(_url: str, destination: Path) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"frame")

        def run(command: list[str], _cwd: Path) -> None:
            scene_dir = Path(command[command.index("--scene-dir") + 1]) if "--scene-dir" in command else None
            if "--stage" in command and command[command.index("--stage") + 1] == "geometry":
                sparse = scene_dir / "sparse" / "0"
                sparse.mkdir(parents=True)
                for name in ("cameras.bin", "images.bin", "points3D.bin"):
                    (sparse / name).write_bytes(b"geometry")
            elif "--stage" in command:
                result_dir = Path(command[command.index("--result-dir") + 1])
                (result_dir / "ply").mkdir(parents=True)
                (result_dir / "stats").mkdir(parents=True)
                (result_dir / "ply" / "point_cloud_29999.ply").write_bytes(b"ply")
                (result_dir / "stats" / "val_step29999.json").write_text('{"psnr": 24.1}')
            else:
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text('{"coordinate_convention":"opencv","frames":[{}]}')

        def upload(url: str, source: Path, _content_type: str) -> None:
            self.assertTrue(source.is_file())
            uploads.append((url, source.name))

        payload = {
            "job_id": "job-1",
            "frame_urls": ["frame-a", "frame-b"],
            "scene_upload_url": "put-scene",
            "cameras_upload_url": "put-cameras",
            "geometry_upload_url": "put-geometry",
            "quality_profile": "high",
        }
        with patch.object(worker, "_download", side_effect=download), \
             patch.object(worker, "_run", side_effect=run), \
             patch.object(worker, "_upload", side_effect=upload):
            result = worker.reconstruct(payload, lambda *args: updates.append(args))

        self.assertEqual(result["stage"], "complete")
        self.assertEqual(result["metrics"]["psnr"], 24.1)
        self.assertEqual({url for url, _ in uploads}, {"put-scene", "put-cameras", "put-geometry"})
        self.assertEqual([stage for stage, _, _ in updates], [
            "preparing_frames", "vggt_geometry", "vggt_geometry",
            "gaussian_optimization", "finalizing",
        ])

    def test_payload_requires_portable_resume_artifact_url(self):
        with self.assertRaisesRegex(ValueError, "geometry_upload_url"):
            worker.validate_payload({
                "job_id": "job-1", "frame_urls": ["a", "b"],
                "scene_upload_url": "scene", "cameras_upload_url": "cameras",
            })


if __name__ == "__main__":
    unittest.main()
