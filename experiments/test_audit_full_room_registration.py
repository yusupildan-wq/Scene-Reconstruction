from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from experiments.audit_full_room_registration import (
    AuditFailure,
    assert_unique_chronology,
    audit_registration,
    compare_rgb,
    convention_score,
    independent_pixel_transform,
    main,
    preprocess_current_path,
    validate_intrinsics,
    validate_pose,
)


class AuditPrimitiveTests(unittest.TestCase):
    def test_standard_16_by_9_transform_has_no_crop(self):
        transform = independent_pixel_transform(1920, 1080, dust3r_size=512, target_long_edge=1440)
        self.assertEqual(transform.processed_size, (512, 288))
        self.assertEqual(transform.crop_box, (0.0, 0.0, 1920.0, 1080.0))
        self.assertEqual(transform.final_size, (1440, 810))
        self.assertAlmostEqual(transform.scale_x, 1440 / 512)
        self.assertAlmostEqual(transform.scale_y, 810 / 288)

    def test_duplicate_or_nonmonotonic_chronology_fails(self):
        with self.assertRaises(AuditFailure):
            assert_unique_chronology(["frame_000010.jpg", "frame_000010.jpg"])
        with self.assertRaises(AuditFailure):
            assert_unique_chronology(["frame_000020.jpg", "frame_000010.jpg"])

    def test_invalid_intrinsics_and_pose_fail_loudly(self):
        with self.assertRaises(AuditFailure):
            validate_intrinsics(np.diag([-1.0, 1.0, 1.0]), 10, 10, 0)
        reflected = np.eye(4)
        reflected[0, 0] = -1
        with self.assertRaises(AuditFailure):
            validate_pose(reflected, 0)

    def test_documented_projection_convention_reprojects(self):
        height, width = 6, 8
        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        points = np.stack((xs, ys, np.ones_like(xs)), axis=-1).astype(np.float64)
        mask = np.ones((height, width), dtype=bool)
        score = convention_score(points, mask, np.eye(3), np.eye(4))
        self.assertEqual(score["positive_depth_ratio"], 1.0)
        self.assertAlmostEqual(score["median_pixels"], 0.0)
        self.assertAlmostEqual(score["p95_pixels"], 0.0)

    def test_rgb_shape_mismatch_fails(self):
        with self.assertRaises(AuditFailure):
            compare_rgb(np.zeros((4, 5, 3)), np.zeros((5, 4, 3)), 7)

    def test_current_preprocessing_produces_expected_geometry_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame_000010.jpg"
            Image.new("RGB", (32, 18), (100, 120, 140)).save(path)
            rgb, transform, metadata = preprocess_current_path(
                path, dust3r_size=32, target_long_edge=None
            )
            self.assertEqual(rgb.shape, (16, 32, 3))
            self.assertEqual(transform.processed_size, (32, 16))
            self.assertEqual(metadata["oriented_size"], [32, 18])


class AuditStructureTests(unittest.TestCase):
    def _make_checkpoint(self, root: Path, *, graph_name: str = "frame_000010.jpg") -> tuple[Path, Path]:
        run_dir = root / "run"
        output_dir = root / "audit"
        geometry = run_dir / "geometry"
        frames = run_dir / "frames"
        geometry.mkdir(parents=True)
        frames.mkdir()
        Image.new("RGB", (16, 16), (100, 120, 140)).save(frames / "frame_000010.jpg")
        np.savez(
            geometry / "cameras.npz",
            poses=np.eye(4, dtype=np.float32)[None],
            intrinsics=np.array([[[1, 0, 7], [0, 1, 7], [0, 0, 1]]], dtype=np.float32),
            frame_names=np.array(["frame_000010.jpg"]),
        )
        (geometry / "COMPLETE.json").write_text('{"views": 1, "point_maps": 1}', encoding="utf-8")
        ys, xs = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        points = np.stack((xs - 7, ys - 7, np.ones_like(xs)), axis=-1).astype(np.float32)
        np.save(geometry / "points_0000.npy", points)
        np.save(geometry / "mask_0000.npy", np.ones((16, 16), dtype=bool))
        (run_dir / "pair_graph.json").write_text(
            '{"frames": ["' + graph_name + '"]}', encoding="utf-8"
        )
        return run_dir, output_dir

    def test_frame_identity_mismatch_fails_before_dust3r(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, output_dir = self._make_checkpoint(Path(directory), graph_name="frame_000020.jpg")
            with self.assertRaisesRegex(AuditFailure, "Pair-graph frame identities"):
                audit_registration(run_dir, output_dir, selected_views=[0], dust3r_size=16, target_long_edge=16)

    def test_cli_returns_nonzero_and_writes_failure_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "audit"
            exit_code = main([str(root / "missing"), "--output-dir", str(output), "--views", "0"])
            self.assertEqual(exit_code, 2)
            self.assertTrue((output / "registration_failure.json").exists())

    def test_shape_mismatch_is_structural_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, output_dir = self._make_checkpoint(Path(directory))
            np.save(run_dir / "geometry" / "mask_0000.npy", np.ones((15, 16), dtype=bool))
            fake_reference = [{"index": 0, "instance": "frame_000010.jpg", "true_shape": [16, 16], "rgb": np.zeros((16, 16, 3), dtype=np.float32)}]
            with mock.patch(
                "experiments.audit_full_room_registration.load_dust3r_reference",
                return_value=fake_reference,
            ):
                with self.assertRaisesRegex(AuditFailure, "point/mask shapes differ"):
                    audit_registration(run_dir, output_dir, selected_views=[0], dust3r_size=16, target_long_edge=16)

    def test_complete_synthetic_audit_writes_reports_and_montages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, output_dir = root / "run", root / "audit"
            geometry, frames = run_dir / "geometry", run_dir / "frames"
            geometry.mkdir(parents=True)
            frames.mkdir()
            names = ["frame_000010.jpg", "frame_000020.jpg"]
            for index, name in enumerate(names):
                Image.new("RGB", (32, 18), (100 + index * 20, 120, 140)).save(frames / name)
            poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
            Ks = np.repeat(np.array([[[1, 0, 15], [0, 1, 7], [0, 0, 1]]], dtype=np.float32), 2, axis=0)
            np.savez(geometry / "cameras.npz", poses=poses, intrinsics=Ks, frame_names=np.array(names))
            (geometry / "COMPLETE.json").write_text('{"views": 2, "point_maps": 2}', encoding="utf-8")
            ys, xs = np.meshgrid(np.arange(16), np.arange(32), indexing="ij")
            points = np.stack((xs - 15, ys - 7, np.ones_like(xs)), axis=-1).astype(np.float32)
            for index in range(2):
                np.save(geometry / f"points_{index:04d}.npy", points)
                np.save(geometry / f"mask_{index:04d}.npy", np.ones((16, 32), dtype=bool))
            (run_dir / "pair_graph.json").write_text(
                '{"frames": ["frame_000010.jpg", "frame_000020.jpg"]}', encoding="utf-8"
            )
            references = []
            for index, name in enumerate(names):
                rgb, _, _ = preprocess_current_path(frames / name, dust3r_size=32, target_long_edge=None)
                references.append(
                    {
                        "index": index,
                        "instance": name,
                        "identity_evidence": "path_instance",
                        "true_shape": [16, 32],
                        "rgb": rgb,
                    }
                )
            with mock.patch(
                "experiments.audit_full_room_registration.load_dust3r_reference",
                return_value=references,
            ):
                report = audit_registration(
                    run_dir,
                    output_dir,
                    selected_views=[0, 1],
                    dust3r_size=32,
                    target_long_edge=32,
                )
            self.assertEqual(report["status"], "passed")
            self.assertTrue((output_dir / "registration_report.json").exists())
            self.assertTrue((output_dir / "registration_summary.json").exists())
            self.assertTrue((output_dir / "montage_view_000.jpg").exists())
            self.assertTrue((output_dir / "chronology_boundaries.jpg").exists())


if __name__ == "__main__":
    unittest.main()
