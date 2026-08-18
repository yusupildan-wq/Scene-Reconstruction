import unittest
from argparse import Namespace

import numpy as np

from experiments.run_full_room_gaussians import (
    _cross_view_pair_score,
    cross_view_diagnostics,
    resolve_profile,
)


class CrossViewDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        height, width = 8, 10
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        self.points = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float32)
        self.mask = np.ones((height, width), dtype=bool)
        self.K = np.eye(3, dtype=np.float32)
        self.pose = np.eye(4, dtype=np.float32)

    def test_matching_surfaces_are_inliers(self):
        result = _cross_view_pair_score(
            self.points, self.mask, self.points.copy(), self.mask,
            self.K, np.eye(4, dtype=np.float32),
        )
        self.assertAlmostEqual(result["inlier_ratio"], 1.0)
        self.assertAlmostEqual(result["median_relative_error"], 0.0)

    def test_inconsistent_surface_marks_both_views_weak(self):
        inconsistent = self.points.copy()
        inconsistent[..., 2] += 1.0
        result = cross_view_diagnostics(
            [self.points, inconsistent], [self.mask, self.mask],
            [self.K, self.K], [self.pose, self.pose], radius=1,
        )
        self.assertEqual(result["weak_views"], [0, 1])
        self.assertLess(result["median_pair_inlier_ratio"], 0.01)


class TrainingProfileTests(unittest.TestCase):
    def test_photoreal_profile_enables_high_detail_training(self):
        config = resolve_profile(
            "photoreal",
            Namespace(iterations=None, target_long_edge=None, max_initial_points=None),
        )
        self.assertEqual(config["target_long_edge"], 1440)
        self.assertEqual(config["max_initial_points"], 400_000)
        self.assertEqual(config["sh_degree"], 2)
        self.assertGreater(config["densify_until"], 2400)

    def test_cli_values_override_profile(self):
        config = resolve_profile(
            "photoreal",
            Namespace(iterations=12, target_long_edge=800, max_initial_points=123),
        )
        self.assertEqual(config["iterations"], 12)
        self.assertEqual(config["target_long_edge"], 800)
        self.assertEqual(config["max_initial_points"], 123)


if __name__ == "__main__":
    unittest.main()
