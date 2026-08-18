import unittest

import numpy as np

from experiments.run_full_room_gaussians import (
    _cross_view_pair_score,
    cross_view_diagnostics,
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


if __name__ == "__main__":
    unittest.main()
