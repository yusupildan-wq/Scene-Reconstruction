import unittest

import numpy as np

from experiments.robust_fusion import (
    RobustFusionConfig,
    concatenation_fusion,
    robust_consensus_fusion,
)


class RobustFusionTests(unittest.TestCase):
    def test_concatenation_preserves_order_and_values_exactly(self):
        points = [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([[7, 8, 9]], dtype=np.float32),
        ]
        colors = [array / 10 for array in points]
        xyz, rgb, stats = concatenation_fusion(points, colors)
        np.testing.assert_array_equal(xyz, np.concatenate(points))
        np.testing.assert_array_equal(rgb, np.concatenate(colors))
        self.assertEqual(stats["input_observations"], 3)
        self.assertEqual(stats["rejected_observations"], 0)

    def test_consensus_suppresses_duplicates_and_uses_distinct_view_support(self):
        points = [
            np.array([[0.001, 0, 0], [0.002, 0, 0]], dtype=np.float32),
            np.array([[0.003, 0, 0]], dtype=np.float32),
            np.array([[0.004, 0, 0]], dtype=np.float32),
        ]
        colors = [
            np.tile([1.0, 0.0, 0.0], (len(view), 1)).astype(np.float32)
            for view in points
        ]
        xyz, rgb, stats = robust_consensus_fusion(
            points,
            colors,
            RobustFusionConfig(voxel_size=0.1, max_position_disagreement=0.05),
        )
        self.assertEqual(len(xyz), 1)
        self.assertAlmostEqual(float(xyz[0, 0]), 0.003, places=6)
        np.testing.assert_allclose(rgb[0], [1, 0, 0])
        self.assertEqual(stats["support_distribution"]["max"], 3.0)
        self.assertEqual(stats["input_observations"], 4)
        self.assertGreater(stats["duplicate_observations_suppressed"], 0)

    def test_position_and_color_outlier_is_rejected(self):
        points = [
            np.array([[0.001, 0, 0]], dtype=np.float32),
            np.array([[0.002, 0, 0]], dtype=np.float32),
            np.array([[0.08, 0, 0]], dtype=np.float32),
        ]
        colors = [
            np.array([[0.2, 0.4, 0.6]], dtype=np.float32),
            np.array([[0.22, 0.42, 0.62]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        ]
        xyz, rgb, stats = robust_consensus_fusion(
            points,
            colors,
            RobustFusionConfig(
                voxel_size=0.1,
                max_position_disagreement=0.02,
                mad_multiplier=2.0,
            ),
        )
        self.assertEqual(len(xyz), 1)
        self.assertLess(float(xyz[0, 0]), 0.01)
        np.testing.assert_allclose(rgb[0], [0.21, 0.41, 0.61], atol=1e-6)
        self.assertEqual(stats["rejected_observations"], 1)
        self.assertGreater(
            stats["spatial_disagreement_before"]["max"],
            stats["accepted_consensus_residuals"]["max"],
        )
        self.assertEqual(
            stats["input_observations"],
            stats["rejected_observations"]
            + stats["duplicate_observations_suppressed"]
            + stats["fused_points"],
        )

    def test_cells_without_cross_view_support_fail_loudly(self):
        points = [
            np.array([[0, 0, 0]], dtype=np.float32),
            np.array([[1, 0, 0]], dtype=np.float32),
        ]
        colors = [np.zeros((1, 3), dtype=np.float32) for _ in points]
        with self.assertRaisesRegex(RuntimeError, "rejected every spatial cell"):
            robust_consensus_fusion(
                points,
                colors,
                RobustFusionConfig(voxel_size=0.1, min_view_support=2),
            )

    def test_invalid_thresholds_are_rejected(self):
        points = [np.zeros((1, 3), dtype=np.float32)]
        colors = [np.zeros((1, 3), dtype=np.float32)]
        with self.assertRaises(ValueError):
            robust_consensus_fusion(
                points, colors, RobustFusionConfig(voxel_size=0)
            )


if __name__ == "__main__":
    unittest.main()
