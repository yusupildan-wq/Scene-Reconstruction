import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.run_subscene_experiments import (
    _assert_resume_compatible,
    _json_write,
    _pair_geometry_metrics,
    _sample_observations,
    fuse_sampled_observations,
    image_metrics,
    nested_subsets,
    parse_index_list,
)


class SubsetTests(unittest.TestCase):
    def test_default_shape_is_nested_and_keeps_evaluation_identity(self):
        subsets = nested_subsets(range(46, 62), (1, 4, 8, 16))
        self.assertEqual(subsets[1], (54,))
        for views in subsets.values():
            self.assertIn(54, views)
        self.assertTrue(set(subsets[8]).issubset(subsets[16]))

    def test_explicit_views_reject_duplicates(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_index_list("1,1,2")


class SamplingAndFusionTests(unittest.TestCase):
    def test_sampling_is_fixed_per_camera_and_reproducible(self):
        points = [np.arange(90, dtype=np.float32).reshape(30, 3)] * 2
        colors = [item / 90 for item in points]
        first, _ = _sample_observations(points, colors, 7, 42)
        second, _ = _sample_observations(points, colors, 7, 42)
        self.assertEqual([len(item) for item in first], [7, 7])
        np.testing.assert_array_equal(first[0], second[0])
        self.assertFalse(np.array_equal(first[0], first[1]))

    def test_concatenation_preserves_every_sample(self):
        points = [np.zeros((2, 3), np.float32), np.ones((3, 3), np.float32)]
        colors = [item.copy() for item in points]
        xyz, rgb, stats = fuse_sampled_observations(points, colors, "concatenation")
        self.assertEqual(len(xyz), 5)
        np.testing.assert_array_equal(xyz, rgb)
        self.assertEqual(stats["rejected_observations"], 0)


class GeometryAndMetricTests(unittest.TestCase):
    def test_pair_thickness_detects_displaced_surface(self):
        x, y = np.meshgrid(np.arange(5), np.arange(4))
        source = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float32)
        target = source.copy(); target[..., 2] += 0.1
        mask = np.ones((4, 5), bool)
        result = _pair_geometry_metrics(source, mask, target, mask, np.eye(3), np.eye(4))
        self.assertAlmostEqual(result["median_world_thickness"], 0.1, places=5)
        self.assertAlmostEqual(result["median_relative_disagreement"], 0.1, places=5)

    def test_edge_metrics_penalize_shifted_edge(self):
        target = np.zeros((16, 16, 3), np.float32); target[:, 8:] = 1
        shifted = np.zeros_like(target); shifted[:, 9:] = 1
        result = image_metrics(shifted, target, lambda _a, _b: 0.5)
        self.assertLess(result["edge_psnr"], result["psnr"])
        self.assertGreater(result["gradient_mae"], 0)


class ResumeTests(unittest.TestCase):
    def test_resume_requires_exact_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            _json_write(path, {"seed": 1, "views": [53]})
            _assert_resume_compatible(path, {"seed": 1, "views": [53]})
            with self.assertRaises(RuntimeError):
                _assert_resume_compatible(path, {"seed": 2, "views": [53]})


if __name__ == "__main__":
    unittest.main()
