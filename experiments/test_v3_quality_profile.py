import argparse
import unittest

from experiments.run_v3_vggt import resolve_quality_profile


class V3QualityProfileTests(unittest.TestCase):
    def test_high_profile(self):
        args = argparse.Namespace(quality_profile="high", data_factor=None, max_steps=None,
                                  pose_opt=None, antialiased=None)
        self.assertEqual(resolve_quality_profile(args), {
            "data_factor": 1, "max_steps": 30_000,
            "pose_opt": True, "antialiased": True,
        })

    def test_explicit_overrides(self):
        args = argparse.Namespace(quality_profile="high", data_factor=2, max_steps=12_000,
                                  pose_opt=False, antialiased=False)
        config = resolve_quality_profile(args)
        self.assertEqual((config["data_factor"], config["max_steps"]), (2, 12_000))
        self.assertFalse(config["pose_opt"])
        self.assertFalse(config["antialiased"])


if __name__ == "__main__":
    unittest.main()
