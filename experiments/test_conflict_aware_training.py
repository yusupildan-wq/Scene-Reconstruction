import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:
    torch = None


from worker.training_schedule import (  # noqa: E402
    camera_cycle_densification_schedule as _camera_cycle_densification_schedule,
    camera_order_for_cycle as _camera_order_for_cycle,
)
from experiments.run_full_room_gaussians import (  # noqa: E402
    _image_quality,
    _load_exposure_parameters,
)


class CameraCycleSchedulingTests(unittest.TestCase):
    def test_sequential_mode_preserves_baseline_order(self):
        np.testing.assert_array_equal(
            _camera_order_for_cycle(4, cycle=99), np.array([0, 1, 2, 3])
        )

    def test_shuffled_cycle_is_balanced_deterministic_and_cycle_specific(self):
        first = _camera_order_for_cycle(16, cycle=7, mode="shuffled_cycle", seed=3)
        resumed = _camera_order_for_cycle(
            16, cycle=7, mode="shuffled_cycle", seed=3
        )
        following = _camera_order_for_cycle(
            16, cycle=8, mode="shuffled_cycle", seed=3
        )
        np.testing.assert_array_equal(np.sort(first), np.arange(16))
        np.testing.assert_array_equal(first, resumed)
        self.assertFalse(np.array_equal(first, following))

    def test_densification_interval_and_start_cover_whole_camera_cycles(self):
        start, interval = _camera_cycle_densification_schedule(
            camera_count=126,
            requested_from=500,
            start_step=0,
            cycles_per_refinement=1,
        )
        self.assertEqual(interval, 126)
        self.assertEqual(start, 504)
        self.assertEqual(start % 126, 0)

    def test_invalid_cycle_configuration_fails_loudly(self):
        with self.assertRaises(ValueError):
            _camera_cycle_densification_schedule(8, 500, 0, 0)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ExposureEvaluationTests(unittest.TestCase):
    def test_loads_matching_exposure_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "checkpoints" / "training_state_latest.pt"
            checkpoint.parent.mkdir()
            torch.save(
                {
                    "step": 12,
                    "exposure_log_gains": torch.zeros((4, 3)),
                    "exposure_biases": torch.ones((4, 3)),
                },
                checkpoint,
            )
            gains, biases = _load_exposure_parameters(output, 4, expected_step=12)
            self.assertEqual(tuple(gains.shape), (4, 3))
            self.assertTrue(torch.equal(biases, torch.ones((4, 3))))

    def test_mismatched_exposure_state_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "checkpoints" / "training_state_latest.pt"
            checkpoint.parent.mkdir()
            torch.save(
                {
                    "step": 12,
                    "exposure_log_gains": torch.zeros((3, 3)),
                    "exposure_biases": torch.zeros((3, 3)),
                },
                checkpoint,
            )
            with self.assertRaises(RuntimeError):
                _load_exposure_parameters(output, 4)

    def test_mismatched_export_and_checkpoint_steps_fail_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "checkpoints" / "training_state_latest.pt"
            checkpoint.parent.mkdir()
            torch.save(
                {
                    "step": 11,
                    "exposure_log_gains": torch.zeros((4, 3)),
                    "exposure_biases": torch.zeros((4, 3)),
                },
                checkpoint,
            )
            with self.assertRaises(RuntimeError):
                _load_exposure_parameters(output, 4, expected_step=12)

    def test_quality_metrics_reward_correct_exposure_adjustment(self):
        target = torch.full((4, 4, 3), 0.5)
        canonical = torch.full((4, 4, 3), 0.25)
        adjusted = canonical * 2.0

        def exact_similarity(left, right):
            return 1.0 - torch.mean(torch.abs(left - right))

        base = _image_quality(canonical, target, exact_similarity)
        corrected = _image_quality(adjusted, target, exact_similarity)
        self.assertGreater(corrected["psnr"], base["psnr"])
        self.assertGreater(corrected["ssim"], base["ssim"])


if __name__ == "__main__":
    unittest.main()
