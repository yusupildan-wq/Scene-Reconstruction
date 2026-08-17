from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from experiments.loop_closure_pairs import find_verified_loop_closures
from experiments.reconstruction_graph import (
    COLAB_PROFILE,
    _assert_connected,
    temporal_pair_indices,
)


class VerifiedLoopClosureTests(unittest.TestCase):
    def test_temporal_graph_is_connected_and_has_expected_edges(self):
        pairs = temporal_pair_indices(6, radius=2)
        _assert_connected(6, pairs)
        self.assertIn((0, 2), pairs)
        self.assertIn((4, 5), pairs)
        self.assertNotIn((0, 3), pairs)
        self.assertEqual(COLAB_PROFILE.max_frames, 64)

    def test_accepts_real_geometric_revisit_and_rejects_blank_lookalike(self):
        rng = np.random.default_rng(7)
        textured = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        for row in range(40, 440, 80):
            cv2.putText(
                textured,
                f"shelf-{row}",
                (40, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
        transform = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [639, 0], [639, 479], [0, 479]]),
            np.float32([[18, 12], [620, 4], [630, 465], [8, 475]]),
        )
        revisit = cv2.warpPerspective(textured, transform, (640, 480))
        blank = np.full_like(textured, 190)

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            # Put enough temporal distance between the original and revisit.
            images = [textured] + [blank] * 7 + [revisit]
            for index, image in enumerate(images):
                path = Path(directory) / f"frame_{index:06d}.jpg"
                cv2.imwrite(str(path), image)
                paths.append(path)
            edges = find_verified_loop_closures(
                paths,
                min_frame_gap=8,
                retrieval_candidates_per_frame=8,
                similarity_threshold=-1.0,
                min_spatial_coverage=0.15,
            )

        pairs = {(edge.first, edge.second) for edge in edges}
        self.assertIn((0, 8), pairs)
        self.assertFalse(any(first != 0 and second != 8 for first, second in pairs))


if __name__ == "__main__":
    unittest.main()
