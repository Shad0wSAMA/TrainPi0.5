import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts.eval_joints import compare_episode, joint_metrics, plot_joints


class EvalJointsTest(unittest.TestCase):
    def test_chunks_use_current_observation_and_clip_final_chunk(self):
        rows = [{"action": np.array([i, i + 1], dtype=float), "frame": i} for i in range(5)]
        starts = []

        def predict(row):
            starts.append(row["frame"])
            return np.array([[row["frame"] + j, row["frame"] + j + 1] for j in range(3)])

        actions = np.stack([row["action"] for row in rows])
        gt, pred, inference = compare_episode(actions, rows.__getitem__, predict, horizon=3)
        self.assertEqual(starts, [0, 3])
        np.testing.assert_array_equal(gt, pred)
        self.assertEqual(inference, [0, 3])
        self.assertEqual(pred.shape, (5, 2))

    def test_metrics_are_in_original_action_units(self):
        gt = np.array([[0.0, 2.0], [2.0, 4.0]])
        pred = np.array([[1.0, 2.0], [1.0, 6.0]])
        rows = joint_metrics(gt, pred, ["joint", "gripper"])
        self.assertEqual(rows[0]["mse"], 1.0)
        self.assertEqual(rows[0]["mae"], 1.0)
        self.assertEqual(rows[1]["mse"], 2.0)
        self.assertEqual(rows[1]["mae"], 1.0)

    def test_short_prediction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prediction chunk"):
            compare_episode(np.zeros((1, 2)), lambda _: {}, lambda _: np.zeros((0, 2)), horizon=3)

    def test_plot_writes_one_figure_for_joint_trajectories(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "joints.png"
            truth = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
            plot_joints(truth, truth + 0.5, [0, 2], ["shoulder", "gripper"], path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
