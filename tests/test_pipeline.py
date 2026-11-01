import json
import tempfile
import unittest
from pathlib import Path

from scripts.pipeline import build_train_command, inspect_dataset, make_conversion_command, unexpected_parquet_columns


def sample_dataset(root: Path, version: str = "v2.1") -> None:
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": version,
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 3,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [12]},
            "action": {"dtype": "float32", "shape": [12]},
            "observation.images.head": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text('{"task_index":0,"task":"Move vial"}\n')
    (root / "meta" / "episodes.jsonl").write_text(
        '{"episode_index":0,"length":3,"tasks":["Move vial"]}\n'
    )
    (root / "meta" / "episodes_stats.jsonl").write_text('{}\n')
    (root / "meta" / "stats.json").write_text(
        json.dumps({key: {"q01": [0] * 12, "q99": [1] * 12} for key in ("action", "observation.state")})
    )
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000" / "episode_000000.parquet").touch()
    for camera in ("observation.images.head", "observation.images.wrist"):
        folder = root / "videos" / "chunk-000" / camera
        folder.mkdir(parents=True)
        (folder / "episode_000000.mp4").touch()


class PipelineTest(unittest.TestCase):
    def test_unexpected_parquet_columns_excludes_video_features(self):
        features = {
            "action": {"dtype": "float32"},
            "observation.state": {"dtype": "float32"},
            "observation.images.head": {"dtype": "video"},
            "next_reward": {"dtype": "float32"},
        }
        columns = ["action", "observation.state", "next_done", "next_reward"]
        self.assertEqual(unexpected_parquet_columns(features, columns), ["next_done"])

    def test_inspect_validates_actual_files_and_features(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp) / "dataset"
            sample_dataset(root)
            result = inspect_dataset(root)
            self.assertEqual(result["version"], "v2.1")
            self.assertEqual(result["cameras"], ["observation.images.head", "observation.images.wrist"])
            (root / "videos" / "chunk-000" / "observation.images.wrist" / "episode_000000.mp4").unlink()
            with self.assertRaisesRegex(ValueError, "Missing video"):
                inspect_dataset(root)

    def test_conversion_is_local_only_and_explicit(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            base = Path(tmp)
            converter = base / "lerobot" / "src" / "lerobot" / "scripts" / "convert_dataset_v21_to_v30.py"
            converter.parent.mkdir(parents=True)
            converter.touch()
            command = make_conversion_command(base / "lerobot", base / "datasets" / "demo", "local/demo")
            self.assertIn("--push-to-hub=false", command)
            self.assertIn(f"--root={(base / 'datasets' / 'demo').resolve()}", command)
            self.assertEqual(command[1], str(converter.resolve()))

    def test_train_command_uses_base_weights_and_local_dataset(self):
        command = build_train_command(Path("/data/demo"), Path("/weights/pi05_base"), Path("/runs/demo"), 8, 100, 2)
        self.assertIn(f"--dataset.root={Path('/data/demo').resolve()}", command)
        self.assertIn(f"--policy.pretrained_path={Path('/weights/pi05_base').resolve()}", command)
        self.assertIn("--policy.type=pi05", command)
        self.assertIn("--policy.dtype=bfloat16", command)
        self.assertIn("--policy.gradient_checkpointing=true", command)
        self.assertIn("--policy.push_to_hub=false", command)
        self.assertIn("--dataset.video_backend=pyav", command)
        self.assertIn("--dataset.use_imagenet_stats=false", command)
        self.assertNotIn("--policy.path=/weights/pi05_base", command)
        full = build_train_command(Path("/data/demo"), Path("/weights/pi05_base"), Path("/runs/demo"), 8, 100, 2, full_finetune=True)
        self.assertIn("--policy.train_expert_only=false", full)
        self.assertIn("--policy.freeze_vision_encoder=false", full)

    def test_train_command_accepts_log_and_save_frequencies(self):
        command = build_train_command(
            Path("/data/demo"), Path("/weights/pi05_base"), Path("/runs/demo"),
            8, 100, 2, log_freq=5, save_freq=50,
        )
        self.assertIn("--log_freq=5", command)
        self.assertIn("--save_freq=50", command)
        self.assertNotIn("--log_freq=20", command)
        self.assertNotIn("--save_freq=1000", command)


if __name__ == "__main__":
    unittest.main()
