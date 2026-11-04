import json
import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.pipeline import (
    build_train_command, ensure_quantile_stats_match, inspect_dataset,
    main, make_conversion_command, repair_quantile_stats, run_training_with_metrics,
    unexpected_parquet_columns,
)


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
    def test_training_uses_lerobot_default_episode_padding(self):
        command = build_train_command(Path("/data/demo"), Path("/weights/pi05_base"), Path("/runs/demo"), 8, 100, 2)
        self.assertEqual(command[0], "lerobot-train")

    def test_train_command_sets_prediction_chunk_size(self):
        command = build_train_command(Path("/data/demo"), Path("/weights/pi05_base"), Path("/runs/demo"), 8, 100, 2, chunk_size=24)
        self.assertIn("--policy.chunk_size=24", command)
        self.assertIn("--policy.n_action_steps=10", command)
        short = build_train_command(Path("/data/demo"), Path("/weights/pi05_base"), Path("/runs/demo"), 8, 100, 2, chunk_size=6)
        self.assertIn("--policy.n_action_steps=6", short)
        with self.assertRaisesRegex(ValueError, "chunk-size"):
            build_train_command(Path("/data/demo"), Path("/weights/pi05_base"), Path("/runs/demo"), 8, 100, 2, chunk_size=0)

    def test_train_cli_passes_chunk_size_to_lerobot(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            weights = Path(tmp) / "weights"
            weights.mkdir()
            (weights / "config.json").touch()
            argv = ["pipeline", "train", "--weights", str(weights), "--chunk-size", "6", "--dry-run"]
            output = io.StringIO()
            with patch.object(sys, "argv", argv), patch("scripts.pipeline.inspect_dataset", return_value={"version": "v3.0"}), patch("scripts.pipeline.ensure_quantile_stats_match"), redirect_stdout(output):
                main()
            self.assertIn("--policy.chunk_size=6", output.getvalue())
            self.assertIn("--policy.n_action_steps=6", output.getvalue())

    def test_repairs_only_quantiles_and_keeps_backup(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp) / "dataset"
            meta = root / "meta"
            meta.mkdir(parents=True)
            source = Path(tmp) / "original.json"
            current = {key: {"count": [10], "min": [0, 0], "max": [10, 10],
                             "mean": [5, 5], "q01": [4, 4], "q99": [6, 6]}
                       for key in ("action", "observation.state")}
            original = json.loads(json.dumps(current))
            original["action"]["q01"] = [1, 2]
            original["action"]["q99"] = [8, 9]
            (meta / "stats.json").write_text(json.dumps(current))
            source.write_text(json.dumps(original))
            self.assertTrue(repair_quantile_stats(root, source))
            fixed = json.loads((meta / "stats.json").read_text())
            self.assertEqual(fixed["action"]["q01"], [1, 2])
            self.assertEqual(fixed["action"]["mean"], [5, 5])
            self.assertEqual(json.loads((meta / "stats.json.before_quantile_fix").read_text()), current)
            self.assertFalse(repair_quantile_stats(root, source))

    def test_train_guard_rejects_wrong_quantiles_when_original_backup_exists(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp) / "dataset"
            backup = Path(tmp) / "dataset_old"
            (root / "meta").mkdir(parents=True)
            (backup / "meta").mkdir(parents=True)
            stats = {key: {"count": [10], "min": [0], "max": [10], "mean": [5],
                           "q01": [2], "q99": [8]} for key in ("action", "observation.state")}
            original = json.loads(json.dumps(stats))
            original["action"]["q99"] = [9]
            (root / "meta" / "stats.json").write_text(json.dumps(stats))
            (backup / "meta" / "stats.json").write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "fix-stats"):
                ensure_quantile_stats_match(root)
            repair_quantile_stats(root, backup / "meta" / "stats.json")
            ensure_quantile_stats_match(root)

    def test_repair_refuses_statistics_from_different_dataset(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp) / "dataset"
            (root / "meta").mkdir(parents=True)
            source = Path(tmp) / "original.json"
            current = {key: {"count": [10], "min": [0], "max": [10], "mean": [5],
                             "q01": [2], "q99": [8]} for key in ("action", "observation.state")}
            original = json.loads(json.dumps(current))
            original["action"]["count"] = [11]
            (root / "meta" / "stats.json").write_text(json.dumps(current))
            source.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "count"):
                repair_quantile_stats(root, source)
            self.assertFalse((root / "meta" / "stats.json.before_quantile_fix").exists())

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

    def test_train_records_loss_and_full_console_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output = Path(tmp) / "run"
            code = (
                "import sys; "
                "print('start'); "
                "print('INFO step:20 smpl:160 loss:0.750 grdn:1.250 lr:2.5e-5', file=sys.stderr); "
                "print('INFO step:40 smpl:320 loss:0.500 grdn:1.000 lr:2.0e-5', file=sys.stderr)"
            )
            log_path, csv_path = run_training_with_metrics([sys.executable, "-c", code], output, log_freq=20)
            self.assertIn("start", log_path.read_text(encoding="utf-8"))
            self.assertIn("loss:0.750", log_path.read_text(encoding="utf-8"))
            with csv_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["step"] for row in rows], ["20", "40"])
            self.assertEqual([row["loss"] for row in rows], ["0.75", "0.5"])

    def test_logger_does_not_create_training_output_before_child_starts(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output = Path(tmp) / "new_run"
            code = (
                "import pathlib,sys; "
                "p=pathlib.Path(sys.argv[1]); "
                "assert not p.exists(), 'output directory existed before trainer validation'; "
                "p.mkdir(); print('step:20 loss:0.25')"
            )
            log_path, csv_path = run_training_with_metrics(
                [sys.executable, "-c", code, str(output)], output, log_freq=20,
            )
            self.assertEqual(log_path.parent, output / "logs")
            self.assertEqual(csv_path.parent, output / "logs")
            self.assertIn("loss:0.25", log_path.read_text(encoding="utf-8"))

    def test_train_log_preserved_when_child_fails(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output = Path(tmp) / "run"
            code = "import sys; print('INFO step:20 loss:1.000', file=sys.stderr); sys.exit(7)"
            with self.assertRaisesRegex(Exception, "7"):
                run_training_with_metrics([sys.executable, "-c", code], output, log_freq=20)
            self.assertFalse(output.exists())
            self.assertEqual(len(list((output.parent / "run_train_logs").glob("*.csv"))), 1)
            self.assertEqual(len(list((output.parent / "run_train_logs").glob("*.log"))), 1)

    def test_train_recovers_steps_after_lerobot_rounds_labels_to_k(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            code = (
                "print('step:980 loss:0.8'); "
                "print('step:1K loss:0.7'); "
                "print('step:1K loss:0.6')"
            )
            _, csv_path = run_training_with_metrics([sys.executable, "-c", code], Path(tmp), log_freq=20)
            with csv_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["step"] for row in rows], ["980", "1000", "1020"])


if __name__ == "__main__":
    unittest.main()
