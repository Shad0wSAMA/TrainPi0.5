"""Prepare a local LeRobot dataset and launch Pi0.5 fine-tuning.

The actual dataset conversion and training are delegated to the pinned LeRobot checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "Lerobot_1028_12_dim"
DEFAULT_LEROBOT = REPO_ROOT / "lerobot"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "pi05_base"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing metadata: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing metadata: {path}")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def unexpected_parquet_columns(features: dict, columns: list[str]) -> list[str]:
    """Find payload columns absent from info.json's non-video feature map."""
    expected = {key for key, value in features.items() if value.get("dtype") != "video"}
    return sorted(set(columns) - expected)


def _parquet_schema_issues(root: Path, features: dict) -> list[tuple[Path, list[str], list[str]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to inspect v3 parquet; install LeRobot first") from exc
    expected = {key for key, value in features.items() if value.get("dtype") != "video"}
    issues = []
    for path in sorted((root / "data").rglob("*.parquet")):
        actual = pq.read_schema(path).names
        extra = unexpected_parquet_columns(features, actual)
        missing = sorted(expected - set(actual))
        if extra or missing:
            issues.append((path, extra, missing))
    return issues


def repair_parquet_schema(root: Path) -> int:
    """Drop known legacy reward columns from v3 parquet, keeping a per-file backup."""
    info = _read_json(root / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Schema repair requires a converted v3.0 dataset")
    import pyarrow.parquet as pq

    issues = _parquet_schema_issues(root, info["features"])
    for path, extra, missing in issues:
        if missing:
            raise ValueError(f"Parquet lacks declared features {missing}: {path}")
        unknown = sorted(set(extra) - {"next_reward", "next_done"})
        if unknown:
            raise ValueError(f"Refusing to remove unrecognized data columns {unknown}: {path}")
        backup = path.with_name(path.name + ".before_schema_fix")
        if backup.exists():
            raise ValueError(f"Backup already exists; inspect before retrying: {backup}")
        temp = path.with_name(path.name + ".schema_fix_tmp")
        original = pq.read_table(path)
        keep = [name for name in original.column_names if name not in extra]
        cleaned = original.select(keep).replace_schema_metadata(None)
        pq.write_table(cleaned, temp, compression="zstd")
        if pq.read_metadata(temp).num_rows != pq.read_metadata(path).num_rows:
            temp.unlink()
            raise ValueError(f"Row count changed during repair: {path}")
        shutil.copy2(path, backup)
        os.replace(temp, path)
        print(f"Removed undeclared columns {extra} from {path}; backup: {backup}", flush=True)
    return len(issues)


def inspect_dataset(root: Path, check_schema: bool = True) -> dict:
    """Check metadata and payload presence before a costly conversion or train run."""
    root = root.resolve()
    info = _read_json(root / "meta" / "info.json")
    version = info.get("codebase_version")
    if version not in ("v2.1", "v3.0"):
        raise ValueError(f"Unsupported dataset version: {version!r}")
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [12]:
            raise ValueError(f"Expected 12 float32 dimensions for {key}: {feature}")
    cameras = sorted(key for key, value in features.items() if value.get("dtype") == "video")
    if not cameras:
        raise ValueError("No video camera features found")
    episodes = info.get("total_episodes", 0)
    if not isinstance(episodes, int) or episodes < 1:
        raise ValueError("Dataset has no episodes")
    if version == "v2.1":
        rows = _jsonl(root / "meta" / "episodes.jsonl")
        tasks = _jsonl(root / "meta" / "tasks.jsonl")
        _jsonl(root / "meta" / "episodes_stats.jsonl")
        if len(rows) != episodes or not tasks:
            raise ValueError("Episode or task metadata count does not match info.json")
        if sum(row["length"] for row in rows) != info.get("total_frames"):
            raise ValueError("Episode lengths do not sum to total_frames")
        for row in rows:
            index = row["episode_index"]
            chunk = index // info["chunks_size"]
            data_path = root / info["data_path"].format(episode_chunk=chunk, episode_index=index)
            if not data_path.is_file():
                raise ValueError(f"Missing parquet: {data_path}")
            for camera in cameras:
                video_path = root / info["video_path"].format(
                    episode_chunk=chunk, episode_index=index, video_key=camera
                )
                if not video_path.is_file():
                    raise ValueError(f"Missing video: {video_path}")
        stats = _read_json(root / "meta" / "stats.json")
        for key in ("observation.state", "action"):
            for quantile in ("q01", "q99"):
                if len(stats.get(key, {}).get(quantile, [])) != 12:
                    raise ValueError(f"Missing 12D {quantile} statistics for {key}")
    else:
        if not list((root / "data").rglob("*.parquet")):
            raise ValueError("Missing v3 parquet data")
        for camera in cameras:
            if not list((root / "videos" / camera).rglob("*.mp4")):
                raise ValueError(f"Missing v3 video for {camera}")
        if not (root / "meta" / "tasks.parquet").is_file():
            raise ValueError("Missing v3 tasks.parquet")
        if not list((root / "meta" / "episodes").rglob("*.parquet")):
            raise ValueError("Missing v3 episode metadata")
        if check_schema:
            issues = _parquet_schema_issues(root, features)
            if issues:
                path, extra, missing = issues[0]
                raise ValueError(
                    f"Parquet columns differ from meta/info.json in {path}: "
                    f"undeclared={extra}, missing={missing}. Run 'repair' before training."
                )
    return {"root": str(root), "version": version, "episodes": episodes, "cameras": cameras}


def make_conversion_command(lerobot: Path, dataset: Path, repo_id: str) -> list[str]:
    converter = lerobot.resolve() / "src" / "lerobot" / "scripts" / "convert_dataset_v21_to_v30.py"
    if not converter.is_file():
        raise ValueError(f"LeRobot v0.6.1 converter missing: {converter}")
    return [sys.executable, str(converter), f"--repo-id={repo_id}", f"--root={dataset.resolve()}", "--push-to-hub=false"]


def build_train_command(
    dataset: Path, weights: Path, output: Path, batch_size: int, steps: int, workers: int,
    full_finetune: bool = False, log_freq: int = 20, save_freq: int = 1000,
) -> list[str]:
    if min(batch_size, steps) < 1 or workers < 0:
        raise ValueError("batch-size and steps must be positive; workers cannot be negative")
    if log_freq < 1:
        raise ValueError("log-freq must be positive")
    return [
        "lerobot-train",
        "--dataset.repo_id=local/pi05_vial",
        f"--dataset.root={dataset.resolve()}",
        "--dataset.video_backend=pyav",
        "--dataset.use_imagenet_stats=false",
        "--policy.type=pi05",
        f"--policy.pretrained_path={weights.resolve()}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        f"--policy.freeze_vision_encoder={str(not full_finetune).lower()}",
        f"--policy.train_expert_only={str(not full_finetune).lower()}",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=10",
        "--policy.empty_cameras=0",
        "--policy.push_to_hub=false",
        f"--output_dir={output.resolve()}",
        "--job_name=pi05_vial",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--num_workers={workers}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        "--seed=1000",
        "--wandb.enable=false",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "convert", "repair", "download", "train"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lerobot", type=Path, default=DEFAULT_LEROBOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "pi05_vial")
    parser.add_argument("--repo-id", default="local/pi05_vial")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=20, help="Log every N training steps")
    parser.add_argument("--save-freq", type=int, default=1000, help="Save every N steps; 0 saves only at the end")
    parser.add_argument("--full-finetune", action="store_true", help="Train the VLM as well; uses more VRAM")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if args.action != "train" and extra:
        parser.error("Extra LeRobot flags are only supported for train")
    if args.action == "download":
        from huggingface_hub import snapshot_download

        args.weights.mkdir(parents=True, exist_ok=True)
        print(f"Downloading lerobot/pi05_base to {args.weights.resolve()}", flush=True)
        snapshot_download(repo_id="lerobot/pi05_base", local_dir=args.weights)
        print("Base weights downloaded. The gated PaliGemma tokenizer requires HF access during training.")
        return
    summary = inspect_dataset(args.dataset, check_schema=args.action == "inspect" or args.action == "train")
    print(json.dumps(summary, indent=2), flush=True)
    if args.action == "inspect":
        return
    if args.action == "repair":
        if args.dry_run:
            print("Schema repair dry run: no files changed.")
            return
        count = repair_parquet_schema(args.dataset)
        print(f"Repaired {count} parquet files.")
        print(json.dumps(inspect_dataset(args.dataset), indent=2))
        return
    if args.action == "convert":
        if summary["version"] == "v3.0":
            print("Dataset already v3.0; conversion skipped.")
            return
        command = make_conversion_command(args.lerobot, args.dataset, args.repo_id)
    else:
        if summary["version"] != "v3.0":
            raise ValueError("Dataset is v2.1. Run 'convert' first.")
        if not (args.weights / "config.json").is_file():
            raise ValueError(f"Base checkpoint config.json missing in {args.weights}; run 'download' first")
        command = build_train_command(
            args.dataset, args.weights, args.output, args.batch_size, args.steps, args.workers,
            full_finetune=args.full_finetune, log_freq=args.log_freq, save_freq=args.save_freq,
        )
        command += extra
    print("Command:", " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)
        if args.action == "convert":
            repair_parquet_schema(args.dataset)
            print(json.dumps(inspect_dataset(args.dataset), indent=2))


if __name__ == "__main__":
    main()
