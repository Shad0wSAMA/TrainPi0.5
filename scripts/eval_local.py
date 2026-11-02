"""Run a short PI0.5 trajectory evaluation with a downloaded local checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np

from . import eval_joints
from .pipeline import DEFAULT_DATASET, REPO_ROOT


DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "pi05"
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def quantile_diagnostics(checkpoint: Path, dataset: Path) -> dict:
    """Compare saved normalizer quantiles with local dataset metadata."""
    stats_path = dataset / "meta" / "stats.json"
    normalizer_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    with stats_path.open(encoding="utf-8") as file:
        dataset_stats = json.load(file)
    with normalizer_path.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
        body = file.read()
    result = {}
    for feature in ("action", "observation.state"):
        result[feature] = {}
        for quantile in ("q01", "q99"):
            key = f"{feature}.{quantile}"
            entry = header[key]
            saved = np.frombuffer(body, dtype="<f4", count=entry["shape"][0], offset=entry["data_offsets"][0])
            local = np.asarray(dataset_stats[feature][quantile], dtype=np.float32)
            if saved.shape != local.shape:
                raise ValueError(f"Quantile shape differs for {key}: {saved.shape} vs {local.shape}")
            result[feature][quantile] = {
                "max_abs_diff": float(np.max(np.abs(saved - local))),
                "checkpoint_first_six": saved[:6].round(3).tolist(),
                "dataset_first_six": local[:6].round(3).tolist(),
            }
    return result


def validate_local_inputs(checkpoint: Path, dataset: Path) -> None:
    for name in REQUIRED_CHECKPOINT_FILES:
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    with info_path.open(encoding="utf-8") as file:
        version = json.load(file).get("codebase_version")
    if version != "v3.0":
        raise RuntimeError(
            f"Dataset is {version}, but LeRobot v0.6.1 evaluation requires v3.0. "
            "Install the pinned LeRobot checkout and run 'python -m scripts.pipeline convert' "
            "on a dataset copy, then pass that copy with --dataset."
        )


def check_dependencies() -> None:
    missing = [name for name in ("torch", "lerobot", "transformers", "safetensors", "av", "pyarrow", "matplotlib")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"Missing Python packages: {', '.join(missing)}. "
            "Install the pinned LeRobot environment as described in README.md, including training/pi extras."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", default="local/pi05_vial")
    parser.add_argument("--episode", type=int, default=7)
    parser.add_argument("--start-frame", type=int, default=390)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "eval_local")
    parser.add_argument("--check-only", action="store_true", help="Validate files and packages without loading the model")
    args = parser.parse_args()
    if (args.dataset / "meta" / "stats.json").is_file() and (
        args.checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    ).is_file():
        print("Saved normalizer vs local dataset quantiles:", flush=True)
        print(json.dumps(quantile_diagnostics(args.checkpoint, args.dataset), indent=2), flush=True)
    problems = []
    for check in (lambda: validate_local_inputs(args.checkpoint, args.dataset), check_dependencies):
        try:
            check()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            problems.append(str(exc))
    if problems:
        parser.exit(2, "Local evaluation is not ready:\n- " + "\n- ".join(problems) + "\n")
    if args.check_only:
        print("Local checkpoint, dataset version, and Python dependencies are ready.")
        return
    eval_joints.run(args)


if __name__ == "__main__":
    main()
