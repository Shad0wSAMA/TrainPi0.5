"""Offline, teacher-forced joint-trajectory comparison for a trained PI0.5 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .pipeline import DEFAULT_DATASET, REPO_ROOT


def compare_episode(
    actions: np.ndarray,
    sample_at: Callable[[int], dict],
    predict_chunk: Callable[[dict], np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Compare nonoverlapping predictions to recorded actions at matching indices."""
    truth = np.asarray(actions, dtype=np.float64)
    if truth.ndim != 2 or len(truth) == 0 or horizon < 1:
        raise ValueError("actions must be a nonempty [time, dim] array; horizon must be positive")
    predicted = np.empty_like(truth)
    starts = list(range(0, len(truth), horizon))
    for start in starts:
        chunk = np.asarray(predict_chunk(sample_at(start)), dtype=np.float64)
        count = min(horizon, len(truth) - start)
        if chunk.ndim != 2 or chunk.shape[0] < count or chunk.shape[1] != truth.shape[1]:
            raise ValueError(f"prediction chunk at {start} must have at least {count} rows and {truth.shape[1]} columns")
        predicted[start : start + count] = chunk[:count]
    return truth, predicted, starts


def select_frame_window(actions: np.ndarray, start_frame: int, max_frames: int | None) -> tuple[np.ndarray, int]:
    """Select a contiguous episode window while preserving its absolute start index."""
    if start_frame < 0 or start_frame >= len(actions):
        raise ValueError(f"start-frame must be between 0 and {len(actions) - 1}")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be positive")
    stop = None if max_frames is None else start_frame + max_frames
    return actions[start_frame:stop], start_frame


def joint_metrics(truth: np.ndarray, predicted: np.ndarray, names: list[str]) -> list[dict]:
    if truth.shape != predicted.shape or truth.ndim != 2 or truth.shape[1] != len(names):
        raise ValueError("truth, prediction, and joint names have incompatible shapes")
    error = predicted - truth
    return [
        {"joint": name, "mse": float(np.mean(error[:, i] ** 2)), "mae": float(np.mean(np.abs(error[:, i])))}
        for i, name in enumerate(names)
    ]


def plot_joints(
    truth: np.ndarray, predicted: np.ndarray, starts: list[int], names: list[str], path: Path,
    frame_offset: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = joint_metrics(truth, predicted, names)
    fig, axes = plt.subplots(len(names), 1, figsize=(15, max(3 * len(names), 5)), sharex=True, squeeze=False)
    x = np.arange(len(truth)) + frame_offset
    for i, (ax,) in enumerate(axes):
        ax.plot(x, truth[:, i], label="gt action", linewidth=1.5)
        ax.plot(x, predicted[:, i], label="pred action", linewidth=1.2)
        ax.scatter(np.asarray(starts) + frame_offset, truth[starts, i], c="red", s=18, zorder=3, label="inference point")
        ax.set_title(f"{names[i]}  |  MSE={metrics[i]['mse']:.4g}  MAE={metrics[i]['mae']:.4g}")
        ax.set_ylabel("Action value")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1, 0].set_xlabel("Episode frame (30 FPS in current dataset)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "config.json").is_file():
        raise ValueError(f"Expected a policy checkpoint directory containing config.json: {checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = LeRobotDataset(args.repo_id, root=args.dataset, episodes=[args.episode], video_backend="pyav")
    if len(dataset) == 0:
        raise ValueError(f"Episode {args.episode} has no frames")
    names = dataset.features["action"].get("names") or [f"action_{i}" for i in range(len(dataset.get_raw_item(0)["action"]))]
    active_dims = [i for i, name in enumerate(names) if not name.startswith("padding_")]
    if not active_dims:
        raise ValueError("No non-padding action dimensions found")
    raw_actions = dataset.select_columns("action")["action"]
    actions = np.asarray(raw_actions, dtype=np.float64)

    policy = PI05Policy.from_pretrained(str(checkpoint)).to(args.device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    horizon = args.horizon or policy.config.n_action_steps
    if horizon > policy.config.chunk_size:
        raise ValueError("horizon cannot exceed policy chunk_size")

    def predict(sample: dict) -> np.ndarray:
        observation = {key: value for key, value in sample.items() if key.startswith("observation.") or key == "task"}
        for camera in dataset.meta.camera_keys:
            if camera in observation and observation[camera].dtype == torch.uint8:
                observation[camera] = observation[camera].float() / 255.0
        with torch.inference_mode():
            batch = preprocessor(observation)
            normalized = policy.predict_action_chunk(batch)[0, :horizon]
            physical = [postprocessor(step.unsqueeze(0)).squeeze(0).detach().cpu().numpy() for step in normalized]
        return np.stack(physical)

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    window, frame_offset = select_frame_window(actions, args.start_frame, args.max_frames)
    truth, predicted, starts = compare_episode(window, lambda i: dataset[frame_offset + i], predict, horizon)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shown_names = [names[i] for i in active_dims]
    metrics = joint_metrics(truth[:, active_dims], predicted[:, active_dims], shown_names)
    plot_joints(
        truth[:, active_dims], predicted[:, active_dims], starts, shown_names,
        output / f"episode_{args.episode:03d}.png", frame_offset=frame_offset,
    )
    with (output / f"episode_{args.episode:03d}_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["joint", "mse", "mae"])
        writer.writeheader()
        writer.writerows(metrics)
    with (output / f"episode_{args.episode:03d}_trajectory.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "inference_point"] + [f"gt_{name}" for name in shown_names] + [f"pred_{name}" for name in shown_names])
        for i in range(len(truth)):
            writer.writerow([i + frame_offset, int(i in starts)] + truth[i, active_dims].tolist() + predicted[i, active_dims].tolist())
    print(json.dumps({"episode": args.episode, "start_frame": frame_offset, "frames": len(truth), "horizon": horizon, "metrics": metrics, "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="LeRobot policy directory containing config.json and saved processors")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", default="local/pi05_vial")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0, help="First episode frame to evaluate")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit evaluated frames; default entire episode")
    parser.add_argument("--horizon", type=int, default=None, help="Executed predictions per inference; default checkpoint n_action_steps")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "eval_joints")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
