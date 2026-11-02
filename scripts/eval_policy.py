# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import load_data_config, _resolve_valid_index, build_data_config
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory, calc_mse_for_single_trajectory_rtc
import random, torch, yaml

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    plot: bool = True
    """Whether to plot the images."""

    modality_keys: List[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    """Modality keys to evaluate."""

    eval_valid_names: List[str] = field(default_factory=list)
    """Robot body groups to evaluate, e.g. left_arm. Empty means evaluate all dimensions."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.
    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """
    metadata_name: str = "metadata.json"  ###  ["metadata.json", metadata_valid-index.json]
    
    steps: int = 250
    """Number of steps to evaluate."""

    trajs: int = 1
    """Number of trajectories to evaluate."""

    start_traj: int = 0
    """Start trajectory to evaluate."""

    action_horizon: int = None
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    model_path: str = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str = "./plot.png"
    """Path to save the plot."""

    plot_state: bool = False
    """Whether to plot the state."""

    async_frames_num: int = 9

    num_terminal_frames: int = 5

    data_config_yaml: str = 'data_config.yaml'

    rtc: str = None  # None, rtc, simplertc(*)
    """Whether to enable RTC evaluation."""

    inference_delay: int = 0
    """Inference delay in action steps."""

    execution_horizon: int = 10
    """RTC execution horizon."""

    prefix_attention_schedule: Literal["LINEAR", "EXP", "ONES", "ZEROS"] = "EXP"
    """RTC prefix weighting schedule."""

    max_guidance_weight: float = 10.0
    
    absolute: bool = True
    """Whether to use absolute action for training. If False, model will be trained to predict Relative (Delta) actions."""
    
    relative_eef: bool = False
    """Whether to use absolute action for eval."""
    

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args: ArgsConfig):
    # data_config = load_data_config(config.data_config)
    if config.data_config == "unified_data_config":
        with open(config.data_config_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        data_config = build_data_config(config.data_config, cfg=cfg['data'])
        eval_valid_index = _resolve_valid_index(args.eval_valid_names, cfg['data']['robot_body_type'])
    else:
        data_config = load_data_config(config.data_config)
        eval_valid_index = None
    set_seed(42)

    # Set action_horizon from data config if not provided
    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Using action_horizon={args.action_horizon} from data config '{args.data_config}'")
    
    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        if not config.absolute:
            print("Infer with RELATIVE (Delta) Actions!")
            from gr00t.data.transform.state_action import RelativeChunkTransform
            from gr00t.data.transform.concat import ConcatTransform
            
            action_keys = data_config.action_keys
            state_keys = data_config.state_keys
            
            new_transforms = []
            for t in modality_transform.transforms:
                if isinstance(t, ConcatTransform):
                    new_transforms.append(RelativeChunkTransform(
                        apply_to=action_keys,
                        action_keys=action_keys,
                        state_keys=state_keys
                    ))
                new_transforms.append(t)
            modality_transform.transforms = new_transforms
        
        #import pdb; pdb.set_trace()
        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
            metadata_name=args.metadata_name
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)
    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
        async_frames_num=args.async_frames_num,
        num_terminal_frames=args.num_terminal_frames,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", args.modality_keys)

    all_mse = []
    for traj_id in range(args.start_traj, args.start_traj + args.trajs):
        print("Running trajectory:", traj_id)
        if args.rtc is not None:
            mse = calc_mse_for_single_trajectory_rtc(
                policy,
                dataset,
                traj_id,
                modality_keys=args.modality_keys,
                eval_valid_index=eval_valid_index,
                steps=args.steps,
                action_horizon=args.action_horizon,
                relative_eef_flag=args.relative_eef,
                rtc_type = args.rtc,
                inference_delay=args.inference_delay,
                execution_horizon=args.execution_horizon,
                prefix_attention_schedule=args.prefix_attention_schedule,
                max_guidance_weight=args.max_guidance_weight,
                plot=args.plot,
                plot_state=args.plot_state,
                save_plot_path=args.save_plot_path,
            )
        else:
            mse = calc_mse_for_single_trajectory(
                policy,
                dataset,
                traj_id,
                modality_keys=args.modality_keys,
                eval_valid_index=eval_valid_index,
                steps=args.steps,
                action_horizon=args.action_horizon,
                relative_eef_flag=args.relative_eef,
                plot=args.plot,
                plot_state=args.plot_state,
                save_plot_path=args.save_plot_path,
            )
        print("MSE:", mse)
        all_mse.append(mse)
    print("Average MSE across all trajs:", np.mean(all_mse))
    print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
