# π₀.₅ 训练流程与注意事项

本文说明本仓库**当前代码**如何准备数据并启动 π₀.₅ 微调。命令以 Ubuntu/Bash 为例，均在仓库根目录运行。环境安装与服务器初始化命令见 [README](../README.md)。

## 代码分工

`scripts/pipeline.py` 是本仓库的入口，提供 `inspect`、`convert`、`repair`、`download`、`train` 五个动作。它检查本地数据，调用 LeRobot v0.6.1 的转换器与 `lerobot-train`；模型结构、数据加载、损失计算和优化器均由 LeRobot 实现。本仓库没有自行实现 π₀.₅ 的训练循环。

```text
原始 LeRobot v2.1 数据 ──inspect──> convert ──> v3.0 数据 ──inspect──┐
                                  └──> 修复旧 Parquet 字段             ├──> train ──> outputs/ 检查点
Hugging Face 基座权重 ────────────────download─────────────────────────┘
```

## 训练前确认

1. 安装 Python、FFmpeg、CUDA 版 PyTorch 和本地 LeRobot v0.6.1，并确保 `lerobot-train` 可执行。安装步骤见 [README 的环境安装章节](../README.md#2-安装环境)。
2. 把数据放在默认路径 `datasets/Lerobot_1028_12_dim/`，保留完整的 `meta/`、`data/`、`videos/` 目录；也可在命令中使用 `--dataset` 指定其他路径。
3. 确认 GPU 可用，并为基座模型、原始数据备份、转换结果和训练输出预留磁盘空间。
4. 登录有权访问 `google/paligemma-3b-pt-224` 的 Hugging Face 账号。`download` 只下载 `lerobot/pi05_base`；训练所需的 gated PaliGemma tokenizer 还需要相应访问权限。

当前数据元信息为 75 个 episode、41,952 帧、30 FPS、两个任务；视频是 `observation.images.head` 与 `observation.images.wrist`，状态和动作均为 12 维 `float32`。脚本会检查这些维度、相机文件和元信息。末尾六维是零填充；README 记录动作为绝对关节位置。换数据集时，必须重新核对维度、相机键、动作语义及归一化统计。

## 按顺序执行

### 1. 检查原始数据

```bash
python -m scripts.pipeline inspect
```

v2.1 预检会核对 episode/task 元信息、总帧数、各 episode 的 Parquet 和视频文件，以及状态/动作的 `q01`、`q99` 统计。它不解码视频，也不执行 GPU 训练，因此通过预检仍需做短跑验证。

### 2. 下载基座权重

```bash
hf auth login
python -m scripts.pipeline download
test -f models/pi05_base/config.json
```

默认下载到 `models/pi05_base/`。`train` 会使用 `--policy.pretrained_path` 加载这里的权重；若模型已在其他位置，训练时指定 `--weights`。

### 3. 转换数据并复检

```bash
python -m scripts.pipeline convert
python -m scripts.pipeline inspect
```

`convert` 调用本地 `lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py`，传入数据根目录并强制 `--push-to-hub=false`。LeRobot 转换器会在原位置生成 v3.0 数据，保留 `_old` 原始备份。转换完成后，本脚本还会移除 Parquet 中未在 `meta/info.json` 声明的旧字段 `next_reward`、`next_done`，并为每个修改的文件留下 `.before_schema_fix` 备份。已是 v3.0 时再次运行 `convert` 会跳过转换。

如果以前转换过、现在 `inspect` 报 Parquet 列与 `meta/info.json` 不一致，可执行：

```bash
python -m scripts.pipeline repair
python -m scripts.pipeline inspect
```

`repair` 只接受上述两个多余字段；若缺少已声明字段、遇到其他额外字段或已有同名备份，它会停止。转换中断时先检查数据目录及 `_old`、`_v30` 目录，不要直接删除备份或覆盖数据。

### 4. 短跑并正式训练

```bash
python -m scripts.pipeline train --steps 2 --batch-size 1 --workers 0 --output outputs/smoke
python -m scripts.pipeline train --steps 30000 --batch-size 8 --workers 4 --output outputs/pi05_vial
```

短跑用于实际验证视频解码、tokenizer 访问、CUDA 和模型前向/反向路径。正式训练建议用 `tmux` 保持会话，观察日志、GPU 显存和磁盘空间。训练输出由 LeRobot 写入 `--output` 指定的目录。

## 当前训练配置

`train` 先要求数据已是 v3.0，并检查权重目录中存在 `config.json`，随后构造 `lerobot-train` 命令。关键默认值如下：

| 项目 | 当前值 | 影响 |
| --- | --- | --- |
| 数据 | `--dataset.repo_id=local/pi05_vial`、本地 `--dataset.root` | 训练读取本地 v3.0 数据；训练 repo ID 在代码中固定 |
| 视频 | `--dataset.video_backend=pyav` | 使用 PyAV 解码 |
| 图像统计 | `--dataset.use_imagenet_stats=false` | 避免对缺失的 ImageNet 图像统计写入 |
| 策略 | `--policy.type=pi05`、本地 `--policy.pretrained_path` | 从基座权重微调 |
| 精度与显存 | CUDA、`bfloat16`、梯度检查点 | 降低显存压力，仍需按实际显存调整 batch |
| 默认微调范围 | `freeze_vision_encoder=true`、`train_expert_only=true` | 默认训练 action expert；`--full-finetune` 会把两项设为 `false` |
| 动作预测 | `chunk_size=50`、`n_action_steps=10`、`empty_cameras=0` | 与部署端的动作执行方式和相机输入保持一致 |
| 训练规模 | 30,000 steps、batch 8、workers 4 | 可用 `--steps`、`--batch-size`、`--workers` 调整 |
| 记录与保存 | `save_freq=1000`、`log_freq=20`、seed 1000 | 每 1,000 步保存、每 20 步记录 |
| 在线服务 | `policy.push_to_hub=false`、`wandb.enable=false` | 默认不上传策略、不启用 W&B |

可用 `--dry-run` 打印将执行的命令；但它仍会做数据和权重的前置检查。`--full-finetune` 会提高显存需求，建议从较小 batch 开始。训练命令末尾还可追加未封装的 LeRobot 参数；若参数与脚本默认项重复，应检查最终打印的命令及 LeRobot 的解析结果。

## π₀.₅ 训练的关键注意事项

- **动作语义和填充维度：**当前数据的 12 维中只有前六维对应关节/夹爪，后六维为零填充。部署时须保持相同的维度顺序、单位和绝对动作语义；不要把零填充当作真实关节输出。
- **统计与归一化：**v2.1 数据必须具备 12 维 `q01/q99`。换数据后应重新生成与新数据匹配的统计，不能沿用当前数据集的数值。
- **相机与语言条件：**保持训练时两路相机键、视角和任务文本的含义一致；改变相机布局或任务描述时先核对数据特征与模型输入。脚本只检查至少存在一路视频，不会验证真实画面质量或任务文本是否正确。
- **转换的磁盘开销：**v2.1 原始备份、v3.0 文件和 schema 修复备份可能并存。确认转换、复检和训练正常后，再自行决定备份的保留策略。
- **显存与吞吐：**先完成两步短跑，再逐步调整 batch 和 worker。出现显存不足时优先降低 batch；启用 `--full-finetune` 前重新评估显存。
- **验证效果：**loss 下降仅说明训练目标在优化。部署前还需确认动作范围、执行频率和真实机器人任务成功率。

## 续训与常用检查

已有 LeRobot 训练配置时，可按 README 中的方式用 `lerobot-train` 续训：

```bash
find outputs/pi05_vial -name train_config.json
lerobot-train --config_path=/实际路径/train_config.json --resume=true
```

`pipeline.py` 本身没有 `resume` 动作。自定义路径和参数可先查看帮助并做 dry run：

```bash
python -m scripts.pipeline --help
python -m scripts.pipeline train --dataset /data/my_dataset --weights /models/pi05_base --output /runs/exp1 --steps 1000 --batch-size 4 --workers 2 --dry-run
```
