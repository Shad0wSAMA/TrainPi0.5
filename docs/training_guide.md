# π₀.₅ 训练流程与注意事项

本文说明本仓库**当前代码**如何准备数据并启动 π₀.₅ 微调。命令以 Ubuntu/Bash 为例，均在仓库根目录运行。环境安装与服务器初始化命令见 [README](../README.md)。

## 代码分工

`scripts/pipeline.py` 是本仓库的入口，提供 `inspect`、`convert`、`repair`、`download`、`train` 五个动作。它检查本地数据，调用 LeRobot v0.6.1 的转换器与训练入口；模型结构、数据加载、损失计算和优化器均由 LeRobot 实现。本仓库没有自行实现 π₀.₅ 的训练循环。

## π₀ 与本仓库使用的 π₀.₅

π₀ 系列属于视觉、语言、动作联合建模的机器人策略。本仓库训练的是 LeRobot `policy.type=pi05`，因此下面的细节以 **LeRobot v0.6.1 的 π₀.₅ 实现**为准，不能直接套用到其他 π₀ 实现。

### 输入、主干和动作专家

一次推理读取当前时刻的两路相机图像、`observation.state` 和任务文本。视觉编码器把图像转成视觉 token；状态先按训练集的 `q01/q99` 做分位数归一化，再离散成 256 个区间，与任务说明拼成语言提示并送进 PaliGemma 的语言主干。π₀.₅ 在此实现中**没有单独的连续状态分支**：状态包含在文本 token 中。图像会按比例缩放并填充到 224×224，进入视觉主干前转换到 `[-1, 1]`。参考 [LeRobot π₀.₅ 模型代码](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/modeling_pi05.py) 和 [输入/输出处理器](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/processor_pi05.py)。

PaliGemma 形成图像与语言条件的前缀；另一个 Gemma action expert 接收带噪动作块和时间步，输出动作流的速度估计。两个分支在联合注意力中交互。模型内部把较短的状态/动作维度填充到最多 32 维，但输出和损失会裁回数据实际的 12 维。当前 12 维中后六维本身又是数据集提供的零填充，因此查看运动关节时应重点看前六维。默认 `train_expert_only=true` 冻结视觉语言主干，训练动作专家及相关投影；`--full-finetune` 解除此限制。配置细节见 [π₀.₅ 配置源码](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/configuration_pi05.py)。

### 动作块与 flow matching

每次模型预测一个 50 帧的 action chunk。训练时，对真实动作块 `a` 采样噪声 `ε` 和时间 `t`，构造 `x_t = (1-t)a + tε`；网络预测目标速度 `ε-a`，并以逐维均方误差训练。它学习的是从噪声回到动作的流场，训练 loss **不是**直接计算预测关节值与真值关节值的误差。推理时从随机噪声出发，默认用 10 个积分步骤生成整块动作。本仓库设 `n_action_steps=10`，即每次只执行前 10 帧后再根据新观测重新规划。训练目标与推理实现见 [LeRobot π₀.₅ 模型代码](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/modeling_pi05.py)。

状态和动作的数值由预处理器按训练集统计归一化，模型输出由后处理器恢复到原始动作单位。评估或部署时必须加载**该训练检查点旁保存的预处理器与后处理器**，否则关节曲线的尺度没有可比性。当前配置 `use_relative_actions=false`，所以输出仍是绝对关节目标。参考 [处理器源码](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/processor_pi05.py)。

### 相对动作是否更适合这份数据

不能仅凭原始数据是绝对关节位置就决定关闭相对动作。启用 `use_relative_actions=true` 后，LeRobot 在训练前对选中的维度计算 `a_relative = a_absolute - 当前 observation.state`，模型学习从当前姿态到目标姿态的偏移；后处理再加回当前状态，给机器人输出绝对目标。默认 `relative_exclude_joints=["gripper"]`，名称含 `gripper` 的夹爪维度仍保持绝对值。实现见 [相对动作处理器](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/processor/relative_action_processor.py)。

这种表示可能更适合“从当前姿态移动到目标”的任务，但并非 π₀.₅ 的普遍最优设置。[官方 π₀.₅ 指南](https://huggingface.co/docs/lerobot/v0.6.1/pi05#relative-actions)也只说它在某些设置下可能改善稳定性。本仓库当前 `meta/stats.json` 的动作统计是**绝对动作分布**；若直接追加 `--policy.use_relative_actions=true`，处理器会先减去状态、再使用不匹配的绝对动作统计归一化，因此不应直接这样训练。应先在独立数据副本上按训练实际使用的 `chunk_size` 重算相对动作统计，再使用相同的排除维度配置训练，并保留原绝对动作实验作为对照。对这份数据，先确认动作与状态前六维逐一对应、单位一致，尤其要核对夹爪是否应保持绝对值；后六维零填充减去零状态仍为零。

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

转换器会将各 episode 的 `q01/q99` 按帧数加权平均；对于跨 episode 姿态分布差异大的数据，这不等于原始 v2.1 `meta/stats.json` 中的全局分位数。现在 `pipeline convert` 完成后会从 `_old/meta/stats.json` 恢复 `action` 和 `observation.state` 的全局 `q01/q99`，保留转换后的其他统计，并将修改前的 v3 统计备份为 `meta/stats.json.before_quantile_fix`。训练入口若发现 `_old` 中的全局分位数与当前 v3 数据不一致，会在启动前报错。

**服务器上已经转换的数据无需重新转换视频。** 更新仓库后运行：

```bash
python -m scripts.pipeline fix-stats
python -m scripts.pipeline inspect
```

`fix-stats` 会核对样本数、min/max、mean，避免误用另一份数据的统计；原始备份若不在默认的 `datasets/Lerobot_1028_12_dim_old/meta/stats.json`，可用 `--source-stats /原始v2.1数据/meta/stats.json` 指定。已经用旧分位数训练出的检查点需要从基座权重重新训练，不能只替换评估处理器，也不应从旧训练检查点续训。使用**新的**输出目录，例如 `outputs/pi05_vial_global_q`，避免混淆旧实验。

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

下面是列出全部常用可调训练参数的正式训练命令；路径和数值可按服务器环境修改：

```bash
python -m scripts.pipeline train \
  --dataset datasets/Lerobot_1028_12_dim \
  --weights models/pi05_base \
  --output outputs/pi05_vial \
  --source-stats datasets/Lerobot_1028_12_dim_old/meta/stats.json \
  --steps 30000 \
  --batch-size 8 \
  --chunk-size 50 \
  --workers 4 \
  --log-freq 20 \
  --save-freq 1000
```

`--source-stats` 可省略，此时脚本会自动查找上面所示的转换备份。需要训练整个视觉语言模型时，在命令末尾添加 `--full-finetune`；只想打印命令并做前置检查时添加 `--dry-run`。`--lerobot` 和 `--repo-id` 仅供转换使用，不会改变 `train` 的命令。其他 LeRobot 训练参数可以追加在末尾，但须以实际安装版本的 `lerobot-train --help` 为准。

一次训练更新中，LeRobot 从 episode 中采样帧和未来动作窗口，解码两路视频，准备任务文本，归一化状态与动作；然后计算 flow matching loss、反向传播、更新优化器与学习率计划。动作窗口越过 episode 末尾时，LeRobot 会重复该 episode 最后一帧动作来补齐，不会接到下一条 episode。终端日志中的 `loss` 是**归一化空间的流匹配损失**，并非实际关节单位下的跟踪 MSE。π₀.₅ 虽在前向结果中提供 `loss_per_dim` 列表，但当前 LeRobot 日志聚合器只记录数值标量，不会把该列表逐维写入终端。当前脚本每 20 步打印日志、每 1,000 步保存检查点。训练代码见 [LeRobot v0.6.1 训练入口](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_train.py)。

本仓库的 `pipeline train` 会把终端输出实时保存到 `--output` 同级的 `<输出目录名>_train_logs/train_<时间>.log`，并将每次打印的窗口平均 `loss`、梯度范数和学习率写入同级的 `loss_<时间>.csv`。训练成功后这两个文件移到 `--output/logs/`；失败时留在暂存目录。这样 LeRobot 启动前不会因为日志目录而发现已存在的输出目录。每次启动都会生成新文件；CSV 中的 loss 精度与 LeRobot 终端打印值相同，且不是每一步的原始 loss。当前脚本仍关闭 W&B。直接调用续训入口不会经过这个日志记录入口，需自行保存终端输出或启用 W&B。

## 当前训练配置

`train` 先要求数据已是 v3.0，并检查权重目录中存在 `config.json`，随后调用 `lerobot-train`。关键默认值如下：

| 项目 | 当前值 | 影响 |
| --- | --- | --- |
| 数据 | `--dataset.repo_id=local/pi05_vial`、本地 `--dataset.root` | 训练读取本地 v3.0 数据；训练 repo ID 在代码中固定 |
| 视频 | `--dataset.video_backend=pyav` | 使用 PyAV 解码 |
| 图像统计 | `--dataset.use_imagenet_stats=false` | 避免对缺失的 ImageNet 图像统计写入 |
| 策略 | `--policy.type=pi05`、本地 `--policy.pretrained_path` | 从基座权重微调 |
| 精度与显存 | CUDA、`bfloat16`、梯度检查点 | 降低显存压力，仍需按实际显存调整 batch |
| 默认微调范围 | `freeze_vision_encoder=true`、`train_expert_only=true` | 默认训练 action expert；`--full-finetune` 会把两项设为 `false` |
| 动作预测 | `chunk_size=50`、`n_action_steps=10`、`empty_cameras=0` | `--chunk-size` 可调整预测长度；执行步数最多为 10 |
| 训练规模 | 30,000 steps、batch 8、workers 4 | 可用 `--steps`、`--batch-size`、`--workers` 调整 |
| 记录与保存 | `save_freq=1000`、`log_freq=20`、seed 1000 | 每 1,000 步保存、每 20 步记录 |
| 在线服务 | `policy.push_to_hub=false`；W&B 使用 LeRobot 默认的关闭状态 | 默认不上传策略、不启用 W&B |

可用 `--dry-run` 打印将执行的命令；但它仍会做数据和权重的前置检查。`--full-finetune` 会提高显存需求，建议从较小 batch 开始。训练命令末尾还可追加未封装的 LeRobot 参数；若参数与脚本默认项重复，应检查最终打印的命令及 LeRobot 的解析结果。

## π₀.₅ 训练的关键注意事项

- **动作语义和填充维度：**当前数据的 12 维中只有前六维对应关节/夹爪，后六维为零填充。部署时须保持相同的维度顺序、单位和绝对动作语义；不要把零填充当作真实关节输出。
- **统计与归一化：**v2.1 数据必须具备 12 维 `q01/q99`。换数据后应重新生成与新数据匹配的统计，不能沿用当前数据集的数值。
- **相机与语言条件：**保持训练时两路相机键、视角和任务文本的含义一致；改变相机布局或任务描述时先核对数据特征与模型输入。脚本只检查至少存在一路视频，不会验证真实画面质量或任务文本是否正确。
- **转换的磁盘开销：**v2.1 原始备份、v3.0 文件和 schema 修复备份可能并存。确认转换、复检和训练正常后，再自行决定备份的保留策略。
- **显存与吞吐：**先完成两步短跑，再逐步调整 batch 和 worker。出现显存不足时优先降低 batch；启用 `--full-finetune` 前重新评估显存。
- **验证效果：**loss 下降仅说明训练目标在优化。部署前还需确认动作范围、执行频率和真实机器人任务成功率。

## 续训与常用检查

已有 LeRobot 训练配置时，可按 README 中的方式用 LeRobot 自带入口续训：

```bash
find outputs/pi05_vial -name train_config.json
lerobot-train --config_path=/实际路径/train_config.json --resume=true
```

`pipeline.py` 本身没有 `resume` 动作。自定义路径和参数可先查看帮助并做 dry run：

```bash
python -m scripts.pipeline --help
python -m scripts.pipeline train --dataset /data/my_dataset --weights /models/pi05_base --output /runs/exp1 --steps 1000 --batch-size 4 --workers 2 --dry-run
```

## 训练完成后如何检验效果

### 本地下载检查点后快速试跑

将训练后的 `model.safetensors`、`config.json` 和四个预/后处理器文件放在 `models/pi05/`。本地入口先检查文件、数据版本和依赖，默认评估 episode 7 的第 390–469 帧（8 次推理），便于先查看肩关节的大幅运动：

```powershell
python -m scripts.eval_local --check-only
python -m scripts.eval_local
# 显存不足时可改用 CPU，运行会慢得多：
python -m scripts.eval_local --device cpu
```

可用 `--episode`、`--start-frame`、`--max-frames`、`--horizon`、`--output` 调整评估范围。输出与 `eval_joints.py` 相同，CSV 的 `frame` 和图中的横轴保留 episode 原始帧号。需要 LeRobot v0.6.1、PyAV、Parquet 依赖、Matplotlib，以及已获许可并可访问的 `google/paligemma-3b-pt-224` tokenizer。RTX 5070 Ti Laptop GPU 的约 12 GB 显存不保证容纳模型与推理中间张量；CUDA 报显存不足时使用 `--device cpu`。

入口还会逐维比较检查点保存的 `q01/q99` 与本地数据 `meta/stats.json`。目前下载的检查点与本地原始 v2.1 数据的样本数、min/max、mean 基本一致，但 action 和 state 的分位数明显不同。例如 action 的 `shoulder_lift.pos` 的 `q99` 分别为约 -33.38 与 +26.07。这是需要排查的训练数据统计差异；不可只在评估时替换处理器统计值，因为模型是按保存的统计值训练的。LeRobot v0.6.1 的 v2.1→v3.0 转换器会根据每个 episode 的统计重新聚合全局统计，其聚合实现对分位数取按 episode 长度加权的平均，这不等于所有帧的全局分位数。详见[转换器](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/convert_dataset_v21_to_v30.py)和[统计聚合代码](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/compute_stats.py)。

当前本地数据仍是 v2.1。先按上文安装固定版本 LeRobot，再在数据副本上转换，避免更改原始文件：

```powershell
Copy-Item datasets/Lerobot_1028_12_dim datasets/Lerobot_1028_12_dim_eval -Recurse
python -m scripts.pipeline convert --dataset datasets/Lerobot_1028_12_dim_eval
python -m scripts.eval_local --dataset datasets/Lerobot_1028_12_dim_eval --check-only
python -m scripts.eval_local --dataset datasets/Lerobot_1028_12_dim_eval
```

### 1. 逐关节离线轨迹对比

先找到**训练后的策略目录**（其中有 `config.json`、模型权重以及预/后处理器配置），再运行评估脚本。不要把 `models/pi05_base` 当成训练结果。

脚本需要训练环境中的 LeRobot v0.6.1、NumPy 和 Matplotlib；如缺少绘图库，运行 `python -m pip install matplotlib`。

```bash
find outputs/pi05_vial -name config.json
python -m scripts.eval_joints \
  --checkpoint /实际路径/训练后策略目录 \
  --dataset datasets/Lerobot_1028_12_dim \
  --episode 7 \
  --output outputs/eval_joints
```

脚本针对一个 episode，在第 0、10、20……帧（或检查点配置的 `n_action_steps` 间隔）读取**数据集记录的真实图像、状态和任务文本**，每次预测一个动作块并取前 10 帧。最后一块按 episode 长度裁剪。它将预测动作经检查点后处理器恢复原始单位，与同一帧的 `action` 比较。默认只绘制非 `padding_*` 的六个关节/夹爪维度，并生成：

- `episode_007.png`：每个关节一幅子图，蓝线为真值、橙线为预测、红点为推理起点，与示例图的读法一致。
- `episode_007_metrics.csv`：每个维度的原始单位 MSE 和 MAE。
- `episode_007_trajectory.csv`：逐帧真值、预测值、推理起点标记，便于进一步分析。

可用 `--horizon` 改变每次取用的预测步数，数值不能超过 `chunk_size`；为了模拟当前部署设置，通常保持默认。`--seed` 固定推理噪声以便重复比较。图中的红点标记**重新推理时刻的真值位置**，不是机器人实际执行后的观测。

曲线分析时重点看关节跳变处的滞后、过冲、平台段抖动，以及夹爪开合是否及时。MSE 对大误差更敏感，MAE 更容易解释为平均偏差；不同关节量纲和活动范围可能不同，应逐维比较。示例图中个别维度明显振荡，即使总 MSE 看起来可接受也值得检查。

### 2. 解释离线指标的边界

此评估每次都从数据集读取**真实历史观测**，预测动作不会驱动下一帧状态，因此它是 teacher-forced 的离线动作拟合检查，不能证明闭环控制稳定。当前元数据把 75 个 episode 全部列在训练 split 中；若直接用这些 episode 评估，得到的是训练集拟合程度，不能当作泛化性能。要比较模型或调参，应按 episode 留出一组从未参与训练的验证/测试数据，并保持两路相机、任务描述和关节单位一致。对同一评估集，建议比较基座、不同 checkpoint 及简单的“保持上一动作”基线。

### 3. 真机/仿真闭环验证

在实际部署接口中，逐次输入当前相机、状态、任务文本，执行前 10 步预测动作后再次观测。先核对 12 维顺序、前六维的单位与限位、夹爪定义、30 FPS 与控制频率的映射，以及后六维零填充的处理；然后在安全约束下记录任务成功率、失败类型、动作延迟和异常振荡。当前数据目录包含 `labels/task_success.csv`，但训练脚本不使用它；评估成功率时可作为标签来源之一，仍需核对标签与 episode 的对应关系。
