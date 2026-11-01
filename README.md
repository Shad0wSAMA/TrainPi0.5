# 单卡 H200 微调 π₀.₅

这个仓库提供本地数据检查、LeRobot v2.1 → v3.0 转换、基座权重下载和训练启动脚本。实际模型、数据加载与优化器由固定版本的 [LeRobot v0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1) 提供，避免复制并维护一套会过时的训练实现。

训练步骤、默认参数及 π₀.₅ 训练注意事项见[训练流程指南](docs/training_guide.md)。

## 数据格式

脚本默认读取 `datasets/Lerobot_1028_12_dim/`。当前样本元数据：75 个 episode、41,952 帧、30 FPS、两个任务；`observation.images.head` 和 `observation.images.wrist` 两路 480×640 AV1 视频；`observation.state` 与 `action` 均为 12 维 float32。末尾 6 维名为 `padding_6` 至 `padding_11`，统计值全为零。数据是**绝对关节位置**，因此默认不启用相对动作。`meta/stats.json` 已包含量化归一化需要的 `q01/q99`。标签文件 `labels/task_success.csv` 不参与 LeRobot 的行为克隆训练。

本仓库忽略 `datasets/`、`lerobot/`、`models/`、`outputs/` 等目录。解压数据和下载模型后，`git status` 不会把这些大文件列为待提交内容。

## Ubuntu 24.04 服务器：从克隆到训练

以下命令在 Bash 中运行。服务器需要一张可由 `nvidia-smi` 识别的 H200、足够的磁盘空间，以及能访问 GitHub 和 Hugging Face 的网络。模型权重、转换中间目录和原始备份会同时占空间；先用 `df -h` 检查剩余空间。

### 1. 克隆仓库与放置数据

```bash
git clone <你的仓库 URL> TrainPi0.5
cd TrainPi0.5
mkdir -p datasets
# 将压缩包解压后，确保下面的路径存在：
test -f datasets/Lerobot_1028_12_dim/meta/info.json
test -d datasets/Lerobot_1028_12_dim/data
test -d datasets/Lerobot_1028_12_dim/videos
```

如果压缩包多包了一层目录，移动**整个** `Lerobot_1028_12_dim` 文件夹到 `datasets/` 下，不要拆散 `meta/`、`data/` 和 `videos/`。可先运行 `python -m scripts.pipeline inspect`；它只使用 Python 标准库，能查缺失的 episode 文件。

### 2. 安装环境

推荐 Miniforge 的 Python 3.12 环境；系统已有兼容 Conda 时可跳过安装 Miniforge。LeRobot 当前[安装说明](https://huggingface.co/docs/lerobot/v0.6.1/installation)采用 Python 3.12 和环境内 FFmpeg。Ubuntu 24.04 自带的系统 Python 3.12 也可用于虚拟环境，但请确保 PyTorch/CUDA 与 FFmpeg 兼容。

```bash
sudo apt update
sudo apt install -y git git-lfs curl build-essential pkg-config
git lfs install
# 若尚未安装 Conda/Miniforge：
curl -L -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -y -n pi05 python=3.12
conda activate pi05
conda install -y -c conda-forge ffmpeg

git clone --branch v0.6.1 --depth 1 https://github.com/huggingface/lerobot.git lerobot
python -m pip install --upgrade pip
python -m pip install -e './lerobot[training,pi]'
python -c 'import torch, lerobot; print("torch", torch.__version__, "CUDA", torch.version.cuda); print("GPU", torch.cuda.get_device_name(0))'
ffmpeg -version
```

如果最后一行 Python 检测提示 CUDA 不可用，先检查 `nvidia-smi` 和 PyTorch 安装；不要启动训练。H200 支持 bfloat16。若 LeRobot 安装过程中 PyTorch wheel 与驱动不匹配，参照上述官方安装页的 CUDA wheel 段落选合适版本。

### 3. 登录 Hugging Face 并下载权重

先在浏览器打开 [PaliGemma tokenizer 页面](https://huggingface.co/google/paligemma-3b-pt-224)，接受许可；使用拥有该许可的账号登录。π₀.₅ 训练时会从这个 gated repo 读取 tokenizer，仅下载 `lerobot/pi05_base` 权重还不够。[官方 π₀.₅ 指南](https://huggingface.co/docs/lerobot/v0.6.1/pi05)也列出此要求。

```bash
hf auth login
python -m scripts.pipeline download
test -f models/pi05_base/config.json
```

权重保存在仓库内被忽略的 `models/pi05_base/`。如服务器只允许预先下载模型文件，还须缓存上述 gated tokenizer；登录后正常训练会自动下载。

### 4. 检查并转换数据

```bash
python -m scripts.pipeline inspect
python -m scripts.pipeline convert
python -m scripts.pipeline inspect
```

LeRobot v0.6.1 的转换器会将 `datasets/Lerobot_1028_12_dim/` 原位置替换为 v3.0，并保留 `datasets/Lerobot_1028_12_dim_old/` 原始 v2.1 备份。脚本强制 `--push-to-hub=false`，不会上传数据。转换后，脚本还会删除 Parquet 中不在 `meta/info.json` 声明的旧字段 `next_reward`、`next_done`；每个修改过的 Parquet 文件都会留下 `.before_schema_fix` 备份。**转换可能需要较长时间和额外磁盘空间**，尤其因为视频要拼接。转换中断时先检查 `_old`、`_v30` 目录及 `meta/info.json`，不要盲目删除备份。转换成功后再次运行会跳过。

如果此前已运行旧版脚本，数据已是 v3.0，但训练报 `CastError: column names don't match`，请在更新本仓库后运行：

```bash
python -m scripts.pipeline repair
python -m scripts.pipeline inspect
```

`repair` 只处理 `next_reward`、`next_done` 这两个多余字段，不会再次转换视频或上传数据。若出现其他未声明字段，它会停止并报告列名。

### 5. 启动训练

建议先跑 2 步验证完整链路，再正式训练。预检只检查文件和元数据；真正的视频解码、tokenizer 和 CUDA 路径由这次短跑验证。

```bash
python -m scripts.pipeline train --steps 2 --batch-size 1 --workers 0 --output outputs/smoke
python -m scripts.pipeline train --steps 30000 --batch-size 8 --workers 4 --output outputs/pi05_vial
```

默认设置：加载 `lerobot/pi05_base` 的**权重**并按本地数据构造输入特征；两路相机、12 维状态/动作、50 步 action chunk、执行 10 步；bfloat16、梯度检查点、冻结视觉语言模型并训练 action expert。默认单卡 batch size 8，偏向稳定启动；若显存充裕可提高到 16/32，并用 `nvidia-smi` 观察峰值。训练步数、batch size 和 worker 数均可通过脚本参数调整。更完整但显存需求更高的微调可添加 `--full-finetune`；建议先降低 batch size。其他未封装的 LeRobot 参数可附加在命令末尾。

日志默认每 20 步输出一次，检查点默认每 1000 步保存一次。可用 `--log-freq` 和 `--save-freq` 调整，例如：

```bash
python -m scripts.pipeline train --steps 30000 --batch-size 8 --log-freq 10 --save-freq 500 --output outputs/pi05_vial
```

`--save-freq 0` 关闭周期保存，训练结束时仍保存最终检查点。检查点可能很大，应按磁盘空间选择保存间隔。

此数据集没有相机的图像统计项，且 π₀.₅ 的 `VISUAL` 归一化为 `IDENTITY`。脚本因此设置 `--dataset.use_imagenet_stats=false`，避免 LeRobot 对不存在的图像统计项执行写入。已经转换和修复的数据不需要再次处理。

建议用 `tmux` 保持 SSH 断线后训练继续：

```bash
tmux new -s pi05
conda activate pi05
cd /path/to/TrainPi0.5
python -m scripts.pipeline train --steps 30000 --batch-size 8 --workers 4 --output outputs/pi05_vial 2>&1 | tee outputs_pi05_vial.log
# Ctrl+B 然后 D 可退出 tmux，之后用 tmux attach -t pi05 返回。
```

LeRobot 会在指定 `outputs/` 下保存训练配置、日志和检查点。续训时使用 LeRobot 自带接口，指向保存的 `train_config.json`，例如：

```bash
find outputs/pi05_vial -name train_config.json
lerobot-train --config_path=/实际路径/train_config.json --resume=true
```

训练结果仍需在真实机器人上验证；离线训练 loss 不能单独说明抓取成功率。

## 自定义路径与检查

```bash
python -m scripts.pipeline --help
python -m scripts.pipeline train --dataset /data/my_dataset --weights /models/pi05_base --output /runs/exp1 --batch-size 4 --steps 1000 --workers 2 --dry-run
python -m unittest discover -s tests -v
git status --short --ignored
```

脚本针对当前数据结构的 12 维状态/动作编写。若以后换成不同维数或相机布局，需要先调整预检要求、重新确认统计值和机器人动作语义。
