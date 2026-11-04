# π₀.₅ 训练时 RTC

使用现有 12 维数据（后六维为零填充）和 `lerobot/pi05_base` 基座权重，冻结视觉语言主干，微调 action expert。训练窗口末尾沿用 LeRobot 的最后动作填充。训练后模型输出仍为 12 维，部署时应按实际接口处理后六维。

在 Ubuntu 服务器安装含训练时 RTC 的 LeRobot：

```bash
git clone https://github.com/huggingface/lerobot.git lerobot
git -C lerobot checkout 6e3b67c5f0b77c93a5179d88d7c1d6a5f7ff7b27
python -m pip install -e './lerobot[training,pi]'
python -m scripts.pipeline download
python -m scripts.pipeline inspect
python -m scripts.pipeline convert
python -m scripts.pipeline inspect
```

如果服务器已存在 lerobot 目录，先检查其 commit 和本地修改，再决定如何切换版本。数据转换及统计修复细节参见训练指南。训练时会检查当前 Python 导入的 RTC 实现并输出路径。

```bash
# 先检查真实的视频解码、权重加载和 CUDA 前后向
python -m scripts.pipeline train --chunk-size 40 --rtc-max-delay 15 --steps 2 --batch-size 1 --workers 0 --output outputs/rtc_smoke

# 从基座权重重新训练，指定新的输出目录
python -m scripts.pipeline train --chunk-size 40 --rtc-max-delay 15 --steps 30000 --batch-size 8 --workers 4 --save-freq 500 --output outputs/pi05_rtc
```

`--rtc-max-delay` 表示训练时随机采样的最大干净动作前缀长度，必须小于 chunk size；设为 0 则沿用原有普通训练。15 步在 30 Hz 下约为 500 ms，是待部署延迟验证的候选值。执行长度、部署前缀衔接及端到端延迟需要部署端单独配置和验证。应按 episode 留出验证集并验证保存后的检查点及其处理器；训练 loss 不能证明闭环成功。
