# π₀.₅ Trained RTC 模型训练交接

本文交给模型训练端。目标是基于现有 SO-101 任务模型，训练能够以已承诺动作前缀为条件生成后续动作的 π₀.₅ checkpoint，供 Windows 推理、Linux 执行。方案为待验证的训练计划；当前 `trained_models/Pi05_1103` 尚不是 trained RTC 模型。Linux 和 Windows 的部署协议见 [PI05_LINUX_HANDOFF.md](PI05_LINUX_HANDOFF.md) 的「Trained RTC：Linux 开发交接」。

## 当前资产与输入契约

- 初始化候选：`trained_models/Pi05_1103`，是已有任务微调 checkpoint；保留原模型作对照，建立独立训练运行目录。推荐加载其权重继续微调，并新建优化器和调度器，而不是把修改过的配置冒充已训练模型。
- 现有模型：π₀.₅，`chunk_size=25`、`n_action_steps=10`、`num_inference_steps=10`、`use_relative_actions=false`，六维 state/action；checkpoint 内 `rtc_config=null`，未声明训练时 RTC 延迟。
- 两路 RGB feature：`observation.images.head`、`observation.images.wrist`，输入 feature 为 480×640，模型配置的图像分辨率为 224×224。沿用原 checkpoint 的处理器、图像方向和颜色顺序。
- 状态和动作是 LeRobot 校准后的六维 `.pos`；动作为**绝对关节目标**，不是原始 ticks、速度或相对偏移。按以下顺序排列：

```text
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
wrist_flex.pos, wrist_roll.pos, gripper.pos
```

- 当前仓库 `dataset/Lerobot_1028_12_dim` 元信息为 30 Hz、75 个 episode、41,952 帧、两个任务；state/action 各为六维有效值加六维零 padding。原训练配置引用训练机 `/opt/NeMo/TrainPi0.5/datasets/Lerobot_1028_6_dim`，路径只表示历史配置，不能假定该数据仍存在。
- 优先使用原六维数据；否则制作独立六维数据副本，正确更新 parquet、feature shape/names 和统计量，并验证前六维名称、数据与 checkpoint 统计一致。不要直接把十二维输出模型伪装为六维部署。任务文字须与对应数据集任务一致。

## 训练实现要求

本仓库 `lerobot/src/lerobot/policies/pi05/modeling_pi05.py` 已包含训练时 RTC 实现；训练机原先使用 `/opt/NeMo/TrainPi0.5`，**必须确认训练实际导入的 LeRobot 版本也包含这些改动**。记录源码 commit、实际导入路径、依赖版本和本地补丁。

每个训练样本的未来动作长度为 C，独立均匀采样前缀长度 d，范围为 0 到 D（含两端），其中 D 是 `policy.rtc_training_max_delay`。前 d 步作为干净动作输入，并使用干净时间条件；后缀进行正常 flow matching 加噪，仅对后缀及真实动作维度计算 loss。推理时对前缀硬约束，生成后续动作。确认逐动作时间条件在 π₀.₅ action expert 中正确生效。

重点核对本仓库以下实现和测试是否在训练机可运行：

- `configuration_pi05.py` 的 `rtc_training_max_delay`；`modeling_pi05.py` 的 `_sample_training_rtc_prefix_mask`、`_reduce_training_rtc_loss` 和 `_prepare_trained_rtc_prefix`。
- `lerobot/tests/policies/pi0_pi05/test_pi05_training_time_rtc.py` 以及对应 flow matching 的 hard-prefix 测试。记录实际测试命令、结果及跳过项。
- 保证训练窗口不跨 episode；检查 episode 尾部 padding 的采样和 loss 处理，避免填充目标污染后缀训练。

**仅在 config 中填写 D 不会产生 RTC 能力。** 必须实际以该训练目标更新权重，并对保存、加载后的模型做带前缀推理验证。

## 参数与首轮实验

| 项目 | 建议起点及条件 |
| --- | --- |
| 初始权重 | `Pi05_1103`，独立新训练运行 |
| 动作频率 | 30 Hz，沿用数据时间语义 |
| 预测步数 C | 候选 `chunk_size=40`，较原模型的 25 步有变化，必须训练并验证 |
| 最大前缀 D | 候选 `rtc_training_max_delay=15`，对应 500 ms |
| 部署前缀容量 H | 候选 `execution_horizon=15`，是部署参数 |
| 可训练部分 | 首轮沿用 `train_expert_only=true`，保留原冻结策略 |
| 初始学习率 | 试验范围 `2.5e-6`～`1e-5`；确保 optimizer 和 scheduler 实际使用一致值 |
| 首轮训练 | 约 2,000～5,000 次参数更新，建议每 500 次保存并评估 |
| 推理去噪步数 | 首轮沿用 10，另行评估延迟后再调整 |

这些均为实验起点，不是已经测得的最佳参数。实际 D 应根据真实观测采集、编码、网络、推理、结果处理和调度的端到端高分位延迟，加交接余量确定；此前约 292 ms 仅是虚拟输入的模型耗时。若延迟经常超过 500 ms，先与部署端共同调整方案。当前仓库的保守部署准入要求 `D <= H <= C-D`；上述候选满足，但不保证吞吐量足够。`n_action_steps` 不等同于 RTC 前缀长度或 H，部署完整块的行为需单独确认。

按完整 episode 划分训练/验证，两个任务均要覆盖，避免相邻帧泄漏。优先保持与原模型相同的数据变换和状态/动作归一化；若改变数据统计、采样或预处理，须记录并验证对初始化权重和部署处理器的影响。训练前检查实际 batch 中观测、动作、feature shape、任务文字和量纲，确认六维一致。

## 建议执行顺序

1. 确认训练机 GPU/显存、代码版本、六维数据和当前 checkpoint 可用；记录数据版本与划分、随机种子和训练配置。
2. 跑相关单元测试及一个小批次前向/反向，检查 d=0、短前缀和 d=D，确认 loss 有限、梯度和冻结参数符合预期。
3. 做少量训练更新，保存完整 checkpoint，并在新的进程从产物重新加载；核对 C、D、六维 feature、处理器及权重，不依赖训练进程缓存。
4. 再运行首轮实验，定期评估多个 checkpoint，记录训练/验证损失、各关节误差及计算开销；不以 loss 单独决定最终模型。
5. 将候选交给 Windows/Linux 联调，在相同任务和条件下对比现有普通模型、guided RTC 对照（可用时）和 trained RTC，记录成功率、完成时间、动作边界变化及延迟。

## 交付前验证与交付物

- 使用完整部署预处理/后处理，在 `d=0`、中间值和 `d=D` 做实际 `predict_action_chunk()`；确认输出形状 `[1,C,6]`、数值有限，前缀硬约束在模型空间成立，反归一化后也在明确容差内一致。
- 测试 d>D、前缀不足、NaN/Inf 和错误维度被拒绝；验证 action normalizer 与 unnormalizer 的有效范围往返，不把 state normalizer 用作动作转换。
- 测量预热后首块和带前缀的推理耗时分布、峰值显存、硬件和 batch size；这些模型侧数字不代替双机端到端测量。
- 交付完整 LeRobot `pretrained_model`：`config.json`、`model.safetensors`、`policy_preprocessor.json`、`policy_postprocessor.json` 和所有处理器引用的统计 `.safetensors`；确认是实际训练后的权重，不能只交 config 或权重文件。
- 附实验记录：训练代码 commit/补丁、依赖和 GPU、初始化 checkpoint 标识、数据版本/划分、完整训练配置与真实启动命令、C/D、步数、保存点、训练与验证结果、测试输出、模型文件摘要和已知限制。提供动作频率、关节语义、相机名和任务文字。
- 发布前由 Windows 确认加载器能加载整个产物并启用 trained RTC，由 Linux 确认 C/D/H 与部署协议一致；真实双机运行、断流处理及实机任务表现需单独验收。
