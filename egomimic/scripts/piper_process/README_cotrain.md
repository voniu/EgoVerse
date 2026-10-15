# Piper A/B：机器人与左手示范联合训练

B 配置继承 A 的网络与训练参数，仅增加人类域。这里的“双域”是两类
具身数据共用视觉编码器与 HPT 主干，不是分别训练两个独立策略。

| 模块 | Piper | Aria 左手 |
|---|---|---|
| 主视角 | D435IF RGB | 校正后的 Aria RGB |
| 主视角编码器与视觉 stem | 两域共享 ResNet18、投影及 16 个视觉 token | 同左 |
| 腕部视角 | 独立编码器与 16 个 token | 无此模态，不补假图像 |
| 状态 stem | 7 维 → 16 个 token | 6 维 → 16 个 token |
| HPT 主干 | 两域共享 16 层、256 维、8 头 | 同左 |
| Flow Matching 动作头 | 独立，输出 `[30,7]` | 独立，输出 `[30,6]` |

6 维为当前相机系 xyz + ZYX 欧拉角，机器人第 7 维为夹爪开度。
人类采用已人工确认的工具轴约定；保留掌心位置，不虚构人类夹爪开度。
动作块所有未来位姿都表达在当前相机坐标系中。

## 数据和对比约定

- Piper 使用与 A 完全相同的训练/验证 episode 划分。
- Aria 按 episode 划分训练/验证，不按帧随机拆分。人类片段的裁剪需人工确认包含完整任务过程。
- 正式配置每步一个机器人 batch + 一个同大小的人类 batch，较短的 loader 循环采样。
  与 A 保持相同机器人样本数/步和训练步数预算，B 的总数据量及计算量更大。
- 目标为 `(robot_loss + human_loss) / 2`，沿用当前 HPT 的两域平均损失。
- 单独记录两个域的验证损失，B 按 `Valid/piper_right_arm_loss` 保存最佳模型。
  A 的单域 `Valid/action_loss` 就是机器人损失。B 的平均损失不能直接与 A 比较。
- 归一化各域独立、仅使用各自训练集全部帧。机器人统计应与 A 相同。
- 正式 B 从头初始化，使用相同随机种子与 ImageNet 视觉初始化；不从 A 或 smoke
  checkpoint 微调。相同种子不保证新增分支后的所有公共参数逐元素初始化相同。

## 配置与运行

入口为 `train_piper_human.yaml`，小规模检查为 `train_piper_human_smoke.yaml`。
模型配置为 `model/hpt_piper_human_flow.yaml`，数据配置为
`data/piper_human_cotrain.yaml`。A 的入口和单域模型配置保持不变。

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m egomimic.trainHydra \
  --config-name train_piper_human_smoke \
  norm_stats.num_workers=4 \
  hydra.run.dir=<smoke_run_dir>
```

smoke 使用每域 batch=2、10 次参数更新、1 个联合验证 batch。
归一化仍使用完整训练集，不缩减统计样本。它验证读取、归一化、双域前向、
反向、动作采样、验证及 checkpoint 保存，不用于判断 B 是否比 A 更好。

正式训练可复用此运行目录中的 `norm_stats/norm_stats.json`，但不传 `ckpt_path`：

```bash
python -m egomimic.trainHydra --config-name train_piper_human \
  norm_stats.precomputed_norm_path=<smoke_run_dir>/norm_stats/norm_stats.json
```

若变更数据、划分或动作表示，应重新计算统计。在集群上运行时需覆盖机器人和人类数据路径。
