# Piper A/B 完整离线评估

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg \
python -m egomimic.scripts.piper_process.evaluate_full \
  --a .local_calibration/hpc/results/checkpoints/a/best.ckpt \
  --b .local_calibration/hpc/results/checkpoints/b/best.ckpt \
  --dataset .local_calibration/piper_all \
  --split .local_calibration/aria_gen2_aligned/cotrain_data_check.json \
  --output .local_calibration/full_validation --previews
```

使用 A step=9100、B step=9800 的最佳模型，仅评估机器人分支。
从保存的划分清单读取 7 条验证 episode，并与 seed=42 的原始划分再核对。
两模型各遍历全部 2,381 帧，各使用推理种子 0、1、2，batch=32，FP32、
50 次 flow 积分步，无训练图像增强。3 个种子的变化不是独立训练方差或置信区间。

## 避免评估样本被替换

直接使用 `ZarrDataset(strict=True)`，不用会检查归一化边界并重采样的 MultiDataset。
仅显式应用 checkpoint 的训练统计，允许归一化后超出 [-1,1]，不裁剪、不丢帧。
JPEG 读取失败直接停止。每帧只读取一次，三个种子复用该帧同一份观测特征。
核对两组机器人归一化完全一致，反归一化目标可还原原始单位。

## 指标

- 位置：预测与示范 TCP 位置的欧氏距离，mm。
- 朝向：`R_target.inv() * R_pred` 的 SO(3) 旋转角，度；不相减欧拉角。
- 夹爪：开度绝对误差，mm。
- 辅助 FM loss：预测速度与 flow target 的逐时刻 MSE，在 7 个维度上取均值。
  两组使用配对 Gaussian 噪声和 Beta 时间采样，不能与训练时未屏蔽尾部且会重采样的
  验证 loss 直接当作同一指标比较。

对每个物理指标同时记录第一动作和完整 30 帧块。对于起点 t，仅统计
`k < min(30, episode_length - t)` 的真实未来帧。填充仍提供给网络所需固定形状，
但不计入指标。FM loss 也只统计有效位置；它的 head 输入仍含训练中使用的填充目标。

汇总顺序：先对每个起点的有效 future steps 求均值，再对该 episode 的所有起点求均值，
最后对 7 个 episode 等权平均。对三个种子分别完成上述汇总，再报均值及种子标准差。
长示范不会在最终结果中拥有更大权重。

相同 seed 配合固定 episode 顺序、batch=32 和帧起点产生配对初始噪声。
改变 batch size 后不保证逐帧随机数相同，故 A/B 必须用同一批大小。

## 输出

- `report.json`：checkpoint SHA256、实际步数、完整协议与汇总指标。
- `comparison.png`：完整动作块的三个物理误差对照，误差线是推理种子标准差。
- `per_episode_seed.csv`：每个模型 × 7 episode × 3 seed，共 42 行。
- `A_*.npz` / `B_*.npz`：原单位预测、目标、有效 mask、帧编号、逐位置 FM loss。
- 可选投影视频：按名称排序前 3 条 episode，左 A、右 B，绿色示范轨迹，紫色预测轨迹，
  使用 seed 0。没有按结果好坏挑选示范。

这是使用示范观测的离线预测评估，不是闭环执行，也不直接衡量任务成功率。
实际操控前还需要检查时序、可达性、控制转换，并进行同条件真机试验。

## 2026-09-28 完整结果

两组各完成 7 条、2,381 帧、3 个推理种子的评估。每组每个种子统计 68,385 个
有效未来动作位置；目标、mask、帧索引和 seed 清单逐项完全一致。
以下是 episode 等权、再对 3 个 seed 平均的结果，均为越低越好：

| 指标 | A | B |
|---|---:|---:|
| 第一个动作位置误差 mm | 104.727 | 107.622 |
| 完整动作块位置误差 mm | 97.120 | 101.002 |
| 第一个动作朝向误差 ° | 18.171 | 20.342 |
| 完整动作块朝向误差 ° | 18.440 | 20.567 |
| 第一个动作夹爪误差 mm | 16.509 | 14.408 |
| 完整动作块夹爪误差 mm | 15.352 | 14.315 |
| masked FM loss | 0.235510 | 0.230538 |

B 的夹爪误差和辅助 FM loss 略低，但位置和朝向误差更高；7 条 episode 的
平均动作块位置误差均是 B 高于 A。不能据此支持“加入 Ego 后整体更好”的结论，
也不能把推理 seed 的标准差当作模型泛化提升的显著性检验。

平均位置误差约 10 cm，不建议据此直接执行真机抓取。
需要下一步诊断误差分布、时序与动作表示/推理质量，再决定训练调整和上线验证。
对照视频中同像素附近也可能有不同深度，不能仅凭 2D 投影重合判断 3D 误差小。

已完成独立检查：

- 已知 10 mm 平移、90° 旋转、2 mm 开度误差，及 ±π 朝向等价性。
- 大误差填充帧不影响 masked 指标。
- 模拟坏 JPEG 时 strict reader 直接报错，不替换原帧。
- 使用同一真实 batch、噪声，原 HPT `policy.forward` 与评估器的
  `forward_features + sample_action` 输出完全一致（最大差 0）。
- 3 个投影视频帧数与对应 episode 长度一致。

原始数值和逐条结果位于 `.local_calibration/full_validation/`；本次没有控制机械臂。
