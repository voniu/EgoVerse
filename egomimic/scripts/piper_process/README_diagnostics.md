# 完整验证集误差诊断（`diagnose_full.py`）

复用 `evaluate_full` 的预测做三项诊断，不重新推理、训练或移动机械臂。聚合方式与完整评估相同：先平均每个 anchor 的有效 horizon，再平均帧、等权 episode 和种子；尾部重复填充不计分。单位为 mm，像素指标除外。

## 1. 坐标分量与投影

- 3D 同时刻位置误差，及 X（相机右）、Y（相机下）、Z（相机深度）绝对误差和 Z 占三轴平方误差的比例。
- 同时刻像素距离；预测与目标均在画面内时的像素距离；预测/目标落在画面外的比例（等权 anchor / episode 统计）。

全量像素统计保留画面外点，而投影视频仅绘制画面内点，因此视频展示的是有选择的子集，局部重叠不足以证明同时刻预测准确。画面外比例本身也不证明标定有误。

## 2. 时间对应与轨迹连续性

- 忽略顺序，预测点到同一未来 chunk 最近目标点的距离（3D 与像素）。允许多个预测点匹配同一目标点，只作乐观诊断；若它明显小于同时刻误差，说明点的范围合理但执行序列不准确。
- 预测与示范的相邻 horizon 点平均位移，用于判断预测序列是否跳动（不是实际机器人速度测量）。
- 固定延迟扫描：在 -30 至 +30 帧比较 `pred[t,h]` 与 `command[t+h+delta]`，各延迟使用完全相同的内部 anchor（`30 <= t < N-59`，h=0..29），因此零延迟值与全量分数不同；正 delta 表示与更晚的目标比较。若最佳延迟改善很小，说明单一固定时间偏移解释不了主要误差。
- `time_order_example.png`：固定示例的逐坐标时间曲线，显示多次采样的跳动。

## 3. 保持当前状态基线

与模型使用相同目标、mask 和聚合方式的离线 persistence baseline：

- 重复当前实际 TCP 状态（`right.obs_ee_pose_camframe` 与实际夹爪开度）30 次。
- 重复当前记录命令（辅助参照）。它使用目标第 0 步，包含待预测的当前命令信息，不是可部署基线，仅用于说明近未来命令自身的变化规模。

这不是闭环执行成功率。实际状态与当前命令之间的差异可能来自跟随滞后或状态/命令转换差异，本诊断不判定原因。

## 运行与产物

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
  python -m egomimic.scripts.piper_process.diagnose_full \
  --evaluation <evaluation_dir> --dataset <piper_zarr_dir>
```

产物写入 `<evaluation_dir>/diagnostics/`：`report.json`、`per_episode_seed.csv`、`diagnostics.png`、`time_order_example.png`。
