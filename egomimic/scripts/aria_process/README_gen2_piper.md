# Aria Gen2 → Piper B 组：左手试转换

更新：用户已确认 Piper 对应朝向，固定矩阵保存于
`egomimic/robot/piper/aria_left_alignment.json`。以下“原生手腕轴”描述指旧试转换。
批量转换应用已确认的固定旋转，保留掌心位置及原始手腕/关键点：

```bash
source emimic/bin/activate
python -m egomimic.scripts.aria_process.gen2_to_zarr \
  /mnt/wd10/zekun/data/Aria_Gen2/raw \
  --output .local_calibration/aria_gen2_aligned \
  --alignment egomimic/robot/piper/aria_left_alignment.json
```

`manifest.json` 随每条转换更新，只有全部成功才标记 `complete: true`。
新数据标记 `pose_axes: piper_like_left_tool`，附完整固定矩阵；
原始 `source_wrist_pose` 保持原生手腕轴，便于独立检查。
朝向确认不等于每条裁剪片段都已人工检查，因此保留 `requires_clip_review` 标记。

批量结果：50 条、8,383 帧；`check_gen2.py` 已核对全部 episode 的掌心位置、
固定旋转仅应用一次、连续时间戳、当前相机系动作块和末尾填充，并验证真实图像批次。

```bash
python -m egomimic.scripts.aria_process.check_gen2 \
  .local_calibration/aria_gen2_aligned \
  --alignment egomimic/robot/piper/aria_left_alignment.json
```

`data/piper_human_cotrain.yaml` 继承 A 的机器人配置并增加人类域，真实联合批次已通过：

| 数据 | 训练条数 / 帧数 | 验证条数 / 帧数 |
|---|---|---|
| Piper | 63 / 22,406 | 7 / 2,381 |
| Aria 左手 | 45 / 7,546 | 5 / 837 |

显式核对了 B 的机器人划分与 A 完全相同，两个域各自的训练/验证 episode 不重叠。
报告为 `.local_calibration/aria_gen2_aligned/cotrain_data_check.json`。
每步各取一份等大小的人类、机器人 batch，较短的人类 loader 循环采样。
数据配置不能直接配合 A 的单域模型使用。现已提供 `train_piper_human.yaml`
和 `train_piper_human_smoke.yaml`，双域统计和 10 步混合训练检查已通过，
详见 `egomimic/scripts/piper_process/README_cotrain.md`。

裁剪审查优先项：第 7 条原始 130 帧仅保留 `[38,127)` 共 89 帧，
用户已确认起始只是接触、尚未完成抓握，且末尾松手完整保留，因此第 7 条保留使用。

用户确认 50 条 `Put bear into basket` 示范全部使用左手。不镜像图像；
Piper 数据的 `right_arm` 是机器人的命名，与人的惯用手无关。

## 当前状态

50 个原始 VRS 共 10,231 张 RGB 图像。初筛中 8,508 帧满足左手置信度
≥ 0.5、最近手部跟踪时间差 ≤ 25 ms，以及最近 VIO 位姿有效的条件。
这不是最终训练帧数：转换还要求插值两端 VIO 均有效、属于同一会话，
间隔 ≤ 150 ms，并只保留最长连续有效片段。

试转换第 17 条：238 帧中保留原始索引 `[34, 235)`，共 201 帧。
开头 VIO 初始化阶段不能作为有效世界坐标。视频保留全部原始帧，
绿色 KEPT / 红色 EXCLUDED 标识是否进入 Zarr，供检查裁剪是否破坏完整任务。
黄色圆圈为掌心投影，红、绿、蓝轴为原生手腕坐标系的 X、Y、Z。

**这是数据管线试转换，尚不能直接启动正式 B 训练。** 需要确认：

- 有效连续片段保留完整抓取、移动、放置过程。
- 人类左手末端朝向与 Piper TCP 的语义转换；当前没有施加这一转换。
- B 组配置、模态缺失处理、采样比例与独立人类动作头。

本次使用 VRS 内置手部跟踪和 VIO，不依赖云端 MPS，也未上传录像。
其轨迹不是云端闭环轨迹；是否有足以影响训练的漂移仍需质量检查。

## 执行

从仓库根目录运行：

```bash
source emimic/bin/activate
python -m egomimic.scripts.aria_process.inspect_gen2 \
  /mnt/wd10/zekun/data/Aria_Gen2/raw \
  --output .local_calibration/aria_gen2_pilot/audit_50.json

python -m egomimic.scripts.aria_process.gen2_to_zarr \
  /mnt/wd10/zekun/data/Aria_Gen2/raw/Put_bear_into_basket_17_20260919_163143.vrs \
  --output .local_calibration/aria_gen2_pilot
```

转换器拒绝覆盖已有 Zarr。再次试转换请指定另一个输出目录。

## 数据规格与坐标系

RGB 按设备标定去畸变为 640×480 针孔图像，Zarr 保存 JPEG 与对应内参。
使用 DEVICE_TIME 对齐 RGB、手跟踪和 VIO；采样率约 30 Hz。
手部位置在实际手跟踪时间变换至世界系，相机位姿在 RGB 时间插值。

| Zarr 键 | 单帧规格 | 含义 |
|---|---|---|
| `left.obs_ee_pose` | 7 | 世界系掌心 xyz + 原生左手腕四元数 wxyz |
| `head_pose` | 7 | 世界系中的校正 RGB 光学相机位姿 |
| `left.source_wrist_pose` | 7 | 原始手腕位姿，供后续对齐使用 |
| `left.source_keypoints` | 63 | 世界系 21 个手部点 xyz |
| `left.confidence` | 1 | 左手跟踪置信度 |
| `images.front_1` | JPEG | 校正后的 RGB 图像 |
| `source.rgb_index` | 1 | 原始 RGB 帧索引 |
| `source.device_timestamp_ns` | 1 | 原始 RGB 设备时间戳 |

位置单位米。`AriaGen2Left` 读取器输出图像 `[3,480,640]`、状态 `[6]`、
动作 `[30,6]`，状态与动作均为 xyz + ZYX 欧拉角（弧度），没有虚构夹爪开度。
动作来自未来观测位姿，包含当前帧；末尾不足 30 帧时重复最后位姿。

设 `T_W_C(t)` 是当前相机位姿，`T_W_H(t+k)` 是未来掌心位姿，则整个动作块使用：

```text
T_C(t)_H(t+k) = inverse(T_W_C(t)) @ T_W_H(t+k), k = 0..29
```

不能为每个未来动作使用该未来时刻的相机坐标系，否则头部运动会改变动作语义。
现有掌心朝向保留 SDK 原生手腕轴，只供检查数据链路。

## 已完成检查

实际 Zarr 的首帧、中段、部分填充与末帧共 4 个样本通过 DataLoader：
图像 `[4,3,480,640]`、状态 `[4,6]`、动作 `[4,30,6]`。
通过独立旋转矩阵计算核对当前相机系未来动作、欧拉角对应旋转、
末尾填充以及无图像归一化读取的一致性。
结果见 `.local_calibration/aria_gen2_pilot/batch_check.json`。

SDK 数据定义参考：
- https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/dataset/pilot/tutorials/vrs_loading
- https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/hand_tracking/hand_tracking_outputs
