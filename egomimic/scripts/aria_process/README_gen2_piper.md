# Aria Gen2 → Piper B 组：左手示范转换

把 Project Aria Gen2 的 VRS 录像（VRS 内置手部跟踪和 VIO，不依赖云端 MPS）转换为 EgoVerse Zarr，
作为 Piper B 组的人类左手示范。人手末端朝向与 Piper TCP 的对应见
[docs/calibration/aria_left_tcp_axes.md](../../../docs/calibration/aria_left_tcp_axes.md)，
固定旋转保存于 `egomimic/robot/piper/aria_left_alignment.json`。

## 执行

从仓库根目录运行：

```bash
source emimic/bin/activate
# 可选：统计各录像中满足筛选条件的帧数
python -m egomimic.scripts.aria_process.inspect_gen2 <vrs_dir> \
  --output <audit.json>

# 转换（输入可为单个 .vrs 或目录）
python -m egomimic.scripts.aria_process.gen2_to_zarr <vrs_dir> \
  --output <aria_zarr_dir> \
  --task-name "<task description>" \
  --vio-source high_frequency \
  --alignment egomimic/robot/piper/aria_left_alignment.json

# 检查
python -m egomimic.scripts.aria_process.check_gen2 <aria_zarr_dir> \
  --alignment egomimic/robot/piper/aria_left_alignment.json
```

- `--vio-source`：`vio` 为设备端 10 Hz 位姿流，`high_frequency` 为同一里程计坐标系下的 800 Hz `vio_high_frequency` 流。
- `--alignment`：应用固定的 `R_wrist_tool`，保留掌心位置及原始手腕/关键点；省略时保留 SDK 原生手腕轴（仅用于检查数据链路）。
- 转换器拒绝覆盖已有 Zarr；再次转换请指定另一个输出目录。

`manifest.json` 随每条转换更新，只有全部成功才标记 `complete: true`。应用对齐的数据标记
`pose_axes: piper_like_left_tool` 并附完整固定矩阵；`source_wrist_pose` 保持原生手腕轴，便于独立检查。
朝向约定确认不等于每条裁剪片段都已人工检查，因此保留 `requires_clip_review` 标记。

`check_gen2.py` 核对每个 episode 的掌心位置、固定旋转仅应用一次、连续时间戳、当前相机系动作块和末尾填充，并读取真实图像批次。

## 帧筛选与裁剪

候选帧需满足左手置信度 ≥ 0.5、最近手部跟踪时间差 ≤ 25 ms，且最近 VIO 位姿有效。
转换还要求插值两端 VIO 均有效、属于同一会话、间隔 ≤ 150 ms，并只保留最长连续有效片段；
录像开头 VIO 初始化阶段不能作为有效世界坐标。因此应人工检查保留片段是否包含完整的
抓取、移动和放置过程（检查视频保留全部原始帧，绿色 KEPT / 红色 EXCLUDED 标识是否进入 Zarr，
黄色圆圈为掌心投影，红、绿、蓝轴为手腕坐标系的 X、Y、Z）。

示范使用左手时不镜像图像；Piper 数据中的 `right_arm` 是机器人的命名，与人的惯用手无关。
VRS 内置轨迹不是云端闭环轨迹，是否存在影响训练的漂移需单独做质量检查。

## 与 Piper 联合训练

`data/piper_human_cotrain.yaml` 继承 A 的机器人配置并增加人类域；B 的机器人划分须与 A 完全相同，
两个域各自的训练/验证 episode 不重叠。每步各取一份等大小的人类、机器人 batch，较短的人类 loader 循环采样。
该数据配置不能直接配合 A 的单域模型使用，入口为 `train_piper_human.yaml` 与 `train_piper_human_smoke.yaml`，
详见 `egomimic/scripts/piper_process/README_cotrain.md`。

## 数据规格与坐标系

RGB 按设备标定去畸变为 640×480 针孔图像，Zarr 保存 JPEG 与对应内参。
使用 DEVICE_TIME 对齐 RGB、手跟踪和 VIO；采样率约 30 Hz。
手部位置在实际手跟踪时间变换至世界系，相机位姿在 RGB 时间插值。

| Zarr 键 | 单帧规格 | 含义 |
|---|---|---|
| `left.obs_ee_pose` | 7 | 世界系掌心 xyz + 左手工具朝向四元数 wxyz（未对齐时为原生手腕轴） |
| `head_pose` | 7 | 世界系中的校正 RGB 光学相机位姿 |
| `left.source_wrist_pose` | 7 | 原始手腕位姿 |
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

SDK 数据定义参考：
- https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/dataset/pilot/tutorials/vrs_loading
- https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/hand_tracking/hand_tracking_outputs
