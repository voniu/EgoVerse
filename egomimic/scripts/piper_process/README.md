# Piper LeRobot v3 → Zarr（单条示范）

## Piper 训练样本适配

`egomimic.rldb.embodiment.piper.Piper` 使用现有 `ZarrDataset`，直接读取已保存的相机系位姿，不再次应用外参。

| 输出键 | 单样本形状 | 含义 |
| --- | --- | --- |
| `observations.images.front_img_1` | `[3,480,640]` | 当前 D435IF RGB，float32，范围 0–1 |
| `observations.images.right_wrist_img` | `[3,480,640]` | 当前 D405 RGB，float32，范围 0–1 |
| `observations.state.ee_pose` | `[7]` | 当前 `[x,y,z,yaw,pitch,roll,gripper]` |
| `actions_cartesian` | `[H,7]` | 从当前帧起的目标动作序列 |

位置和夹爪行程使用米，欧拉角使用弧度和内禀 `ZYX` 约定。存储的四元数 `wxyz` 通过工程现有变换转换为 yaw/pitch/roll。H 默认 30，可通过 `Piper.get_keymap(action_horizon=...)` 配置；30 Hz 下索引为 `t` 至 `t+29`，不插值、不额外时间偏移。序列到达 episode 末尾后重复最后一条指令，与现有加载器一致；无跨 episode 拼接。

夹爪行程在 Zarr 中保留物理单位，由训练入口的 MultiDataset 统计并执行归一化。`norm_mode=True` 去掉图像读取，仅保留用于统计的状态和动作。人类标准轴对齐尚未应用。

真实 batch 检查（开头、中段、末尾不足 H 步和最后一帧）：

```bash
source emimic/bin/activate
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper_mpl python -m egomimic.scripts.piper_process.check_batch \
  <piper_zarr_dir>/<episode>.zarr \
  --report <piper_zarr_dir>/batch_check.json
```

该检查核对 RGB 与当前帧一致、位姿单位和旋转矩阵、指令起始时间、末尾填充及归一化统计键映射，不代表模型已训练或几何投影误差已消除。

## 单条转换命令

在工程根目录执行，CPU 离线转换，不连接机械臂：

```bash
source emimic/bin/activate
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper_mpl python -m egomimic.scripts.piper_process.lerobot_to_zarr \
  --dataset <lerobot_dataset> \
  --episode 0 \
  --output <piper_zarr_dir> \
  --calibration <setup.yaml> \
  --action-calibration <follower_calibration.json> \
  --wrist-intrinsics <d405_intrinsics.json>
```

输出已存在时停止，不覆盖。当前支持一条 episode 的数值数据和每路视频各自落在单个文件中的布局，使用 v3 元数据定位 episode，按视频时间戳截取，保持原图 640×480。
`<setup.yaml>` 需包含采集时有效的前视内参和 D435IF `T_base_camera`；三个本地文件的得出方法见
[`docs/piper_setup.md`](../../../docs/piper_setup.md)。

## 数据含义

- `right.obs_ee_pose` / `right.cmd_ee_pose`：基座系 TCP 位姿，`xyz + quaternion(wxyz)`，位置单位米。
- `right.*_ee_pose_camframe`：对应的 D435IF 光学坐标系 TCP 位姿；后续加载器使用这些字段时不要再次执行基座到相机变换。
- `right.obs_joints` / `right.cmd_joints`：六个模型关节角，弧度。
- `right.obs_gripper` / `right.cmd_gripper`：硬件报告/指令行程，米；不等同于胶垫之间的精确净间隙。
- `images.front_1` / `images.right_wrist`：D435IF / D405 的 RGB JPEG。
- `source.observation_state` / `source.action`：保留原始硬件状态和归一化动作，便于核对。
- embodiment 为 `piper_right_arm`，编号 7；right 只是单臂数据键约定。

动作标签取同一行记录的控制指令，不以观测状态代替，也不擅自移动时间索引。关节 action 按 follower 的范围还原至原始角度，并按驱动规则向零截断到整数；夹爪 action 使用 `[0,100]` 百分比。提供的 action calibration 必须对应采集时的范围；仅从 parquet 的 action 数值无法证明历史范围。所用参数、相机标定及腕部内参均完整保存到 Zarr metadata。

动作分块、归一化和人类朝向对齐留给后续 embodiment/训练适配。当前 TCP 使用 link6 原生轴，尚未转换为 Eva/人类标准轴。

## 检查产物

`*_overlay.mp4` 为整条示范的 D435IF 投影视频，绿色圆圈表示反馈 TCP，橙色圆圈表示指令 TCP，RGB 线段表示反馈 TCP 的 XYZ 轴；`*_overlay.jpg` 是六帧预览。`*_report.json` 记录帧数、视频索引对齐误差和投影范围。

视频时间戳匹配只验证文件索引对应，不代表已经测量相机与关节传感器的真实采样延迟。投影图用于检查明显错位，不是独立的毫米级外参测量。

## A 组训练配置

在工程根目录激活 `source emimic/bin/activate` 后运行。

```bash
# 单卡流程检查：batch=2，10 次更新，1 个验证 batch，保存 best/last checkpoint。
python -m egomimic.trainHydra --config-name train_piper_smoke

# A 组起始训练配置：batch=8，10000 次更新。
python -m egomimic.trainHydra --config-name train_piper
```

默认读取 `.local_calibration/piper_all`。迁移机器后用
`paths.dataset_dir=/absolute/path/to/piper_all` 指定已转换数据。
输出位于 `.local_calibration/training/<name>/<timestamp>/`，包括 Hydra 配置、
训练集归一化统计、CSV loss 和 `checkpoints/last.ckpt`。
首次运行 ResNet18 会使用 ImageNet 权重缓存，缓存缺失时需要下载。

机器人数据按 episode 固定随机划分训练和验证（seed 固定），不按帧拆分；对照组必须沿用
同一划分。归一化统计只用训练集帧。

输入为 D435IF、D405 各一帧 RGB 和 7 维 TCP 状态；模型内将图像缩放至
224×224，磁盘图像及其标定参数不变。状态/动作均为 D435IF 光学坐标系中的
`[x,y,z,yaw,pitch,roll,gripper]`，单位米/弧度，预测包含当前帧的 30 帧动作。
模型沿用 Eva 配置的 16 层、256 维 HPT trunk，适配单臂 stem 和 7 维 flow matching head。
HPT 从头训练，图像编码器使用 ImageNet 初始化。后续 B 应保持相同机器人分支。

小规模检查分两步：先确认有限 loss、参数更新、动作形状 `[B,30,7]`、保存与恢复；
再固定少量 episode 做数百步过拟合检查，观察 loss 趋势与预测轨迹。
10 步通过只代表训练链路可用，不代表抓取成功率已得到验证。
smoke 配置使用同一划分，只减少 batch 和更新次数。

恢复检查示例（将路径换为实际运行目录）：

```bash
python -m egomimic.trainHydra --config-name train_piper_smoke \
  ckpt_path=/absolute/run/checkpoints/last.ckpt \
  norm_stats.precomputed_norm_path=/absolute/run/norm_stats \
  trainer.max_steps=20
```

CPU 可检查链路：追加 `trainer.accelerator=cpu trainer.precision=32-true`。
单卡从 batch=2 开始，确认显存余量再增加 batch。
如在 Slurm 集群运行，应先申请 GPU allocation。
