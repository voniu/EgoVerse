# 新机器 / 新场景的 Piper L 设置流程

本流程在一台新电脑或一个新场景（相机、桌面、机械臂基座任一移动过）上，从零得出采集、
转换和部署需要的全部本地参数。仓库只保存流程、程序和模板；得出的数值留在本地。

## 本地参数放在哪里

每个场景一个目录，放在仓库外或 git 忽略的 `.local_calibration/` 下，例如
`.local_calibration/setups/<场景名>/`：

| 文件 | 内容 | 由哪一步得出 |
|---|---|---|
| `setup.yaml` | 相机序列号、前视内参、手眼外参（模板 `egomimic/robot/piper/setup.example.yaml`） | 2、3 |
| `d405_<serial>_intrinsics_640x480.json` | 腕部 D405 内参 | 2 |
| `follower_calibration.json` | 遥操 follower 的关节范围（LeRobot 标定文件） | 4 |
| `initial_pose.json` | 部署起始姿态 | 7 |
| `tcp/`、`tcp_correction.json` | TCP 标注帧、标签、拟合报告和修正 | 6 |

各程序用 `--calibration <setup.yaml>` 读取 setup；也可 `export PIPER_SETUP=<setup.yaml>`。
setup 中的值覆盖机型常数 `egomimic/robot/piper/piper_l.yaml`，约定见
[`docs/calibration/piper_parameters.md`](calibration/piper_parameters.md)。

## 1. 环境

在工程根目录建立 Python 3.11 虚拟环境并安装本工程，部署机再安装硬件依赖：

```bash
uv venv emimic --python 3.11 && source emimic/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv pip install -r egomimic/robot/piper/requirements-deploy.txt
```

GPU 机器按其 CUDA 版本安装对应的 torch 构建。CAN 接口按 `piper_sdk` 的说明启用并命名
（程序默认 `can_follower`，可用 `--port` 修改）。

## 2. 相机序列号与内参

连接相机后：

```bash
python -m egomimic.robot.piper.realsense_setup list
python -m egomimic.robot.piper.realsense_setup write \
  --setup .local_calibration/setups/<场景名>/setup.yaml \
  --front <D435IF 序列号> --wrist <D405 序列号>
```

`write` 以录制和部署所用的流（640×480 RGB、30 Hz）读取出厂内参，写入序列号、前视内参，
并在 setup 旁保存腕部内参 JSON。setup 不存在时从模板创建；已存在时保留其中的外参。
`list` 同时显示 USB 类型：多台相机需都在 USB 3 上。

## 3. 前视相机手眼标定（eye-on-base）

前视相机固定、机械臂基座移动过后都要重做。用任一手眼工具（例如 ROS2 的 easy_handeye2，
标定板固定在 link6 上）求 `T_base_camera`，把平移和四元数 `[x, y, z, w]` 填入 setup 的
`cameras.d435if.extrinsics`，并在 `source` 写明工具和日期。

- 保存手眼工具的原始采样（各姿态的关节角和标定板角点），以后可重新联合求解。
- 检验：完成第 5 步转换一条 episode 后，查看输出目录里的 `<episode>_overlay.jpg/mp4`，
  投影的 TCP 坐标轴应落在两指中心；偏差明显时做第 6 步。

## 4. 示范采集与 follower 标定

用 LeRobot 采集遥操示范。把采集时生效的 follower 标定文件（LeRobot 缓存目录
`calibration/robots/piper/<follower-id>.json`）复制为 `follower_calibration.json`：转换时要用它把
归一化 action 还原为关节角。重新标定 follower 后，此前录制的数据必须配合旧文件转换。

## 5. 转换为 Zarr

```bash
python -m egomimic.scripts.piper_process.lerobot_to_zarr \
  --dataset <lerobot_dataset> --episode <i> --output <piper_zarr_dir> \
  --calibration <setup.yaml> \
  --action-calibration <follower_calibration.json> \
  --wrist-intrinsics <d405_intrinsics.json> \
  --task-name "<任务描述>"
python -m egomimic.scripts.piper_process.add_binary_gripper_residual_joint <piper_zarr_dir>
```

内参、外参和 follower 范围会写入每条 Zarr 的元数据，之后训练不再需要 setup。转换细节和
数据字段见 [`egomimic/scripts/piper_process/README.md`](../egomimic/scripts/piper_process/README.md)。

## 6. TCP 投影检查与修正（需要时）

当 overlay 中 TCP 明显偏离夹爪中心时，标注夹爪中心并拟合修正：

```bash
W=.local_calibration/setups/<场景名>/tcp
python -m egomimic.scripts.piper_process.tcp_calibration.select_frames \
  --dataset <piper_zarr_dir> --workdir $W --frames 80
python -m egomimic.scripts.piper_process.tcp_calibration.label_server --workdir $W
#   在 http://127.0.0.1:8765 点击两个指垫，标注约 25 帧（看不清的帧跳过）
python -m egomimic.scripts.piper_process.tcp_calibration.fit_tcp \
  --dataset <piper_zarr_dir> --workdir $W --calibration <setup.yaml>
```

`fit_tcp` 比较 M0（不修正）到 M4（TCP + 外参 + 关节零位）五个模型的留一法重投影误差
（LOO RMS）。选 LOO 误差接近最优的最简单模型；关节零位只放开有证据的关节
（`--fit-joints`）。确定后写出修正并重算位姿标签：

```bash
python -m egomimic.scripts.piper_process.tcp_calibration.fit_tcp \
  --dataset <piper_zarr_dir> --workdir $W --calibration <setup.yaml> \
  --fit-joints <j> --model <M1|M2|M3|M4> \
  --write-correction .local_calibration/setups/<场景名>/tcp_correction.json --name <名字>
python -m egomimic.scripts.piper_process.tcp_calibration.apply_pose_correction \
  --dataset <piper_zarr_dir> --correction .local_calibration/setups/<场景名>/tcp_correction.json
```

修正只作用于位姿标签；不要写进 setup 的 `feedback_to_model`（会改变模型的关节输入）。
以后由关节计算位姿的地方（IK、FK 评估）必须使用同一份修正。

## 7. 部署起始姿态

取示范首帧 follower 状态的均值，并检查各关节的离散度：

```bash
python -m egomimic.robot.piper.initial_pose \
  --dataset <lerobot_dataset> --output .local_calibration/setups/<场景名>/initial_pose.json
```

`joint_std_deg` 较大说明示范起点不一致，此时应改为在现场选定并记录起始姿态。

## 8. 部署前检查与执行

1. 不加载模型的控制链检查（小幅移动一个关节后返回）：
   `python -m egomimic.robot.piper.control_check --execute --calibration <setup.yaml>`
2. 只读运行（不发送运动指令）：`python -m egomimic.robot.piper.rollout_joint --checkpoint <ckpt>
   --calibration <setup.yaml> --duration 5`，检查图像、反馈和模型输出。
3. 执行：见 [`egomimic/robot/piper/README.md`](../egomimic/robot/piper/README.md) 的部署设置。
   现场人员准备急停；程序不会自动回位。
