# Piper L 部署

本目录是 Piper L 的实机部署代码。每个场景的本地参数（相机序列号、内参、手眼外参、
起始姿态）不在仓库中，按 [`docs/piper_setup.md`](../../../docs/piper_setup.md) 得出，并用
`--calibration <setup.yaml>`（或环境变量 `PIPER_SETUP`）传入。机型常数在 `piper_l.yaml`，
约定见 [`docs/calibration/piper_parameters.md`](../../../docs/calibration/piper_parameters.md)。

| 文件 | 作用 |
|---|---|
| `config.py`、`piper_l.yaml`、`setup.example.yaml` | 机型常数 + 本地 setup 的合并加载 |
| `robot_interface.py` | 关节/夹爪单位换算、FK/IK、SDK 与 RealSense 接口 |
| `rollout_joint.py` | 关节输出模型（MLP / FM / 残差关节头）的部署 |
| `rollout.py` | 笛卡尔 TCP 输出模型的部署（IK，需要前视外参） |
| `control_check.py` | 不加载模型的控制链往返检查 |
| `realsense_setup.py`、`initial_pose.py` | 得出相机参数与起始姿态（见设置流程） |

部署附加依赖见 `requirements-deploy.txt`：

```bash
uv pip install --python emimic/bin/python -r egomimic/robot/piper/requirements-deploy.txt
```

## 安全约定

- 退出遥操和其他占用相机或向 follower 发指令的程序；现场人员准备急停。
- 不加 `--execute` 时只读：读取相机和反馈、推理并记录，不使能、不运动。
- 程序不自动回位。结束、Ctrl+C 或异常时，执行模式发送 SDK 快速停止
  `EmergencyStop(0x01)`，不失能、不归零；下次执行先发恢复指令，再使能并检查 CAN 控制状态。
- 连接时强制把 follower 设为固件层的输出角色，进入 MOVE-J 后预置当前关节位置，再发送
  连续绝对关节目标。没有桌面、障碍物或自碰撞模型，低速不能代替现场观察。

## 关节输出模型：`rollout_joint`

输入为前视 D435IF 与腕部 D405 当前 RGB（640×480）以及 `[q1..q6 (rad), gripper (m)]`；
输出 `30×7` 绝对关节目标（残差关节头为当前关节加有界残差）。部署不经过笛卡尔位姿或 IK，
模型关节坐标按 setup 中的符号和零位换回 SDK 反馈坐标后发送。图像缩放、ImageNet 归一化、
状态归一化都复用 checkpoint 内的训练配置和统计。

```bash
# 只读检查图像、反馈和模型输出
python -m egomimic.robot.piper.rollout_joint --checkpoint <ckpt> --calibration <setup.yaml> --duration 5

# 低速短时执行，确认方向后再加长 --duration
python -m egomimic.robot.piper.rollout_joint --checkpoint <ckpt> --calibration <setup.yaml> \
  --execute --speed-percent 30 --duration 3 --execution-mode chunk --query-every 30
```

### 执行方式

- `--execution-mode ensemble`（默认）：每步重新预测，把过去各 chunk 对当前时刻的预测按
  `exp(-decay * age)`（`--ensemble-decay`）加权平均。
- `--execution-mode chunk --query-every N`：执行 N 步预测后再观测、重规划（`--chunk-size` 同义）。
  残差关节头每次重规划都以当前反馈为锚点，首步的系统偏差会随重规划次数累积；
  这类模型宜用较长的 N。
- `--execution-mode time-ensemble`：每次推理后按 `--control-hz` 发送 `--query-every` 条指令，
  各 chunk 按墙钟时间线性插值到当前时刻，再按龄加权。
- `--control-hz`：指令发送频率，默认与训练数据相同的 30 Hz；实际频率还受推理和相机读取耗时限制。
- `--speed-percent`：SDK 关节速度百分比（不是末端线速度）。
- `--gripper-effort 0..5000`：夹爪力度（0.001 N·m），应与示教时一致。
- `--flow-seed`：FM 头的固定初始噪声，使重复实验可复现；MLP 和残差头为确定性输出。

### 起始状态

- `--initialize --initial-pose <initial_pose.json>`：先以 `--init-speed-percent`（默认 20%）回到
  起始姿态，反馈连续三次进入 1°/5 mm 容差后才开始计时。过程写入 `initialization.jsonl` 和
  `initialization.json`。保存姿态来自真实反馈，可能因零位偏差略超 URDF 范围，执行前会先
  裁剪到可命令范围。
- 进入执行后先把夹爪张开到 `--start-gripper-mm`（默认 100 mm）并确认到位，再读第一帧策略输入；
  示范从张开状态开始，夹爪开度偏小会被模型误判为抓取阶段。`--start-gripper-mm 0` 跳过，
  `--min-start-gripper-mm` 设定起始开度下限。

### 运行中的检查

- 要求 CAN/MOVE-J 控制状态和六个电机使能连续稳定 0.5 s 后才启动；控制器离开有效状态立即停止。
- 拒绝相对当前反馈跨度超过 `--max-chunk-displacement-deg`（默认 55°）的 chunk；指令领先真实关节
  超过 `--max-tracking-error-deg`（默认 30°）时停止；`--max-command-step-deg` 限制单次指令增量。
- URDF 限位：超出不超过 `--joint-limit-tolerance-deg`（默认 3°）的预测裁剪回边界，更大则停止。
  follower 标定范围可能略超 URDF 名义边界（如 J2/J3 的 0°），因此示范标签也可能轻微越界。
- `--tracking-gate-deg`：大于 0 时，同一目标保持到反馈进入该误差后再推进，期间保持当前夹爪开度；
  超过 `--tracking-gate-timeout` 停止。
- MLP 头输出无界：部署先把归一化预测裁剪到 `[-1, 1]`，再限制在训练动作各 horizon 的 1%–99%
  分位范围，最后应用 URDF 限位。FM 头沿用 z-score 归一化，不做这一裁剪。

### 日志

默认写入 `.local_calibration/rollouts_joint/<时间戳>/`（`--output` 可改）：`config.json` 记录全部参数、
checkpoint 哈希和代码版本；`commands.jsonl` 逐条记录原预测、实际命令、越界量、控制状态和
是否发送；每 `--save-every` 次推理保存一份无压缩 NPZ（两路图像与完整预测），被拒绝的 chunk
总会保存。异常写入带运行阶段和 traceback 的 `error.json`。

## 笛卡尔输出模型：`rollout`

用于输出相机坐标系 TCP 动作段 `[xyz, yaw, pitch, roll, gripper]` 的模型。每步转换到基座坐标后
做数值 IK（以当前关节角为初值，段内以上一解为初值），位置残差超过 5 mm 或姿态残差超过 3°
时不发送。需要 setup 中的前视外参，且 TCP 必须与训练标签使用的运动学一致（包括拟合的 TCP 修正）。

```bash
python -m egomimic.robot.piper.rollout --checkpoint <ckpt> --calibration <setup.yaml> --duration 5
python -m egomimic.robot.piper.rollout --checkpoint <ckpt> --calibration <setup.yaml> \
  --execute --speed-percent 10 --duration 30 --query-every 1
```

日志写入 `.local_calibration/rollouts/<时间戳>/`：`config.json`、`query_*.npz`（图像、反馈、模型输入、
完整预测）、`commands.jsonl`（TCP 目标、IK 解、残差、耗时、是否发送）。

## 控制链检查：`control_check`

```bash
python -m egomimic.robot.piper.control_check --execute --calibration <setup.yaml>
```

默认以 10% 速度让 J1 正向移动 2°，连续三次进入 0.3° 误差后返回原角度并保持 1 s，只用 SDK
绝对关节位置控制；结果写入 `.local_calibration/control_checks/<时间戳>/`。可用 `--joint`、`--delta-deg`、
`--speed-percent`、`--timeout` 调整。
