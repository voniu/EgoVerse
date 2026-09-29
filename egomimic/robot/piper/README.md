# Piper L：MLP 部署

使用现有无状态历史的 MLP A/B checkpoint。A 默认使用10000步 `final.ckpt`。

输入：D435IF与腕部D405当前RGB（640×480），原始SDK关节反馈经过URDF FK与标定得到的当前TCP相机坐标状态 `[xyz, yaw,pitch,roll, gripper]`，共7维。单位米/弧度。图像缩放、ImageNet归一化、状态归一化均复用checkpoint内的训练配置和统计。

输出：`30×7` TCP动作段，逐步转换为基座坐标并做数值IK。IK以当前关节角开始，段内以上一解为初值；SDK接收0.001度关节命令和0.001mm夹爪开口。夹爪保持连续开度，限定在Piper L物理行程0–100mm，不改成二值信号。此版本没有关节输出神经网络头。

## 启动

在工程根目录执行。退出遥操和其他占用两台相机／发送follower命令的程序，将机器人与物体摆到示范中的开始位置；由现场人员准备急停。脚本不会自动归位。相机和基座须保持原标定位置。

```bash
cd /mnt/wd10/zekun/code/EgoVerse
source emimic/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

默认不发送任何运动指令，可先检查图像、反馈、模型输出和IK：

```bash
python -m egomimic.robot.piper.rollout --duration 5
```

正式执行MLP A，SDK速度10%，30秒后结束：

```bash
python -m egomimic.robot.piper.rollout \
  --execute --speed-percent 10 --duration 30 --query-every 1
```

切换到MLP B仅更换checkpoint：

```bash
python -m egomimic.robot.piper.rollout \
  --checkpoint .local_calibration/hpc/mlp/checkpoints/b/final.ckpt \
  --execute --speed-percent 10 --duration 30 --query-every 1
```

参数说明：

- `--speed-percent 10`：SDK关节运动速度百分比，不等于末端固定线速度。没有额外关节增量限速或轨迹平滑。
- `--query-every 1`：每次观察后只执行预测的第一步，然后使用最新反馈和图像重新规划。该值也是默认值。增大该值会开环执行动作段，实机上可能遇到后续姿态不可达或累积误差；推理、相机读取、日志与IK都有耗时，因此实际频率低于30Hz。
- `--port can_follower`，腕部序列号默认`427622273943`，主相机序列号从`calibration.yaml`读取。
- `--duration`：包括推理与发送的运行时间。结束、Ctrl+C或异常时，执行模式会发送SDK快速停止`EmergencyStop(0x01)`，不会调用失能、归零或自动回家。下一次执行会先发送恢复指令，再重新使能并检查CAN控制状态。

执行模式先预置当前关节位置再请求使能，并随每条关节命令刷新MOVE-J控制模式；否则本机Piper约0.2秒后会退回STANDBY并忽略后续目标。基本有效性检查仅拒绝过期反馈、非有限输出，以及IK无法满足的目标（位置残差超过5mm或姿态残差超过3度）；IK在URDF关节范围内求解。没有桌面、障碍物或自碰撞模型，低速也不替代现场观察。

## 日志与离线检查

每次运行保存到 `.local_calibration/rollouts/时间戳/`：

- `config.json`：checkpoint、执行速度、查询间隔等。
- `query_*.npz`：两路原始RGB、硬件反馈、模型输入状态、完整30步预测。
- `commands.jsonl`：各步TCP目标、IK关节解、残差、耗时及是否发送。

离线核对，不连接机械臂：

```bash
python -m egomimic.robot.piper.evaluate
```

此检查覆盖部署推理与已有A/B评估的一致性、已知姿态的FK/IK往返、模型预测姿态的可达性，以及SDK发送单位。报告位于 `.local_calibration/deployment_check.json`。离线通过不代表已验证实机抓取成功或IK全程连续。

当前离线结果：A/B各3帧的部署预测与已有评估在浮点容差内一致；20个已知关节姿态的FK/IK往返最大位置残差约0.0001mm，发送单位检查通过。所抽取6个动作段的前10步中，51/60个满足IK精度要求，9个未满足，均出现在近零位的第0帧动作段（A4个、B5个）。这不是全面可达性保证，也不说明目标全局无解；当前局部求解器在这些输入上未达标。执行时若出现此问题，日志会保留模型目标和报错，不会把该解发送给机器人。

2026-09-29完成3秒实机闭环检查，日志为`.local_calibration/rollouts/20260929_014948`。逐帧重规划共发送23条命令，TCP净移动80.4mm，最大单次反馈关节变化2.41度，IK最大残差2.92mm/2.74度，无超限退出。该检查验证了SDK控制链和短时闭环运行，不等同于完整抓取成功率测试。

## Joint Head MLP A

Joint Head 使用相同的两路 RGB，状态改为当前硬件关节位置和夹爪开度
`[q1..q6(rad), gripper(m)]`，直接输出 `30×7` 绝对关节目标。部署不经过
Cartesian pose 或 IK：模型关节坐标根据 calibration 中的符号和零位转换回 SDK
反馈坐标后发送。默认 checkpoint 是 A 组验证集最佳模型（step 5300）：

```bash
python -m egomimic.robot.piper.rollout_joint --duration 5
```

上述命令只读取相机和 follower 并记录预测。退出遥操、确认场景已摆到示范起始状态并
准备好急停后，由现场人员执行低速三秒检查：

```bash
python -m egomimic.robot.piper.rollout_joint \
  --execute --speed-percent 30 --duration 3 \
  --execution-mode chunk --chunk-size 30
```

确认三秒动作方向后，可将 `--duration` 改为 `30`。日志写入
`.local_calibration/rollouts_joint/时间戳/`，其中保存每次观测、完整动作段、首个目标
相对当前反馈的关节差和是否实际发送。

执行模式启动后先以 20% 速度回到 `initial_pose.json` 保存的六关节和夹爪姿态，反馈
连续三次进入 1 度/5mm 容差后，才切换到策略速度并开始计算 `--duration`。初始化结果
逐帧写入 `initialization.jsonl`，成功摘要写入 `initialization.json`；异常或 Ctrl+C
另存为带运行阶段和 traceback 的 `error.json`。可用 `--init-speed-percent`、`--init-timeout`、
`--init-joint-tolerance-deg` 和 `--init-gripper-tolerance-mm` 调整；临时绕过使用
`--skip-initialize`。

保存姿态来自真实反馈；限位附近的反馈可能因零位偏差轻微落在 URDF 范围外。初始化命令
会先裁剪到 URDF 可命令范围。本机成功首帧中的 J2=-2.387°、J3=+1.994° 因此按
J2=0°、J3=0°执行，初始化期间每秒打印一次剩余误差。

默认还会拒绝相对当前反馈跨度超过 55 度的预测 chunk，并在命令领先真实关节超过
30 度时停止。两项阈值分别由 `--max-chunk-displacement-deg` 和
`--max-tracking-error-deg` 配置。初始化姿态取自 `20260929_105005` 的首帧反馈，
J2/J3 分别位于 13.899°/-9.608°，避开二者的 0°边界。

部署参数均可从命令行调整：

- `--speed-percent`：Piper SDK 速度百分比，默认 30。
- `--execution-mode ensemble|chunk`：时间集成或连续执行一个 chunk 前缀。
- `--chunk-size 1..30`：chunk 模式中每次重规划前执行的预测步数；等价于
  `--query-every`。模型本身固定输出 30 步，改变这里不会改变网络结构。
- `--control-hz`：命令发送频率，默认及训练数据频率均为 30Hz。
- `--ensemble-decay`：ensemble 模式中新旧预测的指数权重衰减。
- `--save-every`：每隔多少次推理保存一份图像和完整预测，默认 10；设为 0
  可关闭。保存采用无压缩 NPZ，避免 RGB 压缩阻塞控制循环。

2026-09-29 的只读实机检查使用当前姿态完成 52 次推理，平均推理时间约 16ms，输出
全部有限且位于 URDF 关节范围内。当前姿态到首个预测目标的最大关节差约 13.4 度，
主要来自 J5；执行时 SDK 速度百分比限制运动速度，但模型输出仍是绝对位置目标。

URDF 中 J2/J3 的名义边界为 0 度，但本机 follower 标定范围分别延伸到
J2=-2.1 度、J3=+2.0 度，真实反馈和训练标签会轻微跨过 URDF 边界。Joint rollout
默认把不超过 3 度的预测越界裁剪回 URDF 边界，超过 3 度仍停止，并在
`commands.jsonl` 同时记录原预测、实际命令和各关节越界量。

MLP 动作头的最后一层是无界线性层，而训练的 quantile-normalized 动作主要位于
`[-1,1]`。部署时先把归一化预测裁剪到该区间，限制在各 horizon 的训练动作
1%–99% 分位范围，再应用 URDF 限位。日志和 NPZ 保留未约束预测、归一化越界量、
训练分布裁剪结果和最终关节命令。

Joint Head 训练目标是 30 步 action chunk。当前部署默认每步重新预测，并将过去 chunk
对当前绝对时刻的预测按
`exp(-0.05 * age)` 加权平均。它既保留旧 chunk 后续的推进信息，也利用新图像修正目标，
避免只执行 horizon 0 停在 J2/J3 边界，以及每 10 步切换整段目标产生的跳变。旧行为可用
`--execution-mode chunk --query-every 10` 复现。

70 条机器人示范的初始夹爪开度接近 100mm。执行模式默认要求初始开度至少 90mm；若
当前夹爪只有约 70mm，模型输入已偏离示范起始分布，容易把它解释为抓取阶段并过早闭合。
需要有意绕过此检查时可传 `--min-start-gripper-mm 0`。

部署附加依赖见 `requirements-deploy.txt`。本工程虚拟环境没有pip时使用：

```bash
uv pip install --python emimic/bin/python -r egomimic/robot/piper/requirements-deploy.txt
```
