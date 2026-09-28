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
  --execute --speed-percent 10 --duration 30 --query-every 10
```

切换到MLP B仅更换checkpoint：

```bash
python -m egomimic.robot.piper.rollout \
  --checkpoint .local_calibration/hpc/mlp/checkpoints/b/final.ckpt \
  --execute --speed-percent 10 --duration 30 --query-every 10
```

参数说明：

- `--speed-percent 10`：SDK关节运动速度百分比，不等于末端固定线速度。没有额外关节增量限速或轨迹平滑。
- `--query-every 10`：每次观察后使用预测的前10步，按30Hz间隔发送，再重新观察。设为1则每次只执行第一步。推理、相机读取、日志与IK都有耗时，因此实际查询频率略低于理想3Hz；脚本不会为了追赶时间集中发送过期动作。
- `--port can_follower`，腕部序列号默认`427622273943`，主相机序列号从`calibration.yaml`读取。
- `--duration`：包括推理与发送的运行时间。结束、Ctrl+C或异常时，执行模式会发送SDK快速停止`EmergencyStop(0x01)`，不会调用失能、归零或自动回家。若设备保持急停状态，现场按已有操作方式恢复；脚本不会自动清除已有急停。

执行模式先预置当前关节位置再请求使能。基本有效性检查仅拒绝过期反馈、非有限输出，以及IK无法满足的目标（位置残差超过5mm或姿态残差超过3度）；IK在URDF关节范围内求解。没有桌面、障碍物或自碰撞模型，低速也不替代现场观察。

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

当前离线结果：A/B各3帧的部署预测与已有评估在浮点容差内一致；20个已知关节姿态的FK/IK往返最大位置残差约0.0001mm，发送单位检查通过。所抽取6个动作段的前10步中，51/60个满足IK精度要求，9个未满足，均出现在近零位的第0帧动作段（A4个、B5个）。这不是全面可达性保证，也不说明目标全局无解；当前局部求解器在这些输入上未达标。执行时若出现此问题，日志会保留模型目标和报错，不会把该解发送给机器人。尚未进行实时硬件测试。

部署附加依赖见 `requirements-deploy.txt`。本工程虚拟环境没有pip时使用：

```bash
uv pip install --python emimic/bin/python -r egomimic/robot/piper/requirements-deploy.txt
```
