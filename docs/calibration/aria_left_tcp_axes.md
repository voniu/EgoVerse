# Aria 左手与 Piper TCP 的朝向约定

当前状态：固定旋转已通过用户视觉核对，并完成 50 条录像的骨架一致性检查。
这是本实验选择的人机轴语义对应，不是 Aria 或 EgoVerse 规定的唯一映射。
固定参数保存于 `egomimic/robot/piper/aria_left_alignment.json`，
通过转换器的 `--alignment` 参数接入。原始试转换 Zarr 与 A 组数据未修改。

## 机器人端依据

`egomimic/robot/piper/calibration.yaml` 中 TCP 为 link6 下平移 `[0,0,0.135]`，
旋转为单位旋转。用户指定的完整 URDF：

```text
/mnt/wd10/zekun/code/Piper-Calibration-ROS2/ros2_ws/src/agx_arm_urdf/piper_l/urdf/piper_l_with_gripper_description.xacro
```

夹爪基座固定关节没有旋转，手指沿 link6 的 Z 方向伸出；将两个手指的
prismatic joint 轴经过各自 origin 旋转并考虑 mimic 符号后，开合沿 ±Y。
因此 TCP +Z 为腕部到指尖方向，Y 为开合轴，X 由右手规则确定。
对称平行夹爪的开合轴本身没有唯一正号，本实验固定 +Y 的人类对应方向，
不在逐帧处理中翻转符号。

## 人类端定义

保留掌心作为位置原点，用掌骨方向定义工具朝向：

| 工具轴 | 左手对应方向 |
|---|---|
| +Z | 手腕 → 中指根部（MCP），代表沿手指伸出的方向 |
| +Y | 小指根部 → 食指根部，再投影至 Z 的垂直平面，代表手掌横向 |
| +X | Y × Z，补齐右手坐标系 |

采用指根而非指尖，避免手指弯曲让工具坐标轴随之大幅改变。
Aria 的关键点索引为 wrist=5、index proximal=8、middle proximal=11、
pinky proximal=17、palm=20，依据 [Aria 官方数据格式](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/hand_tracking/hand_tracking_outputs)。

这里的“沿手指方向”只是可解释的末端轴约定。人的实际接近速度未必沿此轴，
也不等于任意抓取时的接触面法向。掌心与真实夹持中心也不是同一个物理点。
本步骤只统一朝向语义，不估计接触点、夹爪开度或机器人可执行抓取姿态。

## 固定旋转

在第 17 条试转换的 201 帧上，分别计算解剖坐标系相对于原生手腕的旋转，
取 SO(3) 旋转均值。此后各录像使用同一个旋转，不逐帧重新拟合：

```text
R_camera_tool = R_camera_native_left_wrist @ R_wrist_tool

R_wrist_tool ≈
[[ 0.028544916, -0.014875162,  0.999481824],
 [ 0.992751192,  0.117205234, -0.026608340],
 [-0.116748697,  0.992996306,  0.018112948]]
```

矩阵每一列是一个工具轴在原生手腕系中的表达。实际使用完整精度 JSON。
这是右乘的局部工具轴旋转；掌心 xyz 不变，不旋转图像，不镜像左手，
也不改相机外参。姿态先作矩阵乘法，最后再提取 ZYX 欧拉角，不能直接加减欧拉角。
在世界系先右乘再转换到当前相机系，与在当前相机系右乘的结果一致。

## 验证与边界

- 固定矩阵满足正交性及 det=+1；不是反射矩阵。
- 合成几何检查通过：标准轴输出单位矩阵；整体平移不影响朝向；
  整体旋转后输出相同旋转。
- 50 条录像共 10,218 个置信度 ≥0.5 的左手跟踪记录用于检查，包含拟合录像。
  与每帧骨架重新构造的轴相比，最大角差 2.507°；各录像 p95 中最大者为 1.137°。
  这些是手跟踪记录数，不是 RGB 对齐、VIO 过滤后的训练帧数。
- 关键点和手腕位姿来自同一 SDK 模型。上述误差反映内部一致性，
  **不能解释为真实世界朝向误差或人机迁移精度**。

## 复现与查看

```bash
source emimic/bin/activate
python -m egomimic.scripts.aria_process.align_left_tcp \
  .local_calibration/aria_gen2_pilot/Put_bear_into_basket_17_20260919_163143_left_0034_0235.zarr \
  --output .local_calibration/aria_gen2_alignment \
  --validation-dir /mnt/wd10/zekun/data/Aria_Gen2/raw
```

输出 `alignment.json`（完整旋转及逐录像误差）、`comparison.mp4` 与 `comparison.jpg`。
左侧为原生手腕轴，右侧为 Piper 语义轴；X/Y/Z 分别为红/绿/蓝，黄色原点为掌心。
青色夹爪轮廓只是 60 mm 开口、70 mm 指长的示意，不是 URDF 渲染或预测开度。
图像显示该约定的几何含义；仍需结合完整抓取过程判断是否符合本实验的动作语义。
视觉核对已完成。批量结果位于 `.local_calibration/aria_gen2_aligned`；
正式 B 训练之前仍需检查裁剪片段完整性，并生成 B 组的人类归一化统计。
