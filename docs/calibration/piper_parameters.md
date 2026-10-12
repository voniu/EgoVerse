# Piper L 当前参数来源与约定

运行参数见 [calibration.yaml](../../egomimic/robot/piper/calibration.yaml)，目前尚未接入运行时加载器。本说明保存来源及确认记录。历史诊断中的候选关节补偿和相机外参不用于当前配置。

## 坐标约定

平移单位为米，角度为弧度，四元数顺序为 `[x, y, z, w]`。`T_parent_child` 将子坐标系中的点转换到父坐标系。将用户提供的四元数转成旋转矩阵时应归一化；配置保留原始数值。

`T_base_tcp = T_base_link6 @ T_link6_tcp`。

需要生成相机坐标系 TCP 标签时，
`T_camera_tcp = inverse(T_base_camera) @ T_base_link6 @ T_link6_tcp`。

## D435IF 内参

- 序列号：`327122075831`；固件：`5.17.0.10`。
- 读取时间：`2026-09-25T12:47:57.510423+00:00`。
- 来源：RealSense SDK 读取的设备出厂标定，不是新进行的光学校准；结果与此前一致。
- 对应图像：彩色 RGB、640×480、30 fps、无图像旋转。裁剪、缩放或旋转后需相应调整内参。
- 临时读取记录已按用户要求清理；内参数值保存在 `calibration.yaml`，读取时间、固件及图像规格记录于本节。

## D435IF 外参

- 平台已经拆装，先前的 `T_base_camera` 不再准确，因此仓库配置不包含 D435IF 外参。
- 原先转换完成的 70 条 Zarr 已保存当时使用的标定元数据，可继续复现实验。
- 转换其他原始数据或运行笛卡尔位姿控制前，需要另行提供与采集时安装状态对应的
  D435IF `T_base_camera`；重新读取内参不能恢复外参。

## D405 内参

- 文件：[`egomimic/robot/piper/d405_intrinsics_640x480.json`](../../egomimic/robot/piper/d405_intrinsics_640x480.json)。
- 相机：RealSense D405，序列号 `427622273943`，固件 `5.12.14.100`。
- 来源：RealSense SDK 读取的设备出厂标定和活动 stream profile，不是手工棋盘格重标定。
- 彩色流规格：RGB、640×480、30 fps、无图像旋转。
- 彩色内参：`fx=394.2607117`、`fy=393.7736206`、
  `cx=311.6419678`、`cy=242.5784149`；完整畸变参数和深度参数保存在 JSON。
- 内参属于相机成像系统，与 D405 安装在腕部的刚体位姿无关。拆下再装通常不改变
  内参，但会改变 `T_link6_camera` 外参；更换相机、分辨率、裁剪、旋转或图像缩放后
  必须重新读取或相应变换内参。
- 当前 Residual Joint Transformer 只把 D405 RGB 输入 ResNet，不使用 D405
  腕部外参，也不在前向计算中使用内参。因此当前训练和 Joint Head 部署不需要
  `T_link6_camera405`。内参用于数据元数据、格式核对以及后续投影/几何任务。
- 原先 70 条 Zarr 的 `intrinsics.right_wrist` 和 `wrist_intrinsics` 已嵌入同一组参数；
  直接训练这些 Zarr 不需要额外传入 JSON。重新执行 LeRobot→Zarr 转换时才通过
  `--wrist-intrinsics` 提供该文件。

## TCP

- 用户在浏览器模型中选择 `[0, 0, 135] mm`，以 `link6` 为父坐标系，子坐标系为 `piper_tcp`。
- 位置在夹爪中心线上，比模型最前端指尖平面向内 7.5 mm。
- 坐标轴采用与 `link6` 一致的默认定义；与人类/Eva 的末端朝向对齐属于后续单独的表示转换。
- 这是用户选定的模型参考点，未独立进行实体测量，查看器未建模胶垫。
- 历史 `[0, 0, 142.5] mm` 为模型最前端两指中点，旋转同样为单位阵；仅供追溯，不是当前 TCP。

## 运动学与原始反馈

核对的 ROS 工程根目录为 `/mnt/wd10/zekun/code/Piper-Calibration-ROS2`，以下路径相对于该目录：

- 启动文件：`launch/robot.launch.py`。
- 模型配置：`config/frames.yaml`。
- Xacro：`ros2_ws/src/agx_arm_urdf/piper_l/urdf/piper_l_with_gripper_description.xacro`。
- 关节发布器：`ros2_ws/src/piper_ros/src/piper/piper/piper_read_slave_joint.py`。

Xacro 引用的 `piper_l_description.urdf` 与本工程 `docs/calibration/piper_l_reference.urdf` 字节一致，六关节的父子关系、轴和原点一致。夹爪几何另由 Xacro 提供。参考 URDF SHA256：`d289652c494017ca8d50152343fdbd02fd25c4f1ac3bb80d59d1f2f3cfab6fda`。

发布器向 `joint_states` 发布 `joint1` 至 `joint6`，直接将 SDK 原始角度除以 1000 后从度转换为弧度，没有额外反向或零位补偿。ROS JointState 中的角度已经是弧度，不能再次应用原始值缩放。原始夹爪反馈每单位为 0.001 mm，ROS 发布器除以 1000000 转为米。

上述换算适用于之前检查的 LeRobot 原始硬件状态；归一化 action 不是原始关节反馈，须单独还原。配置不使用之前诊断拟合的零位补偿。软件偏移为零不代表硬件物理零位绝对无误差。

用户确认采集后未重新校零；相机安装连续性及零位连续性的确认适用于当前原数据转换。
