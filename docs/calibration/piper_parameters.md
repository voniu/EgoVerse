# Piper L 参数：约定与来源

机型常数（URDF、关节名、SDK 原始单位换算、名义 TCP）在
[`piper_l.yaml`](../../egomimic/robot/piper/piper_l.yaml)，对所有 Piper L 通用。相机序列号、
内参、手眼外参、拟合修正和起始姿态属于某一台机器和场景，放在仓库外的 setup 文件中
（模板 [`setup.example.yaml`](../../egomimic/robot/piper/setup.example.yaml)），按
[`docs/piper_setup.md`](../piper_setup.md) 的流程在每个新场景得出。加载时 setup 覆盖机型常数
（`egomimic/robot/piper/config.py`）。

## 坐标约定

平移单位为米，角度为弧度，四元数顺序为 `[x, y, z, w]`。`T_parent_child` 将子坐标系中的点
转换到父坐标系。由四元数构造旋转矩阵时先归一化；配置保留原始数值。

- `T_base_tcp = T_base_link6 @ T_link6_tcp`
- 相机坐标系 TCP 标签：`T_camera_tcp = inverse(T_base_camera) @ T_base_link6 @ T_link6_tcp`

## 相机内参

- 使用 RealSense SDK 读取的设备出厂标定（`realsense_setup.py write`），对应彩色 RGB、
  640×480、30 fps、无旋转的流。裁剪、缩放、旋转或换分辨率后必须重新读取或相应变换。
- 内参属于相机本身，与安装位置无关；更换相机后必须重新读取，拆装同一台相机不改变内参。
- 腕部 D405 的内参写入 setup 旁的 JSON（`cameras.d405.intrinsics_file`），转换时由
  `lerobot_to_zarr --wrist-intrinsics` 读入，并嵌入每条 Zarr 的元数据。
- 当前关节残差模型只把图像输入编码器，前向计算不使用内参或腕部外参；内参用于数据元数据、
  投影检查和几何任务。

## 前视相机外参

- `cameras.d435if.extrinsics` 为眼在手外（eye-on-base）的 `T_base_camera`，由手眼标定得出。
- 相机或机械臂基座移动后，旧外参失效；重新读取内参不能恢复外参。
- 转换后的 Zarr 在 `attrs.extrinsics.right` 保存当时的外参，后续工具优先使用它，
  因此已转换的数据可以独立复现。

## TCP

- 名义 TCP：`link6` 下 `[0, 0, 0.135] m`，子坐标系 `piper_tcp`，位于夹爪中心线上、
  比模型最前端指尖平面向内 7.5 mm；坐标轴与 `link6` 相同。这是模型参考点，未经实体测量。
- 若投影检查显示 TCP 偏离夹爪中心，用 `tcp_calibration` 工具拟合 TCP 偏移和关节零位偏差，
  结果存为单独的修正 JSON，由 `apply_pose_correction` 重算已转换数据的位姿标签。之后凡是
  由关节算位姿的地方（笛卡尔头的 IK、FK 评估）都必须使用同一份修正。
- 不要把拟合的零位偏差写进 setup 的 `kinematics.feedback_to_model.offsets_rad`：它会改变
  模型的关节输入，而关节数据和训练都使用未修正的关节角。

## 运动学与原始反馈

- `piper_l_reference.urdf` 与 Piper L ROS2 描述包（`agx_arm_urdf` 的 `piper_l_description.urdf`）
  六关节的父子关系、轴和原点相同；夹爪几何不在该 URDF 中。
- SDK 原始关节反馈单位为 0.001°，原始夹爪反馈单位为 0.001 mm：
  `q_model = signs * radians(q_raw * 0.001) + offsets_rad`，默认 `signs` 全为 1、`offsets_rad` 全为 0。
  ROS `joint_states` 已是弧度，不能再次应用原始单位换算。
- LeRobot 的归一化 action 不是原始关节反馈，须用 follower 关节标定 JSON 还原
  （`lerobot_to_zarr --action-calibration`）。
- 软件零位偏移为零不代表硬件物理零位无误差；需要时用上面的 TCP 拟合检查。
