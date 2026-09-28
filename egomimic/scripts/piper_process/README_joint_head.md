# Piper专用关节动作头

## 目标

保留HPT共享视觉/任务表征和Cartesian pose监督，同时为Piper增加直接关节控制头。部署读取关节头，不再经过`TCP pose -> IK -> joint command`，因此模型输出不存在IK无解或IK分支跳变。模型仍可能产生不平滑、越界或碰撞动作，这些需要单独评估。

FM与MLP被视为独立实验维度，不预设哪一个更好：

|变体|Cartesian pose head|Piper joint head|
|---|---|---|
|joint-flow|Flow Matching|Flow Matching|
|joint-mlp|Chunk MLP|Chunk MLP|

两者共享相同的ResNet18、Piper/Human域内stem、16-block HPT trunk、数据划分、归一化、动作时域和训练预算。

## 数据定义

Piper输入：当前主视角RGB、当前腕部RGB、`[q1..q6, gripper]`。关节单位rad，夹爪单位m。Cartesian状态不再作为Piper proprio输入，避免重复提供由同一组关节FK得到的信息；Human仍使用其6维末端状态。

Piper监督：

- `actions_cartesian[30,7]`：原有相机坐标TCP `[xyz,yaw,pitch,roll,gripper]`，继续训练pose head。
- `actions_joints[30,7]`：采集时实际发送给follower的绝对关节指令 `[q1..q6,gripper]`，直接读取转换数据中的`right.cmd_joints/right.cmd_gripper`。不是用Cartesian标签重新做IK生成的伪标签。

Human只有Cartesian pose监督，不产生虚构的Piper关节标签。`piper_human_joint`数据配置只给Piper增加关节字段。

第一版使用绝对关节角。它与采集命令一一对应、部署无需积分、重新查询时没有累积漂移。若绝对角回归出现平均化，可做第二个表示消融：

`delta_q[t,h] = q_command[t+h] - q_observed[t]`

这里每个未来时域都相对当前真实反馈，而不是预测逐步相加；部署每次重观测后用当前反馈恢复绝对目标。夹爪仍建议输出绝对连续开度。不要先使用相邻增量`q[t+h]-q[t+h-1]`递归积分，因为长动作段会累积误差。

## 损失关系

当前HPT先计算robot pose loss，再计算robot joint auxiliary loss。B组一个训练step包含一个robot batch和一个human batch，模型最后除以两个domain，因此等效权重为：

`0.5 * robot_joint + 0.5 * robot_pose + 0.5 * human_pose`

这与参考图的三项比例一致。A组没有human batch，使用`robot_joint + robot_pose`。各head内部均在各自量化归一化空间计算损失，避免直接把rad、m混加。正式报告还应分别记录每个head的loss；现有总loss日志不足以判断是哪一支改善。

## 配置

|实验|配置|
|---|---|
|A + FM双头|`train_piper_a_joint_flow.yaml`|
|B + FM双头|`train_piper_b_joint_flow.yaml`|
|A + MLP双头|`train_piper_a_joint_mlp.yaml`|
|B + MLP双头|`train_piper_b_joint_mlp.yaml`|

数据配置为`piper_joint.yaml`和`piper_human_joint.yaml`；模型配置为`hpt_piper_joint_{flow,mlp}.yaml`及相应human版本。A组FM/MLP均已完成短训练、反向传播、final checkpoint保存和单轨迹评估链路检查；B组MLP也已完成双域联合反向传播检查，四份Hydra配置均可完整组合。该检查只证明链路可运行，随机/单步权重的误差没有模型效果含义。

## 评估

`evaluate_joint_heads.py`在固定7条验证轨迹上报告：

- 六关节第一步/整段MAE与整段RMSE（deg）；
- 预测关节FK后的TCP位置/朝向误差；
- 连续预测点的关节步长，观察动作段内部跳变；
- URDF关节限位越界率；
- 夹爪开度MAE，以及50mm阈值下的F1。

评估命令示例：

```bash
python -m egomimic.scripts.piper_process.evaluate_joint_heads \
  --checkpoint PATH/final.ckpt \
  --output .local_calibration/joint_eval/VARIANT
```

FM保留多个推理seed；MLP为确定性。推理seed不能当作独立训练seed。正式比较至少包含：旧pose-only FM/MLP离线Cartesian结果、四个新A/B×FM/MLP关节头结果，以及最终真机成功率。关节头与旧模型的输入表示和监督任务均有变化，不能把差值全部归因于“绕开IK”。

训练建议先保持原有10000步预算，以完整7条验证轨迹选模；若曲线仍明显下降，再把同一预注册配置延长到30000步。不要直接采用参考图中的100步时域或30000步结论：本项目数据是30Hz、当前动作段30步，变更时域会同时改变任务和算力。

checkpoint选择以直接部署目标为主：关节RMSE、FK位置误差、夹爪F1和越界率；pose head作为共享表征的约束与Human离线指标单独报告。最终指标仍是每条轨迹多次真机抓取成功率，离线关节误差不能替代闭环实验。
