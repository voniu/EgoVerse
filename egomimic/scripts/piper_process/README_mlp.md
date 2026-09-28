# MLP A/B 对照（2026-09-28）

## 为什么推进这条路线

FM的单片段拟合已达到4.54mm，但8观察仍为24.56mm。同一特征与标签，直接回归能准确记忆。最后一轮固定噪声的条件替换检查表明，FM确实使用条件：

|条件|对原目标误差 mm|对所提供条件对应目标误差 mm|
|---|---:|---:|
|正确条件|24.29|24.29|
|观察条件反序|149.85|24.07|
|所有观察使用平均条件|168.63|不适用|

条件反序使输出相对原输出改变145.08mm。因此不能将大误差解释为完全忽略观察条件。见 `.local_calibration/convergence_eight/conditioning_probe.json`。

teacher插值状态上的归一化速度MSE从t=1处约0.0168升到t=0处约1.174。t=0输入是纯动作，不含随机noise，而监督速度仍含noise，因而有不可预测成分；该数值不能直接证明末端实现错误。此检查也不能替代实际积分轨迹误差。当前没有可验证的FM代码修正，遂转入MLP正式对照，而非无限延长FM调参。

## 模型与公平性

保留原A/B的相机、TCP标签、归一化、split、ResNet18、HPT主干和30步horizon。仅将动作头改为确定性完整chunk回归：

`[B,30,256] -> LayerNorm -> Flatten -> Linear(256) -> GELU -> Linear(30*D) -> [B,30,D]`

Piper D=7，Aria左手D=6，使用独立动作头。监督为归一化动作MSE。A/B共享同样头结构及优化设置，不将FM loss数值与MLP MSE直接比较。

- A：63条机器人训练/7条验证，batch8。
- B：同样机器人数据，加45条人手训练/5条验证，每步机器人8+人手8，域loss平均。
- seed42，10000步，AdamW lr1e-4，cosine至1e-5，weight_decay1e-4，BF16。
- 原有图像增强与模型dropout保持，不沿用小样本诊断中关闭正则化的设置。
- 训练验证每100步进行10个batch，A监控 `Valid/action_loss`，B监控机器人域loss。正式完成后仍必须跑原7条完整验证，排除尾部padding且不重采样，报告mm/deg/夹爪误差。
- 原数据加载器的quantile越界重采样行为保持，以控制其他变量。它的警告不是训练崩溃，但仍需在正式评估中绕过。

MLP改变了头结构、参数量和损失；是明确标注的架构对照，不应宣称严格复现FM模型或唯一变量仅为某个实现bug。

## 检查与作业

本地A、B配置均完成10步训练和验证。新 `SaveFinalCheckpoint` 保存显式 `final.ckpt`，A本地文件内部global_step=10已验证，避免把历史last.ckpt误认为末步权重。

完整评估器已支持MLP并将辅助loss命名为 `masked_action_mse`；FM仍使用 `masked_fm_loss`。MLP确定性评估使用 `--seeds 0`，不会把重复推理解释成置信区间。10步模型用于检查评估链路，不能用于模型效果判断。

HPC代码快照 `/home/zekun/egoverse_piper/code_mlp_20260928`，复用已有ARM环境。两个作业均申请1张GH200、8CPU、32GB host RAM、4小时：

- A：3113376，日志 `/home/zekun/egoverse_piper/logs/a-mlp-3113376.log`。
- B：3113377，日志 `/home/zekun/egoverse_piper/logs/b-mlp-3113377.log`。

作业顺序：同batch/worker的1000步preflight -> 加载显式final恢复到1001步 -> **从头初始化**10000步正式训练。正式权重没有加载预检权重。num_workers=0沿用已验证的GH200规避fork/CUDA问题设置。

结果分别写入 `runs/a-mlp-3113376/{preflight,restore_check,train}` 和 `runs/b-mlp-3113377/{preflight,restore_check,train}`。当前作业状态应实时查询，本文不把提交或运行中当作训练完成。

本地作业索引 `.local_calibration/hpc/mlp_jobs.json`。

## 完成标准与后续

必须检查正式10000步完成、无致命错误、best/final实际global_step，再在完整7条验证上与FM及保持TCP基线比较。只有数据级误差与轨迹连续性确实改善，才能声称此路线改善了大误差问题。小样本记忆、本地smoke或HPC预检通过都不构成目标完成。机器人闭环执行另行安排，当前没有机器人运动操作。

## 正式完成后的评估命令

下载训练阶段选出的best checkpoint（不使用preflight或restore_check权重）到下面路径后：

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.evaluate_full \
 --a .local_calibration/hpc/mlp/checkpoints/a/best.ckpt \
 --b .local_calibration/hpc/mlp/checkpoints/b/best.ckpt \
 --dataset .local_calibration/piper_all \
 --split .local_calibration/aria_gen2_aligned/cotrain_data_check.json \
 --output .local_calibration/mlp_full_validation --seeds 0 --previews
OMP_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.diagnose_full \
 --evaluation .local_calibration/mlp_full_validation
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.compare_heads \
 --mlp .local_calibration/mlp_full_validation \
 --output .local_calibration/fm_mlp_comparison
```

`compare_heads`会逐元素核对两种head评估的目标、有效帧mask及frame_index完全一致，再比较物理误差和时间相邻位移。已用10步测试模型检查全量读取/比较链路，且FM相邻位移数值与此前独立诊断一致；这不是正式MLP效果结论。

## 中途诊断：A checkpoint step3900（不是正式终点）

从运行中作业当时选中的 `epoch_epoch=38.ckpt` 下载并读取内部global_step=3900，完整评估7条验证轨迹、2381帧。独立产物位于 `.local_calibration/mlp_interim_a/`，模型位于 `.local_calibration/hpc/mlp/interim/a_epoch38.ckpt`。

|指标|原FM A best（9100步）|中途MLP A（3900步）|
|---|---:|---:|
|第一步位置误差 mm|104.73|47.64|
|30步位置误差 mm|97.12|56.44|
|30步朝向误差 deg|18.44|12.94|
|30步夹爪误差 mm|15.35|6.30|
|相邻预测点位移 mm|66.19|22.00|

有明显改善，但第一步误差仍高于保持当前实际TCP基线的34.03mm，相邻位移仍高于示范约7.69mm。不能宣称问题解决。两者训练步数不同，此表仅为中途诊断，不代替正式MLP A/B终点评估，也不证明纯架构因果效应。原作业继续运行，未用此中途结果改变正在运行的配置。

## A完成与完整末步评估

Slurm 3113376已COMPLETED、exit0，正式训练10000步完成。`final.ckpt`内部global_step=10000；训练选中的best仍为epoch38/global_step3900，与中途下载副本SHA256完全一致（`2d4403ba0f4a25066a26a1bb1cae140da9639b828679b6f5e93c9dc614b01d17`）。

完整2381帧评估 `.local_calibration/mlp_final_a/report.json`：第一步位置30.93mm，30步位置49.63mm，30步朝向11.06deg，夹爪5.24mm。均优于3900步best的对应完整指标；第一步也略优于保持实际TCP基线34.03mm。

这揭示当前每次仅10个验证batch的选模协议不够代表完整验证集。训练监控best不等于完整验证最优，不能仅依据小子集指标未刷新就判定继续训练无效。A best与final分别保留，不将末步结果伪装为原监控选出的best。

## A/B 正式末步结果

A（3113376）和 B（3113377）均已完成10000步。完整验证集为相同7条轨迹、2381帧；目标、有效mask和帧索引已逐元素核对一致。结果见 `.local_calibration/mlp_final_validation/report.json`，与FM的对照图表见 `.local_calibration/fm_mlp_final_comparison/`。

|指标|FM A|MLP A|FM B|MLP B|
|---|---:|---:|---:|---:|
|第一步位置误差 mm|104.73|30.93|107.62|36.58|
|30步位置误差 mm|97.12|49.63|101.00|54.17|
|30步朝向误差 deg|18.44|11.06|20.57|12.61|
|30步夹爪误差 mm|15.35|5.24|14.32|4.95|
|相邻预测动作位移 mm|66.19|13.05|65.44|12.93|

FM采用原监控选出的9100/9800步权重，MLP此表采用预先设定的10000步终点；不是相同步数及选模协议的纯架构因果对照。MLP降低了误差及动作段内部跳动，但B位置误差仍高于A，且不能以离线误差替代实机成功率。每组只有一次训练，尚无跨训练种子的置信区间。

新增 `train_piper_a_mlp_fullval.yaml` / `train_piper_b_mlp_fullval.yaml` 配置使用 `FullPiperEval` 每1000步完整评估机器人验证集，以 `Valid/robot_chunk_position_mm` 选择权重，排除尾部填充且不重采样。已检查其结果与独立离线评估一致，并以实际Lightning单步训练确认checkpoint监控取到完整物理误差。此改动尚未用于上述正式训练。

## 状态基线和指令反馈审计

`check_state_baseline.py` 在63条训练轨迹拟合只输入当前7维状态的线性/二次岭回归，直接输出未来30步位置；固定正则系数1，不根据验证集调参。标准化仅来自训练集，各时域仅使用真实未来帧。验证目标和mask与MLP逐元素核对，结果见 `.local_calibration/state_baseline.json`：

|模型|第一步位置 mm|30步位置 mm|
|---|---:|---:|
|状态线性回归|32.11|93.27|
|状态二次回归|29.88|83.45|
|MLP A|30.93|49.63|

MLP的整段预测明显优于这两个简单状态基线，但第一步未超过二次回归。该对照同时改变了模型容量和图像输入，不能据此单独证明模型使用了视觉。

`check_visual_conditioning.py` 完成A/B全部7条轨迹的固定权重敏感性检查，保存于 `.local_calibration/visual_conditioning.json`。将图像替换为同轨迹相隔半段的图像，状态及目标不变；原图分支与保存的完整评估预测一致。整段位置误差：

|输入|A mm|B mm|
|---|---:|---:|
|原始图像|49.63|54.17|
|替换主视角|161.52|170.17|
|替换腕部视角|62.44|66.75|
|替换两路|170.76|178.87|

两组都明显依赖主视角，腕部也有影响。此测试引入图像与状态的不一致，证明条件敏感性而非严格量化视觉的独立收益；不能据此判定腕部可以删除。

`.local_calibration/command_limit_audit.json` 使用24787个真实帧（排除Zarr补齐），其中3277帧符合连续15帧各关节指令跨度小于0.5度。静止样本绝大多数在J2=-2.1度/J3=+2度指令附近，反馈为0度；离开零位附近的稳定样本仅3帧，误差约0.1–0.3度。因此现有证据不支持对全部数据统一增加J2/J3偏移；也不能仅凭这3帧认定内部全工作空间一致。转换公式与当前驱动发送公式一致，尚未修改标签。需要把限位附近执行差异与真正的校零偏移区分开。
