# 第三轮：FM 收敛与直接回归对照（2026-09-28）

## 结论

**原始 FM 动作头与原位置编码能够拟合单个动作片段，平均位置误差达到4.54mm；但8个观察的条件拟合仍未通过5mm目标。** 同一组冻结特征和动作标签，直接回归可以把8个片段记忆到接近数值精度。

因此，目前主要线索是FM动作头的条件学习和优化，而不是“模型完全学不会时序”。尚未证明具体实现bug，也未证明位置编码是唯一原因。直接回归改变了网络结构、目标和优化超参数，不能将结果解读成严格同构的算法优劣比较。

所有实验为训练样本记忆检查，没有运行正式模型的改版验证或机器人执行，不能代表物理标定精度、泛化能力、抓取成功率或B优于A。没有更改正式A/B配置、checkpoint和数据。

## 数据与共享输入

正式A checkpoint step9100；训练episode `teddy_0912_1_episode_000`。

- 单片段：frame180，完整30步标签。
- 8片段：frame `[0,60,120,180,240,300,360,421]`，均有完整30步标签。
- 输入使用原A在eval模式下产生的冻结HPT特征，每个观察为30×256。
- 输出为30×7归一化动作，计算误差前用原训练normalizer还原。
- FM和回归共享同一份特征与标签；目标逐元素相等，归一化往返误差已检查。
- 整个实验都没有更新视觉编码器/主干。没有图像增强或dropout；FP32。

`inputs.pt` 保存特征、归一化动作、原始标签和帧号，便于进一步隔离检查。

## 实验配置

### 直接回归

LayerNorm -> flatten(30×256) -> Linear(256) -> GELU -> Linear(210)，reshape为30×7。随机初始化，使用相同冻结特征，直接最小化归一化动作MSE。AdamW lr=3e-4，weight_decay=0，cosine衰减到1e-6，2000次更新，梯度范数上限1。单片段batch1，8片段batch8。

这是一个有独立每步输出维度的确定性MLP；其种子间采样差异为0是结构本身决定的，不能用来证明优于生成策略。单片段能被回归记忆本身也不证明特征质量；8片段结果至少说明这组特征包含足以区分这8个目标的信息。

### 单片段 FM

1. 从正式A动作头开始，原可学习位置编码保持其结构不变；每步16份独立噪声/时间抽样。
2. AdamW lr=3e-5 cosine至1e-6，weight_decay=1e-4，5000次更新，梯度范数上限1。
3. 均值误差降至5.93mm后，加载该诊断动作头，重新创建优化器/scheduler，每步32份噪声，lr=1e-5 cosine至1e-6，再2000次更新。
4. 细化是加载权重并重置优化器，不是无缝恢复原优化器。总共7000次更新。

### 8片段 FM

从正式A动作头重新开始，**没有使用单片段细化权重**。每步8观察×4噪声，共batch32；lr=3e-5 cosine至1e-6，5000次更新，其他设置相同。目标仍为原FM速度目标，推理仍是50次积分。

## 结果

下表FM采用额外推理种子10–19，不是10次独立训练；回归为确定性输出。指标平均所有所选观察、30个horizon和推理种子。P95也在这些点上计算。

|任务与方法|位置均值 mm|位置P95 mm|旋转误差 deg|夹爪误差 mm|相邻点位移 mm|
|---|---:|---:|---:|---:|---:|
|单片段FM，5000步|5.93|10.20|1.36|0.98|10.21|
|单片段FM，细化后|**4.54**|9.17|1.16|0.83|9.65|
|单片段直接回归|<0.001|<0.001|<0.001|<0.001|9.03|
|8片段FM，5000步|**24.56**|78.46|4.59|2.25|24.40|
|8片段直接回归|<0.001|<0.001|<0.001|<0.001|5.39|

单片段示范相邻点位移9.03mm，8片段平均5.39mm。单片段FM预测相对各seed平均轨迹的平均距离为4.44mm，8片段为18.37mm；这是采样离散程度，不是置信区间。单片段残差尚非0，不能理解成所有预测点误差都低于5mm。

8片段FM按anchor分解，平均误差：

|frame|0|60|120|180|240|300|360|421|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|mm|12.46|10.62|14.96|31.22|63.09|13.75|38.49|11.87|

每个片段都尚未达到5mm，尤其240与360较差。frame180单独训练可拟合，在多片段任务中误差变大，支持继续检查条件区分、多任务优化和残余采样噪声的方向；不能仅凭这些数字确定具体机制。

由于8片段拟合没有通过，本轮未继续正式验证集评估。少量片段过拟合成功也不保证验证集结果；此前完整A/B验证结果仍然有效，没有被这些训练集数字替代。

## 复现

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 python -m egomimic.scripts.piper_process.check_convergence \
 --samples 1 --steps 5000 --output .local_calibration/convergence_single
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 python -m egomimic.scripts.piper_process.check_convergence \
 --samples 1 --steps 2000 --lr 1e-5 --noise-batch 32 \
 --resume-fm .local_calibration/convergence_single/fm_checkpoint.pt \
 --output .local_calibration/convergence_single_refine
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 python -m egomimic.scripts.piper_process.check_convergence \
 --samples 8 --steps 5000 --lr 3e-5 --noise-batch 32 \
 --output .local_calibration/convergence_eight
```

各输出目录包含report.json、阶段预测NPZ、regression.pt、fm_checkpoint.pt（包含动作头/优化器/scheduler状态）、inputs.pt。细化脚本会重新训练同一直接回归对照，但不会将回归结果用于FM训练。

图像：

- `.local_calibration/convergence_single_refine/comparison.png`：单片段原预测、细化后预测、直接回归；展示seed0–2，各行共享坐标范围。
- `.local_calibration/convergence_eight/per_observation.png`：8片段FM误差分解。

ruff及git diff --check通过；三个GPU实验均正常退出。后续优先对高误差片段做条件替换/打乱检查，以及残余噪声随FM时间的误差分析，保持直接回归作为可拟合性参照；暂不直接修改正式模型结构或重训完整A/B。
