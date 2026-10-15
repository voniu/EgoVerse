# FM 收敛与直接回归对照（`check_convergence.py`）

在少量训练样本上检查 Flow Matching（FM）动作头能否记忆动作片段，并用同一组冻结特征上的直接回归作为可拟合性参照。这是训练样本记忆检查，不代表泛化能力、物理标定精度或实机成功率，也不修改正式配置、checkpoint 或数据。

## 共享输入

- 从一个已训练 checkpoint 和一条训练 episode 中选取 1 个或 8 个具有完整 30 步标签的观察（`--samples 1|8`）。
- 输入为原模型在 eval 模式下产生的冻结 HPT 特征，每个观察 30×256；输出为 30×7 归一化动作，计算误差前用原训练 normalizer 还原。
- FM 与回归共享同一份特征与标签。不更新视觉编码器/主干，无图像增强或 dropout，FP32。

## 两个分支

**直接回归**：LayerNorm -> flatten(30×256) -> Linear(256) -> GELU -> Linear(210)，reshape 为 30×7。随机初始化，最小化归一化动作 MSE。它是确定性 MLP，种子间采样差异为 0 是结构决定的，不能用来证明优于生成策略；单片段能被回归记忆本身也不证明特征质量。

**FM**：从 checkpoint 的动作头开始，保留原可学习位置编码，每步多份独立噪声/时间抽样（`--noise-batch`），AdamW + cosine，梯度范数上限 1。推理仍为原积分步数。`--resume-fm` 加载此前诊断得到的动作头权重并重建优化器与 scheduler（不是无缝恢复）。

## 运行

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 python -m egomimic.scripts.piper_process.check_convergence \
 --checkpoint <checkpoint> --episode <train_episode.zarr> --valid-report <valid_report.json> \
 --samples 1 --steps 5000 --output <out_single>
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 python -m egomimic.scripts.piper_process.check_convergence \
 --checkpoint <checkpoint> --episode <train_episode.zarr> --valid-report <valid_report.json> \
 --samples 1 --steps 2000 --lr 1e-5 --noise-batch 32 \
 --resume-fm <out_single>/fm_checkpoint.pt --output <out_single_refine>
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 python -m egomimic.scripts.piper_process.check_convergence \
 --checkpoint <checkpoint> --episode <train_episode.zarr> --valid-report <valid_report.json> \
 --samples 8 --steps 5000 --lr 3e-5 --noise-batch 32 --output <out_eight>
```

输入 checkpoint 和 episode 在脚本中设定，运行前请核对。

## 输出与解读

各输出目录包含 `report.json`、阶段预测 NPZ、`regression.pt`、`fm_checkpoint.pt`（动作头/优化器/scheduler 状态）、`inputs.pt`（特征、归一化动作、原始标签和帧号）；单片段细化会生成 `comparison.png`，8 片段生成按观察分解误差的 `per_observation.png`。

指标对所选观察、30 个 horizon 和推理种子取平均，并给出 P95、旋转、夹爪及相邻点位移。FM 的多个推理种子不是独立训练，种子间离散程度不是置信区间。少量片段过拟合成功也不保证验证集结果。直接回归改变了网络结构、目标和优化超参数，两者不是严格同构的算法比较。
