# MLP 动作头 A/B 对照

用确定性 chunk MLP 替换 Flow Matching（FM）动作头的架构对照，以及配套的评估与诊断工具。

## 模型与公平性

保留原 A/B 的相机、TCP 标签、归一化、split、ResNet18、HPT 主干和 30 步 horizon。仅将动作头改为确定性完整 chunk 回归：

`[B,30,256] -> LayerNorm -> Flatten -> Linear(256) -> GELU -> Linear(30*D) -> [B,30,D]`

Piper D=7，Aria 左手 D=6，使用独立动作头。监督为归一化动作 MSE。A/B 共享同样头结构及优化设置；不要将 FM loss 数值与 MLP MSE 直接比较。

- A：仅机器人数据。
- B：同样机器人数据，加人手数据；每步一个机器人 batch 加一个人手 batch，域 loss 平均。
- 其余超参数（seed、步数、AdamW、cosine、BF16、图像增强与 dropout）与 FM 版本保持一致，见 `train_piper_{a,b}_mlp.yaml`。

MLP 改变了头结构、参数量和损失，是明确标注的架构对照，不应宣称严格复现 FM 模型。

训练使用 `SaveFinalCheckpoint` 显式保存 `final.ckpt`，避免把历史 `last.ckpt` 误认为末步权重。评估器支持 MLP，并将辅助 loss 命名为 `masked_action_mse`（FM 为 `masked_fm_loss`）。MLP 为确定性模型，评估时使用 `--seeds 0`。

### 完整验证集选模

训练中每次仅评估少量验证 batch 的选模协议不一定代表完整验证集。`train_piper_a_mlp_fullval.yaml` / `train_piper_b_mlp_fullval.yaml` 使用 `FullPiperEval` 定期完整评估机器人验证集，以 `Valid/robot_chunk_position_mm` 选择权重，排除尾部填充且不重采样。

## 评估命令

使用训练阶段选出的 checkpoint（不使用 preflight 或恢复检查的权重）：

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.evaluate_full \
 --a <a_checkpoint> --b <b_checkpoint> \
 --dataset <piper_zarr_dir> --split <split.json> \
 --output <mlp_eval_dir> --seeds 0 --previews
OMP_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.diagnose_full \
 --evaluation <mlp_eval_dir> --dataset <piper_zarr_dir>
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.compare_heads \
 --flow <fm_eval_dir> --mlp <mlp_eval_dir> --output <comparison_dir>
```

`compare_heads` 会逐元素核对两种 head 评估的目标、有效帧 mask 及 frame_index 完全一致，再比较物理误差和时间相邻位移。

## 辅助检查

- `check_state_baseline.py`：在训练轨迹上拟合只输入当前 7 维状态的线性/二次岭回归（固定正则系数，不按验证集调参；标准化仅来自训练集），直接输出未来 30 步位置，作为不使用图像的简单基线。验证目标和 mask 与模型评估逐元素核对。该对照同时改变模型容量和图像输入，不能单独证明模型使用了视觉。
- `check_visual_conditioning.py`：固定权重的视觉敏感性检查。将主视角、腕部视角或两路图像替换为同一轨迹中相隔半段的图像，状态及目标不变，比较整段位置误差。此测试引入图像与状态不一致，衡量的是条件敏感性，不是视觉的独立收益。

离线误差不能替代实机成功率；每组只有一次训练时也没有跨训练种子的置信区间。
