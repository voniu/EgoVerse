# Flow Matching 时序跳动排查（`check_flow.py`）

用于排查 Flow Matching（FM）动作头预测序列跳动的诊断工具：先核对 FM 实现的数学方向，再在少量训练样本上做隔离拟合实验。收敛性的进一步对照见 [README_convergence.md](README_convergence.md)。

所有实验只使用训练样本，诊断权重保存在独立输出目录，不能直接部署，也不修改正式模型、配置或数据。

## 实现核对

- `fm_policy.py`：训练插值 `x_t = t*noise + (1-t)*action`，监督速度 `noise-action`；采样从 t=1 向 t=0 积分，dt=-1/N。二者方向一致。
- 对实际 `FMPolicy.predict` 检验 `x_t - t*u_t == action`；对实际 `FMPolicy.inference` 输入已知常速度场，按积分步数应恢复指定动作（误差小于 1e-4）。结果写入 `report.json` 的 `implementation_checks` 字段。这只检验数学/实现方向，不证明学习出的速度场正确。
- `_robomimic_to_hpt_data` 与动作头 loss 路径保持 B×30×7 顺序。
- `CrossTransformer` 的动作位置编码为 30×64 可学习 `pos_emb`，初始化为 0；可检查训练后是否非零。
- 置换探针：在固定噪声和 t 上反转动作 token 次序后再还原输出，与打乱观察条件比较速度变化。前者很小说明当前头接近置换等变，时序位置利用较弱；这不是通用精度指标。

## 隔离拟合实验

缓存原模型在 eval 模式下的视觉/主干输出，在固定训练观察上训练 FM：

| 选项 | 含义 |
|---|---|
| `--samples 1\|8` | 单个或 8 个固定训练观察（都具有完整 30 步标签） |
| `--single-repeats N` | 单观察时每次更新的独立噪声抽样份数 |
| `--scope head\|full` | 只训练动作头，或端到端反传整个策略 |
| `--position original\|sinusoidal` | 保持原可学习位置编码，或在训练前向 `pos_emb` 加入标准正弦/余弦编码（之后仍可更新） |
| `--deterministic-fit` | 关闭 dropout、数据增强并固定 BN 运行统计；梯度仍开启 |
| `--steps` | 更新次数 |

配对分支（original 与 sinusoidal）使用相同的起始权重、观察、目标、噪声、时间抽样、优化器和更新次数，唯一干预是位置编码。正弦干预同时改变位置编码的幅度和结构，不能区分二者的作用。`--single-repeats` 下的多份样本只是独立训练噪声，不是更多示范；batch=1 与 batch=N 的噪声抽样预算不同。

这些实验有意简化正式训练（固定 LR、冻结主干或固定 BN 等），用于隔离动作头，**不是端到端训练检查，也不是泛化实验**。源 checkpoint 和训练 episode 在脚本中设定，运行前请核对。

## 运行与产物

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.check_flow \
 --checkpoint <checkpoint> --dataset <piper_zarr_dir> --valid-report <valid_report.json> \
 --position original --output <out_control>
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.check_flow \
 --checkpoint <checkpoint> --dataset <piper_zarr_dir> --valid-report <valid_report.json> \
 --position sinusoidal --output <out_position>
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.check_flow \
 --checkpoint <checkpoint> --dataset <piper_zarr_dir> --valid-report <valid_report.json> \
 --samples 1 --single-repeats 8 --scope head --steps 1000 --deterministic-fit \
 --position original --output <out_single8>
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.check_flow \
 --checkpoint <checkpoint> --dataset <piper_zarr_dir> --valid-report <valid_report.json> \
 --samples 8 --scope full --steps 1000 --deterministic-fit \
 --position original --output <out_fullfit>
```

每个分支务必使用独立输出目录。各目录 `report.json` 包含 loss 曲线、位置误差、相邻预测点距离、帧号、协议与实现核对结果；NPZ 保存预测，`overfit.png` 显示示例轨迹。

## 判别顺序

先在单个动作 chunk 上用稳定的优化设置（多噪声抽样、学习率衰减、足够更新次数）达到明确的低位置误差，再扩到多个片段和端到端拟合；训练集不能准确拟合时继续定位优化/时序结构，能拟合后再看验证集。同时记录采样轨迹误差与连续性，不以随机 FM loss 单独作为通过条件，也不要用事后平滑掩盖预测错误。
