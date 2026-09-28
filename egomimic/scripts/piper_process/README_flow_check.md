# Flow Matching 时序跳动排查（2026-09-28）

最新进展见 [第三轮收敛对照](README_convergence.md)：原FM单片段平均误差已降至4.54mm；8片段仍为24.56mm，而直接回归能准确记忆同一组标签。位置编码尚不能认定为唯一根因。

## 第一轮结论

未发现训练/采样方向反转、动作 horizon 重排或完全缺失动作位置编码的问题。增加积分次数无改善。固定8个训练观察，只训练动作头1000次仍不能准确记住轨迹。配对实验中增强动作位置编码明显减少跳动，支持“当前动作头对 horizon 位置区分不足”这一线索，但没有解决全部误差，不能宣称已经找到唯一根因。

正式 A/B 模型、核心 FM 实现、训练配置和数据均未修改。GPU 实验使用本地4090，所有诊断权重保存在独立目录，不能直接部署。

## 实现核对

- `fm_policy.py`：训练插值 `x_t = t*noise + (1-t)*action`，监督速度 `noise-action`；采样从 t=1 向 t=0 积分，dt=-1/N。二者一致。
- 对实际 `FMPolicy.predict` 检验 `x_t - t*u_t == action`；对实际 `FMPolicy.inference` 输入已知常速度场，50次积分应恢复指定动作，误差小于1e-4。结果见 `flow_check/implementation_checks.json`。这只检验数学/实现方向，不证明学习出的速度场正确。
- `_robomimic_to_hpt_data` 和动作头 loss 路径保持 B×30×7 顺序；此前已逐元素核对评估 chunk 等于对应未来命令，并验证归一化往返与标准推理路径相同。
- `CrossTransformer` 具有30×64可学习 `pos_emb`，初始化为0。正式权重中 A 的标准差约0.00569、B约0.00610，均非零，因此并非位置参数完全未训练。小幅值本身不构成缺陷证明。
- 在8个样本的一次固定噪声/t=0.5探针上，反转动作 token 次序后再还原输出，速度变化RMS约0.027；打乱观察条件变化约0.880。前者说明当前头接近置换等变，提示时序位置利用较弱；不是通用精度指标。

## 训练样本隔离实验

A最佳模型（step9100），训练 episode `teddy_0912_1_episode_000`，帧 `[0,60,120,180,240,300,360,421]`，都具有完整30步标签。不使用验证样本做过拟合。

缓存原模型在eval模式下的视觉/主干输出，只更新6.52M参数的FM动作头。FP32、AdamW lr=1e-4、weight_decay=1e-4、梯度范数上限1、原有dropout，1000次更新，每次全部8样本；三个固定推理种子。

这有意简化了正式训练（主干冻结、无图像增强、固定LR、标准梯度裁剪），用于隔离动作头，**不是端到端训练检查，也不是泛化实验**。

|条件|位置误差 mm|相邻预测点距离 mm|
|---|---:|---:|
|正式A，50次积分|83.66|75.00|
|正式A，200次积分|84.32|76.38|
|原位置编码，配对训练1000次|63.42|54.99|
|添加正弦位置编码，配对训练1000次|53.17|26.95|
|示范自身|—|5.39|

两组配对训练都从相同A权重开始，训练随机种子2026，观察、目标、噪声、时间采样、dropout、优化器及更新次数一致。唯一干预是在训练开始前向动作头的可学习 `pos_emb` 加入标准正弦/余弦位置编码（30×64，频率尺度10000，幅值1），之后仍允许该参数更新。

该干预同时改变位置编码幅度和结构，不能进一步区分是幅度还是正弦结构起作用。位置误差下降约16.2%，相邻跳动下降约51.0%，但仍未准确拟合示范。不能据此认定正式B收益、校准准确性或新模型可部署。

初始探索跑 `flow_check/` 使用另一训练随机序列，1000步位置误差71.49mm，仅用于发现问题；不能拿它与正弦分支做严格配对比较。正式配对对照是 `flow_control/` 与 `flow_position/`。

## 产物与复现

- `.local_calibration/flow_check/implementation_checks.json`：实际FM数学检查。
- `.local_calibration/flow_control/`：原位置编码配对对照。
- `.local_calibration/flow_position/`：增强位置编码分支。
- 各目录 `report.json` 包含loss曲线、误差、帧号与协议；NPZ保存预测。
- `.local_calibration/flow_check/position_comparison.png`：两分支1000次更新后，相同帧180的轨迹，上下坐标范围一致。

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.check_flow \
 --position original --output .local_calibration/flow_control
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.check_flow \
 --position sinusoidal --output .local_calibration/flow_position
```

## 后续应如何判别

当前没有足够证据进行正式A/B重训。下一阶段先做单个动作chunk拟合（去掉多样本和条件区分因素），再做同8个样本的端到端拟合。分别检查确定目标是否可被FM头准确恢复，以及冻结主干特征是否限制条件区分；保持与原位置编码的配对对照。若训练集不能准确拟合，继续定位优化/时序结构；若能拟合再看验证集，才讨论数据量与泛化。不要用事后平滑掩盖预测错误。

## 第二轮：单片段与端到端拟合

最新结论修正：原位置编码也能学出与示范相似的时序趋势，不能将其认定为结构性缺陷或唯一根因。优化设置与每次更新的噪声样本数量明显影响结果；增加位置编码并非在所有隔离实验中都更好。目前仍未达到准确记忆训练样本的目标，正式A/B重训应暂缓。

全部分支从同一正式A权重重新开始，1000次更新，lr=1e-4，FP32，关闭dropout/数据增强、固定BN运行统计。单片段使用训练帧180；8片段使用第一轮相同的8帧。各配对分支训练随机种子均为2026，评估3个推理种子。关闭模型训练模式不会关闭autograd；full分支仍反传并更新整个策略。

|诊断任务|原编码位置误差 mm|增强编码位置误差 mm|原编码相邻点距离 mm|增强编码相邻点距离 mm|
|---|---:|---:|---:|---:|
|单片段，batch=1，仅动作头|126.43|42.54|56.52|42.18|
|单片段，batch=8重复同一观察，仅动作头|28.49|39.01|14.87|17.24|
|8个不同片段，batch=8，端到端|75.36|45.10|63.08|30.13|

单片段示范相邻点距离9.03mm；8片段平均5.39mm。重复观察的8份样本具有独立训练噪声和时间抽样，不是增加了8条不同示范。单片段batch=8评估也包含每个推理种子下8份独立噪声输出。batch=1与batch=8不是同等噪声抽样预算，不能把两者差别仅归因于batch本身的某个机制。

结果含义：

- 单片段batch=8原编码的误差从训练前89.28mm降到28.49mm，相邻跳动从89.99mm降到14.87mm。示例图已出现与示范一致的主要变化趋势，但仍有偏移和随机种子差异；不应宣称已经精准过拟合。
- 同任务中原编码优于增强编码，反驳“只要更换位置编码就能解决”的强结论。第一轮得到的位置编码线索仍有价值，但不唯一。
- 8片段端到端组确有梯度穿过encoders、stems、trunk、heads；各模块代表性权重与正式A权重的数值也确实发生变化，见 `.local_calibration/fullfit_weight_check.json`。因此不能用“实际没有更新主干”解释本轮结果。
- 解冻全网并没有使8片段快速精确拟合；但优化难度也增加，所以不能据此断言主干特征完全没有问题。
- 所有结果仍是训练样本诊断、单次训练随机种子。没有验证集提升或机器人成功率结论，未更新正式A/B。

产物：

- `.local_calibration/single_{original,sinusoidal}/`：单片段batch=1。
- `.local_calibration/single8_{original,sinusoidal}/`：单片段batch=8独立噪声。
- `.local_calibration/fullfit_{original,sinusoidal}/`：8片段端到端。
- `.local_calibration/flow_check/followup_summary.json`：汇总。
- `.local_calibration/flow_check/followup.png`：同一训练帧180的示例，上排单片段原编码，下排8片段端到端增强编码；上下任务不同，不是严格的单因素对照图。配对数字对照以上表为准。

复现示例（position可取original或sinusoidal，务必使用独立输出目录）：

```bash
source emimic/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.check_flow \
 --samples 1 --single-repeats 8 --scope head --steps 1000 --deterministic-fit \
 --position original --output .local_calibration/single8_original
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/piper-eval-mpl \
 python -m egomimic.scripts.piper_process.check_flow \
 --samples 8 --scope full --steps 1000 --deterministic-fit \
 --position original --output .local_calibration/fullfit_original
```

下一阶段应优先在单片段上使用更稳定的优化设置（多噪声抽样、学习率衰减、足够更新次数）达到明确的低位置误差，再扩到8片段。同时记录采样轨迹误差与连续性，不以随机FM loss单独作为通过条件。尚不建议根据当前结果直接改变正式模型结构。
