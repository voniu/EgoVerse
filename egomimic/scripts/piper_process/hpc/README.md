# Arrhenius 上的 Piper A 实验

## 调试资源与后续检查流程

2026-09-28 实际查询 `sinfo -a`、`scontrol show partition`、`sacctmgr show qos`
及当前账户关联：仅见 `cpu`、`gpu`、`fat` 三个分区，QoS 仅见 `normal`，
当前 GPU 账户允许/默认 QoS 均为 `normal`。没有查到可用的独立 debug 分区或 debug QoS。

官方提供 `interactive`（salloc+srun 包装），可在短时 GPU allocation 内反复调试，
无需每次修改都重新申请资源。它仍走普通队列，不保证立即分配：

```bash
ssh arrhenius
interactive -A naiss2026-3-376-gpu -p gpu --gpus=1 -n 1 -c 8 --mem=32G -t 00:30:00
```

进入计算节点后再加载 GPU 模块、GCC 运行库路径和现有 `emimic` 环境。
参考 [官方构建与交互调试说明](https://hpc.pages.naiss.se/user-documentation/support-docs/arrhenius_hpc/software_development/building_applications/)。

后续新环境或数据读取配置变更，先在此短时 allocation 内检查：

1. 项目/模型导入、GPU 前向及反向传播。
2. 与正式训练相同 batch、worker 配置，至少覆盖 1,000 步和多次训练/验证切换，
   检查 checkpoint 保存与恢复。10 步单 epoch smoke 不能覆盖本次第 6–7 轮才出现的问题。
3. 通过后才提交长时间正式训练；正式实验从头启动，或明确从原实验 checkpoint 恢复。

该检查流程提高覆盖范围，不代表能保证长时间任务永不失败。

## 后续故障检查：DataLoader / CUDA

原作业 B 3110040、A 3110041 后续均 FAILED（exit 1）。A 最后完成第 600 步，
B 最后完成第 700 步；最后日志损失有限。Slurm 记录的是 FAILED，不是 OUT_OF_MEMORY。
worker 原始异常为 `CUDA error: initialization error`，上层才是
`DataLoader worker exited unexpectedly`。在 epoch 切换后 worker 重建时出现，
与 Python 3.13/Linux 默认 fork 继承 CUDA 状态的限制相符；这是根因推断，
并非已定位到某个具体张量的触发点。

两组同时采用运行参数 `num_workers=0`，避免此 worker 路径；不改模型、batch、
归一化、损失权重或总步数。会影响吞吐和数据读取随机数消耗，因此不声称逐位复现
不中断的原始运行。参见 [PyTorch multiprocessing](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html)。

- B 恢复作业：**3110128**；A 恢复作业：**3110129**。
- 恢复约 3 分钟后，确认 A 指标已记录到 step=1000、B 到 step=900，
  均越过原失败位置并完成多轮验证，Slurm 为 RUNNING。
  恢复日志未再出现 CUDA initialization error、worker crash、OOM 或 NaN/Inf。
  仍有原有的归一化边界重采样和梯度裁剪提示，未在本次恢复中改变这些策略。
- 从各自原运行 `train/checkpoints/last.ckpt` 恢复，保留模型、优化器、调度器和 global step。
- 沿用 `runs/b-3110040/train`、`runs/a-3110041/train`，使最佳 checkpoint 的历史比较连续。
- 新日志：`logs/b-resume-3110128.log`、`logs/a-resume-3110129.log`。
- `train_ab.sbatch` 的第二参数可指定原运行目录；有此参数时跳过 smoke，执行恢复。

```bash
ssh arrhenius 'squeue -j 3110128,3110129'
ssh arrhenius 'tail -n 30 /home/zekun/egoverse_piper/logs/b-resume-3110128.log'
```

## 2026-09-28：B 提交与 A 重提

- 正式 B 作业：**3110040**，A 重提作业：**3110041**。
- 18:46（集群日志时间）已确认两组 `RUNNING`，各自 10 步 smoke 完成，
  已进入正式训练。B smoke 的机器人/人类验证损失均有限；正式训练尚未完成。
- 原 A 作业 3104384 在环境导入阶段失败（3 分 15 秒），没有开始训练。
  SciPy 需要 `GLIBCXX_3.4.32`，但 flat GPU Python 模块没有带上 GCC 运行库路径。
- 修复：在作业内将 GCCcore 14.3.0 的 `lib64` 加入 `LD_LIBRARY_PATH`；
  随后发现间接导入需要 `scaleapi`，补齐 2.18.4。GPU 检查作业 3110027
  已输出 `ENV_CHECK_PASSED`，PyTorch 2.7.1+cu128、GH200 120 GB，CUDA 反向传播通过。
- 新入口为 `train_ab.sbatch a|b`，两组复用同一个已修复环境；各自先运行
  10 步 smoke，通过后才从头训练 10,000 步。正式训练不会加载 smoke checkpoint。
- A 保持原始 `code/` 快照，B 为独立 `code_b_20260928/` 快照。
- B 增加 `data/aria_left_aligned.tar`（50 条、8,383 帧，约 855 MiB），
  `assets/b_norm_stats.json`、`assets/b_split_check.json`。沿用 A 的机器人归档和 ResNet 权重。
- 数据解包至各作业的 `/scratch/local/piper-<a|b>-<jobid>/`，不在 home 重复展开数据。
- B 正式配置每步 robot=8、human=8，等权平均 loss；按机器人验证 loss 选模。
  两组均为 1 GPU、8 CPU、32 GB host RAM、4 小时 walltime。
- 日志：`logs/b-3110040.log`、`logs/a-3110041.log`；结果目录：
  `runs/b-3110040/{smoke,train}`、`runs/a-3110041/{smoke,train}`。
- 上传后逐项核对 SHA256 一致：
  - 人类归档：`a1b3ccd4736c36d1c8592d02876b5511f7ae9e7186707806dfb976dc270323ac`
  - B 代码归档：`025c37b7469550539dbd8ce4f5fbb39b51c6a9cac38432cf29bb47b1e1ea6baa`
  - 双域统计：`0b6b8f49b13d856a4dc5f491eaa2d21f8659e958ed1e36499876028dd178b61f`

```bash
ssh arrhenius 'squeue -j 3110040,3110041'
ssh arrhenius 'tail -n 40 /home/zekun/egoverse_piper/logs/b-3110040.log'
ssh arrhenius 'sacct -j 3110040,3110041 --format=JobID,State,Elapsed,ExitCode'
```

以下保留首次 A 提交时的记录；其“未验证”状态已被上述故障与修复记录替代。

2026-09-28 提交作业 **3104384**。提交时为 Pending，尚未验证集群环境安装、
GPU smoke 或正式训练完成。实际状态以 `squeue` / `sacct` 为准。

## 固定实验设置

- 70 条转换后的 Zarr，固定 63/7 episode 划分；22,406/2,381 帧。
- A：HPT + flow matching，batch=8，10,000 更新，seed=42，BF16 mixed。
- 机器人参数与本地 `train_piper.yaml` 一致；不会从 smoke 权重继续训练。
- 训练集归一化统计和 ImageNet ResNet18 权重从本地已验证运行复制。
- GPU：1 × GH200 120 GB；8 CPU；32 GB host memory；4 小时上限。
- 10 步 smoke 通过后才进入正式训练；失败时停止，不启动正式训练。

## 远程路径

根目录：`/home/zekun/egoverse_piper`

- `code/`：本次代码快照，包含未提交的 Piper 适配。
- `code/emimic/`：GPU 节点上建立的 ARM 虚拟环境。
- `data/piper_zarr.tar`：约 2.7 GB，70 条 Zarr，不包含投影视频。
- `assets/`：权重、归一化统计、原始 split 检查记录和归档 SHA256。
- `logs/a-3104384.log`：安装、smoke、训练日志。
- `logs/environment.txt`、`modules.txt`：安装成功后记录实际依赖与模块。
- `runs/a-3104384/train/`：正式模型、配置和 CSV 指标。
- `runs/a-3104384/train/checkpoints/`：best 与 last checkpoint。

使用 `GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb`
模块，复用科学计算依赖；额外安装 PyTorch 2.7.1 CUDA 12.8 ARM wheel
及本目录 requirements。登录节点为 x86_64，不能在那里执行 GPU 模块的 Python。
因此只在登录节点做了跨平台 wheel 可用性解析，实际安装和导入检查由作业执行。

个人目录临时保存输入归档、环境和结果；不向已超配额的共享目录写入。
实际训练数据在 `/scratch/local/piper-a-<jobid>/data` 解包，缓存和安装临时文件
也使用计算节点本地磁盘。后续应将持久数据迁回有可用配额的项目存储。
参见 [Arrhenius 存储文档](https://hpc.pages.naiss.se/user-documentation/support-docs/arrhenius_hpc/data_management/storage/)。

## 查看状态

```bash
ssh arrhenius 'squeue -j 3104384'
ssh arrhenius 'tail -n 50 /home/zekun/egoverse_piper/logs/a-3104384.log'
ssh arrhenius 'sacct -j 3104384 --format=JobID,State,Elapsed,ExitCode'
```

作业开始前日志文件可能尚未创建。SSH ControlPersist 到期不会取消已提交的作业。
如果作业超时，已有 last checkpoint 可用于恢复，不能将超时当作完成 10,000 步。

## 再次提交

已有环境时可直接提交 `train.sbatch`；首次用 `run.sbatch` 串行执行 setup 和 train。
在远程 `code/` 目录运行：

```bash
sbatch -A naiss2026-3-376-gpu \
  -o /home/zekun/egoverse_piper/logs/a-%j.log \
  egomimic/scripts/piper_process/hpc/run.sbatch
```

资源和作业脚本遵循 [Arrhenius 作业文档](https://hpc.pages.naiss.se/user-documentation/support-docs/arrhenius_hpc/job_management/scripts/)。
本次仅做 Piper A；Aria SDK 不用于已对齐相机坐标系的 Piper 训练，
其导入已推迟至实际需要它的变换函数，数学逻辑未改动。
