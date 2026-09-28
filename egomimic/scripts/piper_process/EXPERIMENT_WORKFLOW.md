# Piper 实验版本与结果回溯

每个会产生 checkpoint 或真机日志的实验都必须能够回答三件事：运行了哪份代码、
使用了哪份完整配置、产物位于哪里。执行规则如下。

1. 一项独立假设使用一个 `codex/` 分支。数据转换、训练架构、离线评估、真机部署
   分成可独立检出的提交；不要把 checkpoint、视频或 `.local_calibration` 加入 Git。
2. 提交训练前确保 `git status --short` 为空，并记录 `git rev-parse HEAD`。若必须从脏
   工作区做临时诊断，结果只能标为 diagnostic，不能作为 A/B 正式结论。
3. `trainHydra` 会在每个 Hydra 运行目录写入 `run_provenance.json`，包含 commit、分支、
   dirty 状态、启动参数和 Slurm job id。Hydra 的 `.hydra/config.yaml` 是该作业解析后的
   完整配置。这两个文件应和 `checkpoints/`、指标文件一起保存。
4. A/B 或 FM/MLP 对比从同一个基准 commit 建分支，除被检验因素外保持数据划分、seed、
   batch、训练步数和选模指标一致。结果表记录 commit、配置名、job id、best checkpoint
   路径及 checkpoint SHA256。
5. 真机 rollout 的输出目录包含 `config.json`；部署入口自动写入源码 provenance、
   checkpoint step 和 SHA256。一次 rollout 使用一个新目录，禁止覆盖旧日志。

推荐命名：

```text
branch: codex/piper-<hypothesis>
commit: feat(piper): <one reviewable change>
run:    <group>-<head>-<git-short-sha>-<slurm-job-id>
```

回溯一个异常结果时，先读取运行目录的 `run_provenance.json` 和 `.hydra/config.yaml`，
检出其中 commit，再用记录的 checkpoint SHA256 核对权重文件。只有源码为 clean 且
哈希一致时，才把复现实验与原结果视为同一版本。
