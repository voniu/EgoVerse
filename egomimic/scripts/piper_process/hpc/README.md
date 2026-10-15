# Piper HPC 启动脚本

用于在 Slurm 集群（为 Arrhenius 的 GH200/aarch64 节点编写）上训练和评估 Piper Residual Joint
Transformer。训练流程、数据格式与配置说明见
[docs/piper_residual_joint_transformer_training.md](../../../../docs/piper_residual_joint_transformer_training.md)。
所有命令从仓库根目录提交；`<account>`、`<partition>` 及路径均为占位符。

更早的启动脚本（`train_ab`、`train_mlp`、`run`、`setup` 等）已删除，可在 git tag
`piper-assembly-d435-d405-v1` 中查看。

## 文件

| 文件 | 作用 |
|---|---|
| `train_residual_joint_transformer.sbatch` | A/B 训练：短 smoke 后正式训练 |
| `openloop_bias.sbatch` | 单个 checkpoint 的开环首步偏差评估 |
| `bootstrap_arrhenius.sbatch` | 在 GPU 节点上从锁文件构建并验证训练环境 |
| `arrhenius_env.sh` | 激活上述环境，供 `PIPER_ENV_SETUP` 引用（source，不要执行） |
| `requirements-training-aarch64.lock` | aarch64 训练依赖的哈希锁文件（不含仅用于数据转换的 projectaria-tools） |

## 环境：`bootstrap_arrhenius.sbatch` 与 `arrhenius_env.sh`

```bash
sbatch -A <account> -p <partition> --export=ALL,RUN_DIR=<dir under $HPC_RUNS> \
  egomimic/scripts/piper_process/hpc/bootstrap_arrhenius.sbatch
```

依赖 `~/.config/hpc/env.sh` 提供 `HPC_ROOT`、`HPC_ENVS`、`HPC_RUNS`（集群约定）及
`$HPC_ROOT/tools/aarch64/bin/uv`。脚本在 aarch64 GPU 节点上用 uv 创建 Python 3.11 的
relocatable venv，按锁文件 `--require-hashes` 同步依赖，做 CUDA 前向/反向与 torchvision 检查，
运行 Piper 相关单元测试，通过后移动到 `$HPC_ENVS/egoverse-piper/aarch64/<profile>`（已存在则拒绝覆盖）。
`RUN_DIR` 必须是 `$HPC_RUNS` 下已存在的目录，脚本在其中记录 commit、diff、模块、uv/Python 版本和依赖 freeze。

`arrhenius_env.sh` 按 `PIPER_ENV_PROFILE`（默认与 bootstrap 相同的 profile）设置 `VIRTUAL_ENV`/`PATH`
并校验解释器，默认 `PIPER_SCRATCH_ROOT=/scratch/local`。

## 训练：`train_residual_joint_transformer.sbatch`

```bash
sbatch -A <account> -p <partition> \
  --export=ALL,PIPER_ENV_SETUP=$PWD/egomimic/scripts/piper_process/hpc/arrhenius_env.sh,PIPER_WORK_ROOT=<work_root> \
  egomimic/scripts/piper_process/hpc/train_residual_joint_transformer.sbatch a [hydra overrides...]
```

第一个参数为 `a` 或 `b`（对应 `train_piper_{a,b}_residual_joint_transformer`），其余参数作为 Hydra
覆盖项追加。作业先做 20 步 smoke（`hydra.run.dir=$RUN/smoke`），再复用其归一化统计从头开始正式训练
（`$RUN/train`）。默认 `num_workers=0`。

| 变量 | 默认 | 含义 |
|---|---|---|
| `PIPER_CODE` | 仓库根目录 | 代码目录 |
| `PIPER_ENV_SETUP` / `PIPER_VENV` | `$PIPER_CODE/emimic` | 要 source 的环境脚本，或 venv 目录 |
| `PIPER_WORK_ROOT` | `$PIPER_CODE/.local_calibration/hpc_residual_joint_transformer` | 工作根目录 |
| `PIPER_DATA_ARCHIVE` | `$PIPER_WORK_ROOT/data/piper_zarr.tar` | Piper Zarr 归档，解包到节点 scratch |
| `ARIA_DATA_ARCHIVE` | `$PIPER_WORK_ROOT/data/aria_left_aligned.tar` | B 组 Aria 左手归档 |
| `PIPER_DATA_DIR` / `ARIA_DATA_DIR` | 未设置 | 已解包的只读 episode 目录；设置后跳过解包 |
| `PIPER_RUN_DIR` | `$PIPER_WORK_ROOT/runs/residual-joint-transformer-<group>-<jobid>` | 运行目录 |
| `PIPER_SMOOTH` | `0` | 为 `1` 时对解包副本运行 `smooth_robot_actions`（不能与 `PIPER_DATA_DIR` 同用） |
| `PIPER_LEDGER` | `$PIPER_WORK_ROOT/ledger.jsonl` | 每个作业追加一行 JSON（commit、覆盖项、最后验证损失、checkpoint 列表） |
| `PIPER_NOTIFY` | `~/.config/hpc/notify.sh` | 通知脚本（资源分配、开始训练、失败/终止、结束）；不存在则跳过 |
| `SLURM_TMPDIR` / `PIPER_SCRATCH_ROOT` | `/tmp` | 节点临时目录，作业结束时清理 |
| `TORCH_HOME` | `$PIPER_WORK_ROOT/assets` | ImageNet 权重缓存 |

## 开环偏差：`openloop_bias.sbatch`

```bash
sbatch -A <account> -p <partition> \
  --export=ALL,PIPER_CODE=<code>,PIPER_ENV_SETUP=<env.sh>,PIPER_DATA_DIR=<piper_zarr_dir> \
  egomimic/scripts/piper_process/hpc/openloop_bias.sbatch <checkpoint> <output_dir>
```

对验证集和部分训练集 episode 运行 `egomimic.scripts.piper_process.openloop_bias`，输出
`<output_dir>/openloop_bias_{valid,train}.json`。episode 索引写在脚本中，对应特定数据划分，换数据集时需修改。
