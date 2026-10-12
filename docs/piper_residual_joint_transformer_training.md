# Piper Residual Joint Transformer training guide

This guide is the reproducible entry point for the Piper experiments in this
fork. It covers the current A/B comparison:

- **A** trains on Piper demonstrations only.
- **B** uses the same Piper objective and batches, plus aligned left-hand Aria
  demonstrations through the shared 9D pose head.
- Both variants deploy the deterministic Piper residual joint head. The shared
  flow-matching pose head supplies geometric supervision and is not sent to the
  robot.

See [piper_model_architecture_evolution.md](piper_model_architecture_evolution.md)
for the model and loss definitions.

Checkpoints produced before this strategy-based rename remain loadable through
small compatibility import modules. New experiments and documentation use only
the `residual_joint_transformer` names.

## 1. Environment

Run commands from the repository root with Python 3.11:

```bash
uv venv emimic --python 3.11
source emimic/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

Training needs a CUDA GPU. Dataset conversion and configuration checks can run
on CPU. The first run may download ImageNet ResNet18 weights; set `TORCH_HOME`
to a persistent cache when compute nodes have temporary disks.

## 2. Data contract

Training reads a directory containing one EgoVerse Zarr store per episode. Do
not commit datasets, checkpoints, camera calibration, or rollout logs; the
repository ignores `.zarr`, `.local_calibration`, and common model artifacts.

The Piper Residual Joint Transformer loader expects these logical fields:

| Field | Shape used by the model | Unit / meaning |
|---|---:|---|
| front RGB | `[3,480,640]` | D435IF image |
| wrist RGB | `[3,480,640]` | D405 image |
| joint state | `[7]` | six radians plus gripper opening |
| Cartesian action | `[30,9]` | xyz plus rotation-6D |
| joint action | `[30,7]` | six radians plus binary gripper |

`lerobot_to_zarr.py` converts a Piper LeRobot v3 episode. Its required
calibration files are specific to the robot and cameras used during collection:

```bash
python -m egomimic.scripts.piper_process.lerobot_to_zarr \
  --dataset /data/lerobot/task_01 \
  --episode 0 \
  --output /data/piper_zarr \
  --action-calibration /calibration/piper_follower.json \
  --wrist-intrinsics /calibration/d405.json
```

After conversion, add the Residual Joint Transformer open/closed target once:

```bash
python -m egomimic.scripts.piper_process.add_binary_gripper_residual_joint \
  /data/piper_zarr
```

The existing experiment configuration uses episode names
`teddy_0912_1` through `teddy_0912_5` for training and `teddy_0912_6` for
validation. This preserves the reported comparison. For a new dataset, remove
those filters and use the deterministic dataset split:

```bash
SPLIT_OVERRIDES=(
  '~data.train_datasets.piper_right_arm.filters'
  '~data.valid_datasets.piper_right_arm.filters'
  'data.train_datasets.piper_right_arm.mode=train'
  '+data.train_datasets.piper_right_arm.valid_ratio=0.1'
  'data.valid_datasets.piper_right_arm.mode=valid'
  '+data.valid_datasets.piper_right_arm.valid_ratio=0.1'
)
```

Keep a task-level or collection-level held-out set outside this split for final
evaluation. A low validation loss alone does not measure real-robot success.

## 3. Local training

First run a short forward/backward and validation check. It verifies data keys,
normalization, CUDA execution, losses, and checkpoint writing; it does not
evaluate policy quality.

```bash
python -m egomimic.trainHydra \
  --config-name train_piper_a_residual_joint_transformer \
  paths.dataset_dir=/data/piper_zarr \
  trainer.max_steps=20 \
  trainer.limit_train_batches=20 \
  trainer.limit_val_batches=2 \
  hydra.run.dir=.local_calibration/runs/residual-joint-transformer-a-smoke
```

Start the full A run from scratch while reusing only the smoke run's
normalization statistics:

```bash
python -m egomimic.trainHydra \
  --config-name train_piper_a_residual_joint_transformer \
  paths.dataset_dir=/data/piper_zarr \
  norm_stats.precomputed_norm_path=.local_calibration/runs/residual-joint-transformer-a-smoke/norm_stats \
  hydra.run.dir=.local_calibration/runs/residual-joint-transformer-a
```

For B, add the aligned Aria directory:

```bash
python -m egomimic.trainHydra \
  --config-name train_piper_b_residual_joint_transformer \
  paths.dataset_dir=/data/piper_zarr \
  data.train_datasets.human_left_arm.resolver.folder_path=/data/aria_left_aligned \
  hydra.run.dir=.local_calibration/runs/residual-joint-transformer-b
```

Append `"${SPLIT_OVERRIDES[@]}"` to these commands when training on episode
names other than the original teddy dataset. Each formal config runs for 30,000
optimizer steps with batch size 8 per domain, BF16 precision, AdamW at `3e-5`,
and cosine decay to `3e-6`.

Each run directory contains the composed Hydra config, CSV metrics,
normalization statistics, `checkpoints/last.ckpt`, and the best monitored
step-named checkpoint. Resume an interrupted run with both:

```bash
ckpt_path=/absolute/run/checkpoints/last.ckpt \
norm_stats.precomputed_norm_path=/absolute/run/norm_stats
```

## 4. Slurm launcher

`egomimic/scripts/piper_process/hpc/train_residual_joint_transformer.sbatch` runs the same smoke
gate followed by formal training. Cluster-specific account, partition, modules,
storage, and environment setup are intentionally supplied at submission time.

Create archives whose roots contain the episode Zarr stores:

```bash
tar -cf /work/data/piper_zarr.tar -C /data/piper_zarr .
tar -cf /work/data/aria_left_aligned.tar -C /data/aria_left_aligned .
```

If the cluster needs modules or a Conda environment, create an untracked setup
file such as `/work/env/piper.sh`:

```bash
module load <cluster CUDA/Python module>
source /work/envs/emimic/bin/activate
```

Submit A or B. The account and partition examples must be replaced with values
for the target cluster:

```bash
sbatch -A <account> -p <gpu-partition> \
  --export=ALL,PIPER_WORK_ROOT=/work/piper,PIPER_DATA_ARCHIVE=/work/data/piper_zarr.tar,PIPER_ENV_SETUP=/work/env/piper.sh \
  egomimic/scripts/piper_process/hpc/train_residual_joint_transformer.sbatch a

sbatch -A <account> -p <gpu-partition> \
  --export=ALL,PIPER_WORK_ROOT=/work/piper,PIPER_DATA_ARCHIVE=/work/data/piper_zarr.tar,ARIA_DATA_ARCHIVE=/work/data/aria_left_aligned.tar,PIPER_ENV_SETUP=/work/env/piper.sh \
  egomimic/scripts/piper_process/hpc/train_residual_joint_transformer.sbatch b
```

Hydra overrides placed after `a` or `b` are forwarded to both the smoke and
formal run. This also makes the launcher usable with another Piper dataset:

```bash
sbatch -A <account> -p <gpu-partition> \
  --export=ALL,PIPER_WORK_ROOT=/work/piper,PIPER_DATA_ARCHIVE=/work/data/new_task.tar,PIPER_ENV_SETUP=/work/env/piper.sh \
  egomimic/scripts/piper_process/hpc/train_residual_joint_transformer.sbatch a \
  '~data.train_datasets.piper_right_arm.filters' \
  '~data.valid_datasets.piper_right_arm.filters' \
  data.train_datasets.piper_right_arm.mode=train \
  +data.train_datasets.piper_right_arm.valid_ratio=0.1 \
  data.valid_datasets.piper_right_arm.mode=valid \
  +data.valid_datasets.piper_right_arm.valid_ratio=0.1
```

Supported launcher variables are:

| Variable | Default | Purpose |
|---|---|---|
| `PIPER_CODE` | repository containing the script | code checkout |
| `PIPER_WORK_ROOT` | `.local_calibration/hpc_residual_joint_transformer` | persistent runs and caches |
| `PIPER_DATA_ARCHIVE` | `$PIPER_WORK_ROOT/data/piper_zarr.tar` | Piper archive |
| `ARIA_DATA_ARCHIVE` | `$PIPER_WORK_ROOT/data/aria_left_aligned.tar` | B-only Aria archive |
| `PIPER_ENV_SETUP` | unset | sourced module/environment setup file |
| `PIPER_VENV` | `<repo>/emimic` | virtualenv used when no setup file is given |
| `PIPER_SCRATCH_ROOT` | `/tmp` | fallback when `SLURM_TMPDIR` is unavailable |
| `PIPER_RUN_DIR` | generated under work root | explicit output directory |

## 5. Reproduction record

For every run, retain the Git commit, composed Hydra config, normalization
statistics, dataset archive checksum, seed, Slurm log, and selected checkpoint.
Model weights and data require separate artifact storage; cloning this GitHub
repository alone cannot recreate them.

Before moving to a new machine, run the focused checks:

```bash
source emimic/bin/activate
pytest -q \
  egomimic/rldb/embodiment/test_piper_residual_joint.py \
  egomimic/models/test_residual_joint_transformer.py \
  egomimic/models/test_residual_joint_transformer_keyed.py \
  egomimic/robot/piper/test_gripper_utils.py \
  egomimic/robot/piper/test_rollout_joint.py
```
