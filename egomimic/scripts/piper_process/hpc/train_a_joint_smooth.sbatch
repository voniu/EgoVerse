#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=naiss2026-3-376-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zekun/egoverse_piper/logs/%x-%j.log

set -euo pipefail

PIPER_ROOT=/home/zekun/egoverse_piper
PIPER_HEAD=${1:?Pass mlp or flow}
PIPER_CODE=${PIPER_CODE:-$PIPER_ROOT/code_joint_smooth_v1}
case "$PIPER_HEAD" in
  mlp|flow) ;;
  *) echo "PIPER_HEAD must be mlp or flow" >&2; exit 2 ;;
esac

module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
export LD_LIBRARY_PATH=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
source "$PIPER_ROOT/code/emimic/bin/activate"
cd "$PIPER_CODE"
export PYTHONPATH="$PIPER_CODE:${PYTHONPATH:-}"
export EGOVERSE_GIT_COMMIT
EGOVERSE_GIT_COMMIT=$(cat "$PIPER_CODE/SOURCE_COMMIT")
export EGOVERSE_GIT_BRANCH=codex/piper-policy-experiments
export EGOVERSE_SOURCE_DIRTY=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export TORCH_HOME="$PIPER_ROOT/assets"
export TMPDIR=/scratch/local/piper-a-joint-$PIPER_HEAD-smooth-v1-$SLURM_JOB_ID
export MPLCONFIGDIR="$TMPDIR/mpl"

PIPER_DATA=$TMPDIR/robot
PIPER_RUN=$PIPER_ROOT/runs/a-joint-$PIPER_HEAD-smooth-v1-$SLURM_JOB_ID
PIPER_STATS=$PIPER_RUN/norm_stats.json
mkdir -p "$PIPER_DATA" "$MPLCONFIGDIR" "$PIPER_RUN"
tar -xf "$PIPER_ROOT/data/piper_zarr.tar" -C "$PIPER_DATA"

python -m egomimic.scripts.piper_process.smooth_robot_actions \
  --dataset "$PIPER_DATA" \
  --joint-window 9 \
  --gripper-window 5 \
  --report "$PIPER_RUN/action_smoothing_report.json"

python -m egomimic.scripts.piper_process.build_joint_norm_stats \
  --dataset "$PIPER_DATA" \
  --base-a "$PIPER_ROOT/assets/norm_stats.json" \
  --base-b "$PIPER_ROOT/assets/b_norm_stats.json" \
  --output-a "$PIPER_STATS" \
  --output-b "$TMPDIR/unused_b_stats.json"

srun python -m egomimic.trainHydra \
  --config-name "train_piper_a_joint_$PIPER_HEAD" \
  "paths.dataset_dir=$PIPER_DATA" \
  "norm_stats.precomputed_norm_path=$PIPER_STATS" \
  num_workers=0 \
  trainer.max_steps=10000 \
  "hydra.run.dir=$PIPER_RUN"
