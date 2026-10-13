#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

GROUP=${1:?Pass a or b}
shift
EXTRA_OVERRIDES=("$@")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CODE=${PIPER_CODE:-$REPO_ROOT}
WORK_ROOT=${PIPER_WORK_ROOT:-$CODE/.local_calibration/hpc_residual_joint_transformer}
PIPER_DATA_ARCHIVE=${PIPER_DATA_ARCHIVE:-$WORK_ROOT/data/piper_zarr.tar}
ARIA_DATA_ARCHIVE=${ARIA_DATA_ARCHIVE:-$WORK_ROOT/data/aria_left_aligned.tar}
case "$GROUP" in
  a) CONFIG=train_piper_a_residual_joint_transformer ;;
  b) CONFIG=train_piper_b_residual_joint_transformer ;;
  *) echo "GROUP must be a or b" >&2; exit 2 ;;
esac

if [[ -n "${PIPER_ENV_SETUP:-}" ]]; then
  source "$PIPER_ENV_SETUP"
else
  source "${PIPER_VENV:-$CODE/emimic}/bin/activate"
fi
cd "$CODE"
export PYTHONPATH="$CODE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export TORCH_HOME=${TORCH_HOME:-$WORK_ROOT/assets}
SCRATCH_BASE=${SLURM_TMPDIR:-${PIPER_SCRATCH_ROOT:-/tmp}}
export TMPDIR=$SCRATCH_BASE/piper-residual-joint-transformer-$GROUP-${SLURM_JOB_ID:-manual}
export MPLCONFIGDIR="$TMPDIR/mpl"

ROBOT=$TMPDIR/robot
HUMAN=$TMPDIR/human
RUN=${PIPER_RUN_DIR:-$WORK_ROOT/runs/residual-joint-transformer-$GROUP-${SLURM_JOB_ID:-manual}}
mkdir -p "$ROBOT" "$HUMAN" "$MPLCONFIGDIR" "$RUN"
tar -xf "$PIPER_DATA_ARCHIVE" -C "$ROBOT"
# Optional Smooth v1 of the extracted copy (commanded joints only); default off.
if [[ "${PIPER_SMOOTH:-0}" == 1 ]]; then
  python -m egomimic.scripts.piper_process.smooth_robot_actions \
    --dataset "$ROBOT" \
    --report "$RUN/action_smoothing_report.json"
fi

OVERRIDES=("paths.dataset_dir=$ROBOT" "num_workers=0")
if [[ "$GROUP" == b ]]; then
  tar -xf "$ARIA_DATA_ARCHIVE" -C "$HUMAN"
  OVERRIDES+=("data.train_datasets.human_left_arm.resolver.folder_path=$HUMAN")
fi
OVERRIDES+=("${EXTRA_OVERRIDES[@]}")

# Short GPU/data/FK backward check. The formal run reuses only its normalization
# statistics and starts model/optimizer training from scratch.
srun python -m egomimic.trainHydra \
  --config-name "$CONFIG" \
  "${OVERRIDES[@]}" \
  trainer.max_steps=20 \
  trainer.limit_train_batches=20 \
  trainer.limit_val_batches=2 \
  "hydra.run.dir=$RUN/smoke"

test -f "$RUN/smoke/norm_stats/norm_stats.json"
srun python -m egomimic.trainHydra \
  --config-name "$CONFIG" \
  "${OVERRIDES[@]}" \
  "norm_stats.precomputed_norm_path=$RUN/smoke/norm_stats" \
  "hydra.run.dir=$RUN/train"
