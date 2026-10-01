#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=naiss2026-3-376-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zekun/egoverse_piper/logs/%x-%j.log

set -euo pipefail

ROOT=/home/zekun/egoverse_piper
GROUP=${1:?Pass a or b}
CODE=${PIPER_CODE:-$ROOT/code_clean_v2}
case "$GROUP" in
  a) CONFIG=train_piper_a_residual_clean_v2 ;;
  b) CONFIG=train_piper_b_mixed_clean_v2 ;;
  *) echo "GROUP must be a or b" >&2; exit 2 ;;
esac

module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
export LD_LIBRARY_PATH=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
source "$ROOT/code/emimic/bin/activate"
cd "$CODE"
export PYTHONPATH="$CODE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export TORCH_HOME="$ROOT/assets"
export TMPDIR=/scratch/local/piper-clean-v2-$GROUP-$SLURM_JOB_ID
export MPLCONFIGDIR="$TMPDIR/mpl"

ROBOT=$TMPDIR/robot
HUMAN=$TMPDIR/human
RUN=$ROOT/runs/clean-v2-$GROUP-$SLURM_JOB_ID
mkdir -p "$ROBOT" "$HUMAN" "$MPLCONFIGDIR" "$RUN"
tar -xf "$ROOT/data/piper_zarr.tar" -C "$ROBOT"

OVERRIDES=("paths.dataset_dir=$ROBOT" "num_workers=0")
if [[ "$GROUP" == b ]]; then
  tar -xf "$ROOT/data/aria_left_aligned.tar" -C "$HUMAN"
  OVERRIDES+=("data.train_datasets.human_left_arm.resolver.folder_path=$HUMAN")
fi

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
