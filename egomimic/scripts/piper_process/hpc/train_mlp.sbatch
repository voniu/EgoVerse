#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=piper-mlp
set -euo pipefail
PIPER_ROOT=/home/zekun/egoverse_piper
PIPER_GROUP=${1:?Pass a or b}
PIPER_CODE="$PIPER_ROOT/code_mlp_20260928"
case "$PIPER_GROUP" in
  a) PIPER_STATS="$PIPER_ROOT/assets/norm_stats.json" ;;
  b) PIPER_STATS="$PIPER_ROOT/assets/b_norm_stats.json" ;;
  *) exit 2 ;;
esac
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
export LD_LIBRARY_PATH=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
source "$PIPER_ROOT/code/emimic/bin/activate"
cd "$PIPER_CODE"
export PYTHONPATH="$PIPER_CODE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export TORCH_HOME="$PIPER_ROOT/assets"
export TMPDIR=/scratch/local/piper-mlp-$PIPER_GROUP-$SLURM_JOB_ID
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$TMPDIR/robot" "$TMPDIR/human"
tar -xf "$PIPER_ROOT/data/piper_zarr.tar" -C "$TMPDIR/robot"
PIPER_OVERRIDES=("paths.dataset_dir=$TMPDIR/robot" "norm_stats.precomputed_norm_path=$PIPER_STATS" "num_workers=0")
if [[ "$PIPER_GROUP" == b ]]; then
  tar -xf "$PIPER_ROOT/data/aria_left_aligned.tar" -C "$TMPDIR/human"
  PIPER_OVERRIDES+=("data.train_datasets.human_left_arm.resolver.folder_path=$TMPDIR/human")
fi
PIPER_CONFIG=train_piper_${PIPER_GROUP}_mlp
PIPER_RUN="$PIPER_ROOT/runs/${PIPER_GROUP}-mlp-$SLURM_JOB_ID"
# Same batch/workers as formal training, multiple epochs, then checkpoint restore.
srun python -m egomimic.trainHydra --config-name "$PIPER_CONFIG" \
  "${PIPER_OVERRIDES[@]}" trainer.max_steps=1000 "hydra.run.dir=$PIPER_RUN/preflight"
srun python -m egomimic.trainHydra --config-name "$PIPER_CONFIG" \
  "${PIPER_OVERRIDES[@]}" trainer.max_steps=1001 \
  "ckpt_path=$PIPER_RUN/preflight/checkpoints/final.ckpt" "hydra.run.dir=$PIPER_RUN/restore_check"
# Start formal training fresh, with original 10k-step A/B budget and seed.
srun python -m egomimic.trainHydra --config-name "$PIPER_CONFIG" \
  "${PIPER_OVERRIDES[@]}" "hydra.run.dir=$PIPER_RUN/train"
