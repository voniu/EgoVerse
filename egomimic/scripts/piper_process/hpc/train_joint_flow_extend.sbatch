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
PIPER_GROUP=${1:?Pass a or b}
PIPER_CODE=${PIPER_CODE:-$PIPER_ROOT/code_joint_flow_extend}

case "$PIPER_GROUP" in
  a)
    PIPER_CONFIG=train_piper_a_joint_flow
    PIPER_STATS=$PIPER_ROOT/assets/joint_a_norm_stats.json
    PIPER_INIT=$PIPER_ROOT/runs/a-joint-flow-3118129/checkpoints/epoch_epoch=95.ckpt
    ;;
  b)
    PIPER_CONFIG=train_piper_b_joint_flow
    PIPER_STATS=$PIPER_ROOT/assets/joint_b_norm_stats.json
    PIPER_INIT=$PIPER_ROOT/runs/b-joint-flow-3118131/checkpoints/epoch_epoch=97.ckpt
    ;;
  *)
    echo "PIPER_GROUP must be a or b" >&2
    exit 2
    ;;
esac

module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
export LD_LIBRARY_PATH=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
source "$PIPER_ROOT/code/emimic/bin/activate"
cd "$PIPER_CODE"
export PYTHONPATH="$PIPER_CODE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export TORCH_HOME="$PIPER_ROOT/assets"
export TMPDIR=/scratch/local/piper-$PIPER_GROUP-joint-flow-extend-$SLURM_JOB_ID
export MPLCONFIGDIR="$TMPDIR/mpl"

mkdir -p "$TMPDIR/robot" "$TMPDIR/human" "$MPLCONFIGDIR"
tar -xf "$PIPER_ROOT/data/piper_zarr.tar" -C "$TMPDIR/robot"

PIPER_OVERRIDES=(
  "paths.dataset_dir=$TMPDIR/robot"
  "norm_stats.precomputed_norm_path=$PIPER_STATS"
  "num_workers=0"
  "init_ckpt_path=$PIPER_INIT"
  "model.optimizer.lr=1e-5"
  "model.scheduler.T_max=10000"
  "model.scheduler.eta_min=3e-6"
  "trainer.max_steps=10000"
)

if [[ "$PIPER_GROUP" == b ]]; then
  tar -xf "$PIPER_ROOT/data/aria_left_aligned.tar" -C "$TMPDIR/human"
  PIPER_OVERRIDES+=(
    "data.train_datasets.human_left_arm.resolver.folder_path=$TMPDIR/human"
  )
fi

PIPER_RUN=$PIPER_ROOT/runs/$PIPER_GROUP-joint-flow-extend-$SLURM_JOB_ID
srun python -m egomimic.trainHydra \
  --config-name "$PIPER_CONFIG" \
  "${PIPER_OVERRIDES[@]}" \
  "hydra.run.dir=$PIPER_RUN"
