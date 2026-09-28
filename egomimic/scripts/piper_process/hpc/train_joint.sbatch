#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=piper-joint

set -euo pipefail
PIPER_ROOT=/home/zekun/egoverse_piper
PIPER_GROUP=${1:?Pass a or b}
PIPER_HEAD=${2:?Pass flow or mlp}
PIPER_CODE="$PIPER_ROOT/code_joint_head"

case "$PIPER_GROUP:$PIPER_HEAD" in
  a:flow|a:mlp|b:flow|b:mlp) ;;
  *) echo "Expected group a|b and head flow|mlp" >&2; exit 2 ;;
esac

PIPER_CONFIG="train_piper_${PIPER_GROUP}_joint_${PIPER_HEAD}"
PIPER_STATS="$PIPER_ROOT/assets/joint_${PIPER_GROUP}_norm_stats.json"
PIPER_WORK="/scratch/local/piper-${PIPER_GROUP}-joint-${PIPER_HEAD}-${SLURM_JOB_ID}"
PIPER_RESULT="$PIPER_ROOT/runs/${PIPER_GROUP}-joint-${PIPER_HEAD}-${SLURM_JOB_ID}"

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
export TMPDIR="$PIPER_WORK/tmp"
export MPLCONFIGDIR="$TMPDIR/mpl"

mkdir -p "$PIPER_WORK/robot" "$PIPER_WORK/human" "$TMPDIR" "$PIPER_RESULT"
tar -xf "$PIPER_ROOT/data/piper_zarr.tar" -C "$PIPER_WORK/robot"
PIPER_OVERRIDES=(
  "paths.dataset_dir=$PIPER_WORK/robot"
  "norm_stats.precomputed_norm_path=$PIPER_STATS"
  "num_workers=0"
)
if [[ "$PIPER_GROUP" == b ]]; then
  tar -xf "$PIPER_ROOT/data/aria_left_aligned.tar" -C "$PIPER_WORK/human"
  PIPER_OVERRIDES+=("data.train_datasets.human_left_arm.resolver.folder_path=$PIPER_WORK/human")
fi

persist_metadata() {
  local source="$PIPER_WORK/train"
  [[ -d "$source/.hydra" ]] && cp -a "$source/.hydra" "$PIPER_RESULT/"
  [[ -d "$source/metrics" ]] && cp -a "$source/metrics" "$PIPER_RESULT/"
  [[ -f "$source/run_provenance.json" ]] && cp "$source/run_provenance.json" "$PIPER_RESULT/"
  [[ -f "$source/trainHydra.log" ]] && cp "$source/trainHydra.log" "$PIPER_RESULT/"
}
trap persist_metadata EXIT

# GPU/data/checkpoint-path preflight; formal training starts from fresh weights.
srun python -m egomimic.trainHydra --config-name "$PIPER_CONFIG" \
  "${PIPER_OVERRIDES[@]}" trainer.max_steps=10 \
  "hydra.run.dir=$PIPER_WORK/preflight"
srun python -m egomimic.trainHydra --config-name "$PIPER_CONFIG" \
  "${PIPER_OVERRIDES[@]}" "hydra.run.dir=$PIPER_WORK/train"

mkdir -p "$PIPER_RESULT/checkpoints"
cp "$PIPER_WORK/train/checkpoints/final.ckpt" "$PIPER_RESULT/checkpoints/"
find "$PIPER_WORK/train/checkpoints" -maxdepth 1 -name 'epoch_*.ckpt' \
  -exec cp {} "$PIPER_RESULT/checkpoints/" \;
persist_metadata
trap - EXIT
