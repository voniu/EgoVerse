#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=piper-a

set -euo pipefail
PIPER_ROOT=/home/zekun/egoverse_piper
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
export LD_LIBRARY_PATH=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
cd "$PIPER_ROOT/code"
source emimic/bin/activate
export PYTHONPATH="$PIPER_ROOT/code:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export TORCH_HOME="$PIPER_ROOT/assets"
export TMPDIR=/scratch/local/piper-a-$SLURM_JOB_ID
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$TMPDIR/data"
tar -xf "$PIPER_ROOT/data/piper_zarr.tar" -C "$TMPDIR/data"
PIPER_RUN="$PIPER_ROOT/runs/a-$SLURM_JOB_ID"

# A short check on this architecture before starting the unchanged A config.
srun python -m egomimic.trainHydra --config-name train_piper_smoke \
  paths.dataset_dir="$TMPDIR/data" \
  norm_stats.precomputed_norm_path="$PIPER_ROOT/assets/norm_stats.json" \
  hydra.run.dir="$PIPER_RUN/smoke"

# Fresh initialization: do not fine-tune the smoke checkpoint.
srun python -m egomimic.trainHydra --config-name train_piper \
  paths.dataset_dir="$TMPDIR/data" \
  norm_stats.precomputed_norm_path="$PIPER_ROOT/assets/norm_stats.json" \
  hydra.run.dir="$PIPER_RUN/train" "$@"
