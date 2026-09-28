#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=piper-setup

set -euo pipefail
PIPER_ROOT=/home/zekun/egoverse_piper
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
export LD_LIBRARY_PATH=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
cd "$PIPER_ROOT/code"
python -m venv --system-site-packages emimic
source emimic/bin/activate
export PIP_NO_CACHE_DIR=1
export TMPDIR=/scratch/local/piper-setup-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
python -m pip install 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r egomimic/scripts/piper_process/hpc/requirements.txt
export PYTHONPATH="$PIPER_ROOT/code:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$TMPDIR/mpl"
python -c 'import torch; import egomimic.trainHydra; from egomimic.rldb.embodiment.piper import Piper; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
python -m pip freeze > "$PIPER_ROOT/logs/environment.txt"
module list 2> "$PIPER_ROOT/logs/modules.txt"
du -sh "$PIPER_ROOT/code/emimic"
