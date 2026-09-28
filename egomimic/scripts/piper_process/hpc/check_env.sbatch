#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=piper-env
set -euo pipefail
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
# The flat Python module omits the compiler runtime needed by its SciPy build.
export LD_LIBRARY_PATH=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
cd /home/zekun/egoverse_piper/code
source emimic/bin/activate
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
export PIP_NO_CACHE_DIR=1
python -m pip install 'scaleapi==2.18.4'
python - <<'PY'
import torch
import scipy.signal
import scipy.linalg
from scipy.spatial.transform import Rotation
import egomimic.trainHydra
assert torch.cuda.is_available()
x = torch.randn(32, 32, device='cuda', requires_grad=True)
(x @ x.T).sum().backward()
assert torch.isfinite(x.grad).all()
print('ENV_CHECK_PASSED', torch.__version__, torch.cuda.get_device_name(), Rotation.identity().as_quat())
PY
python -m pip freeze > /home/zekun/egoverse_piper/logs/environment-verified.txt
