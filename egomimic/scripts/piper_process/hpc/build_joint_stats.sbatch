#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=piper-joint-stats

set -euo pipefail
PIPER_ROOT=/home/zekun/egoverse_piper
PIPER_CODE="$PIPER_ROOT/code_joint_head"
PIPER_WORK="/scratch/local/piper-joint-stats-${SLURM_JOB_ID}"

module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
export LD_LIBRARY_PATH=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
source "$PIPER_ROOT/code/emimic/bin/activate"
cd "$PIPER_CODE"
export PYTHONPATH="$PIPER_CODE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export TMPDIR="$PIPER_WORK/tmp"
export MPLCONFIGDIR="$TMPDIR/mpl"

mkdir -p "$PIPER_WORK/robot" "$TMPDIR"
trap 'rm -rf "$PIPER_WORK"' EXIT
tar -xf "$PIPER_ROOT/data/piper_zarr.tar" -C "$PIPER_WORK/robot"

python -m egomimic.scripts.piper_process.build_joint_norm_stats \
  --dataset "$PIPER_WORK/robot" \
  --base-a "$PIPER_ROOT/assets/norm_stats.json" \
  --base-b "$PIPER_ROOT/assets/b_norm_stats.json" \
  --output-a "$PIPER_WORK/joint_a_norm_stats.json" \
  --output-b "$PIPER_WORK/joint_b_norm_stats.json"

python - "$PIPER_WORK" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for group in "ab":
    path = root / f"joint_{group}_norm_stats.json"
    payload = json.loads(path.read_text())
    robot = payload["stats"]["7"]
    assert payload["frames"] == 22406
    assert payload["joint_stats"]["episodes"] == 63
    assert len(robot["observations.state.joints"]["mean"]) == 7
    assert len(robot["actions_joints"]["mean"]) == 30
    assert len(robot["actions_joints"]["mean"][0]) == 7
assert "2" not in json.loads((root / "joint_a_norm_stats.json").read_text())["stats"]
assert "2" in json.loads((root / "joint_b_norm_stats.json").read_text())["stats"]
PY

install -m 664 "$PIPER_WORK/joint_a_norm_stats.json" "$PIPER_ROOT/assets/joint_a_norm_stats.json"
install -m 664 "$PIPER_WORK/joint_b_norm_stats.json" "$PIPER_ROOT/assets/joint_b_norm_stats.json"
sha256sum "$PIPER_ROOT/assets/joint_a_norm_stats.json" \
  "$PIPER_ROOT/assets/joint_b_norm_stats.json"
