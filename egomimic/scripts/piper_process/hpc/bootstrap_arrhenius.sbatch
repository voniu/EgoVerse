#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=piper-env

# Build and verify the GH200 training environment from the hashed aarch64 lock.
# Submit from the repository root:
#   sbatch -A <gpu-account> -p gpu --export=ALL,RUN_DIR=<dir under $HPC_RUNS> \
#     egomimic/scripts/piper_process/hpc/bootstrap_arrhenius.sbatch
set -euo pipefail
source "$HOME/.config/hpc/env.sh"
test "$(uname -m)" = aarch64

profile=py31114-uv01220-torch271cu128
env_parent="$HPC_ENVS/egoverse-piper/aarch64"
env_dir="$env_parent/$profile"
repo=${SLURM_SUBMIT_DIR:?}
lock="$repo/egomimic/scripts/piper_process/hpc/requirements-training-aarch64.lock"
uv="$HPC_ROOT/tools/aarch64/bin/uv"

: "${RUN_DIR:?Set RUN_DIR to an existing directory under HPC_RUNS}"
case "$RUN_DIR" in "$HPC_RUNS"/*) ;; *) echo "RUN_DIR must be under HPC_RUNS" >&2; exit 1 ;; esac
test -d "$RUN_DIR" && test -f "$lock" && test -x "$uv"
test "$("$uv" --version | awk '{print $2}')" = 0.12.20
if [[ -e "$env_dir" ]]; then echo "Environment already exists: $env_dir" >&2; exit 1; fi

scratch=$(mktemp -d "/scratch/local/zekun-piper-env-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch"
install -d -m 700 "$env_parent"
python_path="$("$uv" python find 3.11.14)"
stage="$env_parent/.${profile}.staging-${SLURM_JOB_ID}"
"$uv" venv --python "$python_path" "$stage"
"$uv" pip sync --python "$stage/bin/python" --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match "$lock"
"$uv" pip check --python "$stage/bin/python"

cd "$repo"
export PYTHONPATH="$repo" MPLBACKEND=Agg MPLCONFIGDIR="$scratch/mpl"
"$stage/bin/python" - <<'PY'
import platform
import torch
import torchvision
import egomimic.trainHydra  # noqa: F401
print("machine", platform.machine(), "torch", torch.__version__, "cuda", torch.version.cuda,
      "torchvision", torchvision.__version__)
assert platform.machine() == "aarch64" and torch.cuda.is_available()
x = torch.randn(64, 64, device="cuda", requires_grad=True)
(x @ x.T).sum().backward()
assert torch.isfinite(x.grad).all()
boxes = torch.tensor([[0.0, 0, 1, 1]], device="cuda")
torchvision.ops.nms(boxes, torch.ones(1, device="cuda"), 0.5)
print("device", torch.cuda.get_device_name(0), "ENV_CHECK_PASSED")
PY
"$stage/bin/python" -m pytest -q -p no:cacheprovider \
  egomimic/rldb/embodiment/test_piper_residual_joint.py \
  egomimic/models/test_residual_joint_transformer.py \
  egomimic/models/test_residual_joint_transformer_keyed.py \
  egomimic/robot/piper/test_rollout_joint.py \
  egomimic/algo/test_hpt_multihead_loss.py \
  egomimic/scripts/piper_process/test_smooth_robot_actions.py
mv "$stage" "$env_dir"

git -C "$repo" rev-parse HEAD > "$RUN_DIR/git-commit.txt"
git -C "$repo" diff HEAD > "$RUN_DIR/git-diff.patch"
module -t list > "$RUN_DIR/modules.txt" 2>&1 || true
"$uv" --version > "$RUN_DIR/uv-version.txt"
"$env_dir/bin/python" --version > "$RUN_DIR/python-version.txt" 2>&1
"$uv" pip freeze --python "$env_dir/bin/python" > "$RUN_DIR/environment-freeze.txt"
printf '%s\n' "$env_dir" > "$RUN_DIR/environment-path.txt"
df -h /scratch/local > "$RUN_DIR/scratch-local-df.txt"
date -Is > "$RUN_DIR/completion.txt"
