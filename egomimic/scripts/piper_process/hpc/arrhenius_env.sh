# Arrhenius training environment for PIPER_ENV_SETUP (source, do not execute).
# Follows /home/zekun/.config/hpc/CONVENTION.md: paths from env.sh, a uv venv
# per project/arch/profile under $HPC_ENVS, GPU nodes only.
source "$HOME/.config/hpc/env.sh"
test "$(uname -m)" = aarch64
PIPER_ENV_PROFILE=${PIPER_ENV_PROFILE:-py31114-uv01220-torch271cu128}
PIPER_ENV_DIR="$HPC_ENVS/egoverse-piper/aarch64/$PIPER_ENV_PROFILE"
test -x "$PIPER_ENV_DIR/bin/python"
source "$PIPER_ENV_DIR/bin/activate"
export PIPER_SCRATCH_ROOT=${PIPER_SCRATCH_ROOT:-/scratch/local}
