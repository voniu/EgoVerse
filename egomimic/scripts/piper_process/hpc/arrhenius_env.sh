# Arrhenius training environment for PIPER_ENV_SETUP (source, do not execute).
# Follows ~/.config/hpc/CONVENTION.md (cluster convention): paths from env.sh, a uv venv
# per project/arch/profile under $HPC_ENVS, GPU nodes only.
source "$HOME/.config/hpc/env.sh"
test "$(uname -m)" = aarch64
PIPER_ENV_PROFILE=${PIPER_ENV_PROFILE:-py31114-uv01220-torch271cu128}
PIPER_ENV_DIR="$HPC_ENVS/egoverse-piper/aarch64/$PIPER_ENV_PROFILE"
test -x "$PIPER_ENV_DIR/bin/python"
# The venv was built in a staging directory and moved, so its activate script
# points at the old path; set the environment directly and verify it.
export VIRTUAL_ENV="$PIPER_ENV_DIR"
export PATH="$PIPER_ENV_DIR/bin:$PATH"
hash -r
test "$(command -v python)" = "$PIPER_ENV_DIR/bin/python"
export PIPER_SCRATCH_ROOT=${PIPER_SCRATCH_ROOT:-/scratch/local}
