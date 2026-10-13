#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=piper-bias

# Open-loop first-step bias of one checkpoint on train and validation episodes.
#   sbatch -A <acct> -p gpu --export=ALL,PIPER_ENV_SETUP=...,PIPER_DATA_DIR=... \
#     openloop_bias.sbatch <checkpoint> <output-dir>
set -euo pipefail
CHECKPOINT=${1:?checkpoint}
OUT=${2:?output directory}
CODE=${PIPER_CODE:?set PIPER_CODE}
source "${PIPER_ENV_SETUP:?set PIPER_ENV_SETUP}"
PYTHON=$(command -v python)
cd "$CODE"
export PYTHONPATH="$CODE:${PYTHONPATH:-}" MPLBACKEND=Agg OMP_NUM_THREADS=4
mkdir -p "$OUT"
# Validation episodes of the assembly split (valid_ratio 0.1) and five training ones.
for split in valid train; do
  case $split in
    valid) episodes=(6 7 15 22 39 42 66) ;;
    train) episodes=(0 12 30 48 60) ;;
  esac
  srun --export=ALL "$PYTHON" -m egomimic.scripts.piper_process.openloop_bias \
    --checkpoint "$CHECKPOINT" --dataset "${PIPER_DATA_DIR:?set PIPER_DATA_DIR}" \
    --episodes "${episodes[@]}" --output "$OUT/openloop_bias_$split.json"
done
