#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=piper-bias

# Open-loop first-step bias of one checkpoint on train and validation episodes.
#   sbatch -A <acct> -p gpu --export=ALL,PIPER_ENV_SETUP=...,PIPER_DATA_DIR=...,\
#     PIPER_VALID_EPISODES="6 7 ...",PIPER_TRAIN_EPISODES="0 12 ..." \
#     openloop_bias.sbatch <checkpoint> <output-dir>
# Episode indices must match the split of the checkpoint's training run.
set -euo pipefail
CHECKPOINT=${1:?checkpoint}
OUT=${2:?output directory}
CODE=${PIPER_CODE:?set PIPER_CODE}
source "${PIPER_ENV_SETUP:?set PIPER_ENV_SETUP}"
PYTHON=$(command -v python)
cd "$CODE"
export PYTHONPATH="$CODE:${PYTHONPATH:-}" MPLBACKEND=Agg OMP_NUM_THREADS=4
mkdir -p "$OUT"
for split in valid train; do
  case $split in
    valid) read -r -a episodes <<< "${PIPER_VALID_EPISODES:?set PIPER_VALID_EPISODES}" ;;
    train) read -r -a episodes <<< "${PIPER_TRAIN_EPISODES:?set PIPER_TRAIN_EPISODES}" ;;
  esac
  srun --export=ALL "$PYTHON" -m egomimic.scripts.piper_process.openloop_bias \
    --checkpoint "$CHECKPOINT" --dataset "${PIPER_DATA_DIR:?set PIPER_DATA_DIR}" \
    --episodes "${episodes[@]}" --output "$OUT/openloop_bias_$split.json"
done
