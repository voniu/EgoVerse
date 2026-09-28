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
cd /home/zekun/egoverse_piper/code
bash egomimic/scripts/piper_process/hpc/setup.sbatch
bash egomimic/scripts/piper_process/hpc/train.sbatch
