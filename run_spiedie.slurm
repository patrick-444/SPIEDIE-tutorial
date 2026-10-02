#!/bin/bash
# Submit from the project directory, after creating the log folder:
#   mkdir -p logs
#   sbatch run_spiedie.slurm
#   sbatch run_spiedie.slurm 0.001 50 42
#   sbatch run_spiedie.slurm 0.01 50 42
#
# Watch it:
#   squeue -u "$USER"
#   tail -f logs/train_<jobid>.out

#SBATCH --job-name=train
#SBATCH --partition=Standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results

# Batch jobs do not inherit modules from the login session.
# Activate the project environment the same way you would interactively.
module load python
source .venv/bin/activate

export KERAS_BACKEND=torch

python -u train.py --lr "${1:-0.001}" --epochs "${2:-50}" --seed "${3:-42}"
