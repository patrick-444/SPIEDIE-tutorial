#!/bin/bash
# Submit this job from the project directory:
#   sbatch run_spiedie.slurm
#
# Watch it with:
#   squeue -u "USER"
# Read the scheduler log after it finishes:
#   cat logs/slurm-<jobid>.out

#SBATCH --job-name=sklearn-demo
#SBATCH --partition=Standard
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# sbatch starts in the directory where the job was submitted.
mkdir -p logs results

# Batch jobs do not inherit modules loaded in the login session.
module load python

if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
else
    source .venv/bin/activate
fi

python train.py
