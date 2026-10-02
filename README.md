# SPIEDIE sklearn tutorial

A small classification job for demonstrating how to run Python on [SPIEDIE](https://spiediedocs.binghamton.edu/), Binghamton's SLURM cluster.

`train.py` fits a logistic regression model and a random forest on scikit-learn's built-in digits dataset, then writes scores to `results/`.

## Project layout

```
.gitignore
README.md
requirements.txt
train.py
run_spiedie.slurm
logs/                 scheduler stdout and stderr
results/              metrics.json and classification_report.txt
.venv/                created locally, or by the job on first submit
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

## Run on SPIEDIE

Copy this directory to the cluster, then submit from inside it:

```bash
sbatch run_spiedie.slurm
squeue -u "$USER"
```

The script requests the `Standard` partition (short prototyping jobs; they can be preempted after one hour), one task, two CPUs, 4 GB of memory, and a 10 minute limit. The first run creates `.venv` on the compute node and installs scikit-learn. Later runs reuse that environment.

Scheduler output lands in `logs/slurm-<jobid>.out`. Model scores land in `results/`.

To try a different partition without editing the script:

```bash
sbatch --partition=normal run_spiedie.slurm
```
