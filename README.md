# SLURM `#SBATCH` Guide (Spiedie)

This guide walks through one job from start to finish: submitting a Python script with a few command-line arguments, then tracking it.

> Anything in `<angle_brackets>` is a placeholder. Partition names, module names, and limits are cluster-specific, so confirm them with `sinfo` and the Spiedie documentation.

---

## Spiedie-Specific Notes

**Software requirements.** You can use any terminal program that is ssh enabled to access SPIEDIE. This includes Powershell (Windows), Terminal (macOS), or any Linux distro.

**Network access.** You must be connected to the Binghamton VPN or using an on-campus machine to reach Spiedie. If SSH hangs or times out, check your VPN first.

```bash
ssh <netid>@<spiedie.binghamton.edu>
```

**Git on the head node.** The head node has git read and write privileges, so you must authenticate via git before pushing or pulling changes. SSH keys are the preferred method. They take a one-time setup (next section), after which git works every time you log in.

---

## Authenticating Git from Spiedie (One-Time SSH Setup)

Do this once. Afterwards you log in to Spiedie with your username and password as usual, and `git pull` / `git push` just work. 

Run everything below **on the Spiedie head node** (do not change your working directory once you access SPIEDIE). 

### Step 1: Generate a key with no passphrase

First check whether you already have one:

```bash
ls -al ~/.ssh
```

If `id_ed25519` and `id_ed25519.pub` already exist, skip to Step 2. Otherwise:

```bash
mkdir -p ~/.ssh
ssh-keygen -t ed25519 -C "<you@example.com>" -f ~/.ssh/id_ed25519 -N ""
```

`-N ""` sets an empty passphrase. That is what makes this a one-time setup: git can use the key immediately in every future login, with nothing to unlock.

This creates two files:

| File | What it is | Share it? |
|---|---|---|
| `~/.ssh/id_ed25519` | **Private key** | **Never.** Don't copy it off Spiedie, email it, or commit it to git. |
| `~/.ssh/id_ed25519.pub` | **Public key** | Yes. This is what you give to GitHub. |

Because the key has no passphrase, its protection comes from your Spiedie account login and the file permissions in the next step.

### Step 2: Lock down permissions

SSH refuses to use keys that other users can read.

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

### Step 3: Add the public key to GitHub

Print the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the whole output (one line starting with `ssh-ed25519`). Then, in your browser:

1. Click your profile picture, then **Settings**.
2. Choose **SSH and GPG keys** in the sidebar.
3. Click **New SSH key**.
4. Title it something recognizable, such as `spiedie-headnode`.
5. Set the key type to **Authentication Key**.
6. Paste the key into the **Key** box and click **Add SSH key**.

### Step 4: Tell SSH to always use this key for GitHub

Create `~/.ssh/config`:

```bash
cat >> ~/.ssh/config << 'CONF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
CONF
chmod 600 ~/.ssh/config
```

### Step 5: Trust GitHub's host key (first connection only)

```bash
ssh -T git@github.com
```

SSH shows a fingerprint and asks whether to continue. Compare it with the fingerprint GitHub publishes in its documentation, then type `yes`. A successful test prints:

```
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

The "no shell access" part is expected.

### Step 6: Set your git identity

```bash
git config --global user.name  "<Your Name>"
git config --global user.email "<you@example.com>"
```

Use an email tied to your GitHub account so commits are attributed to you.

### Step 7: Use SSH URLs for your repositories

New clone:

```bash
git clone git@github.com:<username>/<repo>.git
```

Existing repo that uses HTTPS:

```bash
git remote -v
git remote set-url origin git@github.com:<username>/<repo>.git
```

An SSH URL starts with `git@`. If the remote still starts with `https://`, git will keep asking for a username and password.

### Step 8: Verify it persists

Log out of Spiedie, log back in, and run:

```bash
cd <repo>
git pull
```

If it succeeds with no prompts, setup is complete. From now on the routine is:

1. Connect to the VPN (or be on campus).
2. `ssh <netid>@<spiedie.binghamton.edu>` and enter your password.
3. Use `git pull` and `git push` freely.

### Setup troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Permission denied (publickey)` | Public key not added to GitHub, or wrong key offered | Redo Step 3. Run `ssh -vT git@github.com` to see which key is offered. |
| Asked for a username and password | Remote uses HTTPS | Switch to the SSH URL (Step 7). |
| `Bad owner or permissions` | Key, `~/.ssh`, or config is too open | Redo Step 2 and `chmod 600 ~/.ssh/config`. |
| `Host key verification failed` | Stale entry in `~/.ssh/known_hosts` | `ssh-keygen -R github.com`, then redo Step 5. |
| Connection times out | Not on VPN or campus network, or port 22 is blocked | Connect to the VPN. GitHub also supports SSH over port 443 via `ssh.github.com`. |
| Key was exposed or the account is compromised | Private key leaked | Delete the key under GitHub Settings, then redo Steps 1 to 3 with a new key. |

Keep large outputs, checkpoints, and logs out of git (use `.gitignore`).

---

## 1. An Example: Python Script with CLI Args

Suppose you have `train.py` that takes three arguments (see train.py):

```bash
python train.py --lr 0.001 --epochs 50 --seed 42
```

To run it on the cluster, wrap that command in a batch script. Create `job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=<partition>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load <python_module>
source ~/envs/myenv/bin/activate

python train.py --lr 0.001 --epochs 50 --seed 42
```

What each part does:

| Line | Meaning |
|---|---|
| `#SBATCH --job-name=train` | Name shown in the queue |
| `#SBATCH --partition=...` | Which queue to run in |
| `#SBATCH --ntasks=1` | One process |
| `#SBATCH --cpus-per-task=4` | Four CPU cores for that process |
| `#SBATCH --mem=16G` | 16 GB of memory |
| `#SBATCH --time=02:00:00` | Kill the job after 2 hours (format `D-HH:MM:SS`) |
| `#SBATCH --output=...` | File for printed output (`%x` = job name, `%j` = job ID) |
| `#SBATCH --error=...` | File for error messages |
| `module load` / `source` | Set up Python the same way you would interactively |
| `python train.py ...` | Your actual command |

Two rules to remember:

1. **All `#SBATCH` lines must come before the first command.** Anything after is ignored.
2. **The `logs/` folder must exist before you submit**, or the job will fail without producing output.

---

## 2. Submit It

```bash
mkdir -p logs
sbatch job.sh
```

SLURM replies with a job ID:

```
Submitted batch job 123456
```

Keep that number. You'll use it for everything below.

### Passing different arguments without editing the file

You can pass values to the script on the `sbatch` line and read them as `$1`, `$2`, `$3`:

```bash
# inside job.sh, replace the last line with:
python train.py --lr $1 --epochs $2 --seed $3
```

```bash
sbatch job.sh 0.001 50 42
sbatch job.sh 0.01  50 42
```

---

## 3. Track Progress

### Is it queued or running?

```bash
squeue -u $USER
```

Output looks like:

```
JOBID   PARTITION  NAME   USER  ST  TIME  NODES  NODELIST(REASON)
123456  <part>     train  you   PD  0:00  1      (Priority)
```

| `ST` | Meaning |
|---|---|
| `PD` | Pending (waiting in line) |
| `R` | Running |
| `CG` | Completing |

If the job is pending, the reason is in parentheses:

| Reason | Meaning |
|---|---|
| `Priority` | Higher-priority jobs are ahead of you |
| `Resources` | You're next, waiting for nodes to free up |
| `Dependency` | Waiting on another job |

To see when SLURM expects it to start:

```bash
squeue -u $USER --start
```

### Watch the output live

Once the job is running, the log file fills in as your script prints:

```bash
tail -f logs/train_123456.out
```

Press `Ctrl+C` to stop watching. This does not stop the job.

For output to appear promptly, make sure your script flushes its prints (`print(..., flush=True)` or run `python -u train.py ...`), since Python buffers output when writing to a file.

### Detailed job info

```bash
scontrol show job 123456
```

This shows the node, requested resources, working directory, and start time.

### Cancel a job

```bash
scancel 123456          # one job
scancel -u $USER        # all of your jobs
```

---

## 4. After It Finishes: Did It Use What You Asked For?

```bash
sacct -j 123456 --format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,ReqMem,ExitCode
seff 123456             # efficiency summary, if installed
```

Use this to right-size future requests:

- `MaxRSS` vs. `ReqMem`: how much memory it actually needed. Request that plus ~20%.
- `Elapsed` vs. your `--time`: how long it really took. Request that plus a buffer.
- `State`: `COMPLETED`, `FAILED`, `TIMEOUT`, or `OUT_OF_MEMORY`.

Smaller, tighter requests generally start sooner, and over-requesting can lower your fair-share priority.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No log file appears | `logs/` folder missing | `mkdir -p logs`, resubmit |
| `TIMEOUT` | Hit `--time` | Increase time, or save checkpoints |
| `OUT_OF_MEMORY` | Exceeded `--mem` | Increase `--mem` |
| `invalid partition` | Wrong partition name | Check with `sinfo` |
| `ModuleNotFoundError` | Environment not activated | Activate it inside the script |
| Job pending a long time | Request is large or the queue is busy | Reduce resources, check `sprio -j <id>` |
| Log is empty while running | Python output buffering | Use `python -u` or `flush=True` |

---

## 6. Quick Reference

```bash
sbatch job.sh                 # submit
squeue -u $USER               # list your jobs
squeue -u $USER --start       # estimated start times
tail -f logs/<name>_<id>.out  # watch output
scontrol show job <id>        # job details
scancel <id>                  # cancel
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ReqMem   # post-run stats
sinfo                         # partitions and node states
```