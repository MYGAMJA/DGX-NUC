# Desktop PhysX Quickstart (v6)

Use this on the external desktop machine where PhysX-only training will run.

## 1) Clone project

```bash
git clone <your-project-repo-url> ~/project_singularity
cd ~/project_singularity
```

## 2) Environment setup (one-time)

```bash
bash ~/project_singularity/δ3/scripts/setup_desktop_physx_env.sh
```

Optional path overrides:

```bash
PROJECT_ROOT=~/project_singularity \
VENV_PATH=~/env_isaaclab \
ISAACLAB_ROOT=~/IsaacLab \
BHL_ROOT=~/Berkeley-Humanoid-Lite \
PYTHON_BIN=python3.12 \
bash ~/project_singularity/δ3/scripts/setup_desktop_physx_env.sh
```

## 3) Start training

```bash
bash ~/project_singularity/δ3/scripts/train_hylion_v6_desktop_physx.sh
```

Or with overrides:

```bash
NUM_ENVS=2048 \
MAX_ITERS=4000 \
LOG_FILE=/tmp/hylion_v6_physx_desktop.log \
bash ~/project_singularity/δ3/scripts/train_hylion_v6_desktop_physx.sh
```

## 4) Monitor

```bash
tail -f /tmp/hylion_v6_physx_desktop.log
ps -ef | grep train_hylion_physx_BG.py | grep -v grep
```

## 5) Notes

- This desktop track is PhysX-only.
- This DGX machine remains Newton-only.
- If your desktop paths differ, use PROJECT_ROOT/BHL_ROOT/VENV_PATH overrides.
