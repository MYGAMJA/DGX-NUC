# Hylion v6 backend-separated training

This note defines separate launchers for v6 PhysX and v6 Newton training.

## Policy

- This DGX machine: Newton-only
- External desktop: PhysX-only

Convenience launchers:
- DGX Newton default: `/home/laba/project_singularity/δ3/scripts/train_hylion_v6_here.sh`
- Desktop PhysX default: `/home/laba/project_singularity/δ3/scripts/train_hylion_v6_desktop_physx.sh`

## 1) PhysX-only v6

Launcher:
- /home/laba/project_singularity/δ3/scripts/train_hylion_v6_physx.sh

Run:
- bash /home/laba/project_singularity/δ3/scripts/train_hylion_v6_physx.sh

Log:
- /tmp/hylion_v6_physx_train.log

Task and script:
- task: Velocity-Hylion-BG-v0
- python: /home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py

## 2) Newton-only v6

Launcher:
- /home/laba/project_singularity/δ3/scripts/train_hylion_v6_newton.sh

Run:
- bash /home/laba/project_singularity/δ3/scripts/train_hylion_v6_newton.sh

Log:
- /tmp/hylion_v6_newton_train.log

Task and script:
- task: Velocity-Hylion-v0
- python: /home/laba/project_singularity/δ3/scripts/train_hylion.py

## 3) Quick checks

- PhysX log tail:
  tail -f /tmp/hylion_v6_physx_train.log
- Newton log tail:
  tail -f /tmp/hylion_v6_newton_train.log
- Running process check:
  ps -ef | grep -E "train_hylion_physx_BG.py|train_hylion.py" | grep -v grep
