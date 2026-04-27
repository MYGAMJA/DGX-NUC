#!/bin/bash
# Hylion v6 걷기 학습 — PhysX 백엔드 (BG)
# task: Velocity-Hylion-BG-v0
# 체크포인트: /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/<timestamp>/

set -e

BHL_SCRIPTS="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl"
TRAIN_SCRIPT="/home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py"

source /home/laba/env_isaaclab/bin/activate

cd "$BHL_SCRIPTS"

PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
  nohup python "$TRAIN_SCRIPT" \
    --task Velocity-Hylion-BG-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 6000 > /tmp/hylion_physx_BG_train.log 2>&1 &

echo "[INFO] 학습 시작 (PID: $!)"
echo "[INFO] 로그: tail -f /tmp/hylion_physx_BG_train.log"
echo "[INFO] 체크포인트: $BHL_SCRIPTS/logs/rsl_rl/hylion/"
