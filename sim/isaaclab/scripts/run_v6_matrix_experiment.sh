#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_v6_matrix_experiment.sh M1
#   bash run_v6_matrix_experiment.sh M4

EXP_ID="${1:-M1}"

case "$EXP_ID" in
  M1) MASS_ADD_KG="0.0"; LEG_GAIN_SCALE="1.0"; FEET_AIR_THRESHOLD="0.2" ;;
  M2) MASS_ADD_KG="0.0"; LEG_GAIN_SCALE="1.2"; FEET_AIR_THRESHOLD="0.2" ;;
  M3) MASS_ADD_KG="0.5"; LEG_GAIN_SCALE="1.0"; FEET_AIR_THRESHOLD="0.2" ;;
  M4) MASS_ADD_KG="0.5"; LEG_GAIN_SCALE="1.2"; FEET_AIR_THRESHOLD="0.2" ;;
  M5) MASS_ADD_KG="1.0"; LEG_GAIN_SCALE="1.2"; FEET_AIR_THRESHOLD="0.2" ;;
  M6) MASS_ADD_KG="1.5"; LEG_GAIN_SCALE="1.2"; FEET_AIR_THRESHOLD="0.2" ;;
  *) echo "Unknown experiment id: $EXP_ID"; exit 1 ;;
esac

LOG_FILE="/tmp/hylion_v6_physx_${EXP_ID}.log"

echo "[INFO] Starting $EXP_ID"
echo "[INFO] HYLION_BASE_MASS_ADD_KG=$MASS_ADD_KG"
echo "[INFO] HYLION_LEG_GAIN_SCALE=$LEG_GAIN_SCALE"
echo "[INFO] HYLION_FEET_AIR_THRESHOLD=$FEET_AIR_THRESHOLD"
echo "[INFO] log=$LOG_FILE"

# Stop any existing Stage-B run.
pids=$(pgrep -f "train_hylion_physx_BG.py --task Velocity-Hylion-BG-v0" || true)
if [[ -n "$pids" ]]; then
  echo "[INFO] Stopping existing Stage-B process(es): $pids"
  kill $pids
fi

cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
source /home/laba/env_isaaclab/bin/activate
unset PYTHONPATH PYTHONHOME
export PYTHONUNBUFFERED=1
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"

export HYLION_BASE_MASS_ADD_KG="$MASS_ADD_KG"
export HYLION_LEG_GAIN_SCALE="$LEG_GAIN_SCALE"
export HYLION_FEET_AIR_THRESHOLD="$FEET_AIR_THRESHOLD"

nohup /home/laba/env_isaaclab/bin/python /home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py \
  --task Velocity-Hylion-BG-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 6250 \
  --hylion_usd_path "/home/laba/project_singularity/δ1 & ε2/usd/hylion_v6/hylion_v6.usda" \
  --pretrained_checkpoint /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/biped/2026-04-06_15-27-27/model_5999.pt \
  > "$LOG_FILE" 2>&1 &

echo "[INFO] started pid=$!"
echo "[INFO] monitor: bash /home/laba/project_singularity/δ3/scripts/monitor_stageb_realtime.sh $LOG_FILE 2"
