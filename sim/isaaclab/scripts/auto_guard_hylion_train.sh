#!/usr/bin/env bash
set -euo pipefail

BHL_DIR="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl"
TRAIN_SCRIPT="/home/laba/project_singularity/δ3/scripts/train_hylion.py"
PYTHON_BIN="/home/laba/env_isaaclab/bin/python"
TRAIN_LOG="/home/laba/project_singularity/δ3/hylion_v4_retrain_stable.log"
GUARD_LOG="/home/laba/project_singularity/δ3/hylion_guard.log"
STATE_FILE="/home/laba/project_singularity/δ3/.hylion_guard_state"

CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
NAN_THRESHOLD="${NAN_THRESHOLD:-8}"

CKPTS=(
  "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-01_22-25-55/model_6500.pt"
  "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-01_22-25-55/model_6400.pt"
  "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-01_22-25-55/model_6300.pt"
  "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/biped/2026-03-27_14-36-49/model_5999.pt"
)

LRS=("2e-5" "1.5e-5" "1e-5" "5e-5")
ENTS=("0.010" "0.012" "0.014" "0.008")

log_msg() {
  echo "[$(date '+%F %T')] $*" | tee -a "$GUARD_LOG"
}

read_stage() {
  if [[ -f "$STATE_FILE" ]]; then
    cat "$STATE_FILE"
  else
    echo 0
  fi
}

write_stage() {
  echo "$1" > "$STATE_FILE"
}

collapse_detected() {
  [[ -f "$TRAIN_LOG" ]] || return 1
  local recent
  recent="$(tail -n 500 "$TRAIN_LOG" || true)"

  local nan_loss_count std_zero_count
  nan_loss_count=$(printf "%s" "$recent" | grep -cE "Mean value loss: nan|Mean surrogate loss: nan" || true)
  std_zero_count=$(printf "%s" "$recent" | grep -cE "Mean action std: 0\.00" || true)

  if (( nan_loss_count >= NAN_THRESHOLD )) || (( std_zero_count >= NAN_THRESHOLD )); then
    log_msg "collapse signature detected (nan_loss=${nan_loss_count}, std_zero=${std_zero_count})"
    return 0
  fi
  return 1
}

launch_train() {
  local stage="$1"
  local ckpt="${CKPTS[$stage]}"
  local lr="${LRS[$stage]}"
  local ent="${ENTS[$stage]}"

  if [[ ! -f "$ckpt" ]]; then
    log_msg "checkpoint missing: $ckpt"
    return 1
  fi

  cd "$BHL_DIR"
  source /home/laba/env_isaaclab/bin/activate
  unset PYTHONPATH
  unset PYTHONHOME

  log_msg "launching stage=${stage} ckpt=${ckpt} lr=${lr} ent=${ent}"
  PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" nohup "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --task Velocity-Hylion-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 6000 \
    --stable_walk \
    --stable_walk_lr "$lr" \
    --stable_walk_entropy "$ent" \
    --stable_walk_schedule fixed \
    --pretrained_checkpoint "$ckpt" \
    > "$TRAIN_LOG" 2>&1 &

  log_msg "launched train pid=$!"
}

ensure_single_guard() {
  local lock_file="/home/laba/project_singularity/δ3/.hylion_guard.lock"
  exec 9>"$lock_file"
  if ! flock -n 9; then
    log_msg "another guard is already running; exiting"
    exit 0
  fi
}

main() {
  ensure_single_guard
  touch "$GUARD_LOG"

  local stage
  stage=$(read_stage)
  if ! [[ "$stage" =~ ^[0-9]+$ ]]; then
    stage=0
  fi
  if (( stage < 0 || stage >= ${#CKPTS[@]} )); then
    stage=0
  fi

  log_msg "guard start (stage=${stage}, interval=${CHECK_INTERVAL}s)"

  while true; do
    if ! pgrep -af "train_hylion.py" >/dev/null 2>&1; then
      log_msg "train process not found; launching"
      launch_train "$stage" || true
      sleep 60
      continue
    fi

    if collapse_detected; then
      log_msg "restarting after collapse"
      pkill -f "train_hylion.py" || true
      sleep 3

      if (( stage + 1 < ${#CKPTS[@]} )); then
        stage=$((stage + 1))
      fi
      write_stage "$stage"
      launch_train "$stage" || true
      sleep 60
      continue
    fi

    sleep "$CHECK_INTERVAL"
  done
}

main "$@"
