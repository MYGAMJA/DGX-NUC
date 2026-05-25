#!/usr/bin/env bash
# =============================================================================
# run_demo_training.sh
# Hylion biped — 발표 시나리오 단일 학습 (2026-05-01)
#
# 시나리오:
#   1. 야외 보도블록 평지 보행 (vx ≤ 0.5 m/s)
#   2. 제자리 서서 SO-ARM으로 물건 집기 (mass 변동 + 정지)
#
# BHL 본가 정공법 + 우리 시나리오 맞춤 변경:
#   - PPO: BHL 그대로 (lr=1e-3, adaptive, max_grad_norm=1.0, epochs=5,
#          init_noise_std=0.5)  ← std만 D4 호환 위해 0.5로
#   - 외란: ±5N interval 0.5~2s (블록 단차 충격 시뮬)
#   - mass: -1.0 ~ +3.0 kg 랜덤 (빈 손 ~ 1.5kg 물건 잡은 상태)
#   - standing 비율: 20% (정지 + 팔 조작 학습)
#   - velocity: ±0.5 m/s
#   - curriculum 없음 — 단일 학습 8000 iter (~5~6시간)
#   - 출발점: stage_d4_hylion_v6/best.pt (검증된 안정 정책)
#
# 사용법:
#   nohup bash sim/isaaclab/scripts/newton/run_demo_training.sh \
#     > /tmp/hylion_demo_orchestrator.log 2>&1 &
#
#   tail -f /tmp/hylion_demo_progress.log
# =============================================================================

set -uo pipefail

# ── 경로 ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
BHL_DIR="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl"
TRAIN_SCRIPT="${REPO_ROOT}/sim/isaaclab/scripts/newton/train_hylion_newton_BG.py"
PYTHON_BIN="/home/laba/env_isaaclab/bin/python"
CKPT_DIR="${REPO_ROOT}/checkpoints/biped"
MUJOCO_SCRIPT="${REPO_ROOT}/sim/mujoco/play_mujoco.py"
MUJOCO_MJCF="${REPO_ROOT}/sim/isaaclab/robot/hylion_v6.xml"
PROGRESS_LOG="/tmp/hylion_demo_progress.log"
TRAINING_LOG_MD="${REPO_ROOT}/docs/jimmy/training_log_2026-04-30.md"
LOCK_FILE="/tmp/hylion_demo_orchestrator.lock"

# ── 학습 파라미터 (시나리오 맞춤) ──────────────────────────────────────────────
PERTURB_FORCE="${PERTURB_FORCE:-5.0}"            # 단차/마찰 변동 시뮬 (작음)
PERTURB_TORQUE="${PERTURB_TORQUE:-2.0}"          # 회전 외란
PERTURB_INTERVAL_MIN="${PERTURB_INTERVAL_MIN:-0.5}"
PERTURB_INTERVAL_MAX="${PERTURB_INTERVAL_MAX:-2.0}"
MASS_RANGE_LO="${MASS_RANGE_LO:--1.0}"           # 빈 손
MASS_RANGE_HI="${MASS_RANGE_HI:-3.0}"            # 1.5kg 물건 들고 (양 팔 합쳐 +3kg)
STANDING_RATIO="${STANDING_RATIO:-0.2}"          # 20% 정지 명령
MAX_VEL="${MAX_VEL:-0.5}"
MAX_ITERATIONS="${MAX_ITERATIONS:-8000}"
START_CKPT="${START_CKPT:-${CKPT_DIR}/stage_d4_hylion_v6/best.pt}"
OUT_NAME="${OUT_NAME:-stage_demo_hylion_v6}"
TASK_ID="${TASK_ID:-Velocity-Hylion-BG-Generic-v0}"

# ── lock ──────────────────────────────────────────────────────────────────────
if [[ -f "$LOCK_FILE" ]]; then
    existing_pid=$(cat "$LOCK_FILE" 2>/dev/null)
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "[ERROR] 다른 인스턴스 실행 중 (PID $existing_pid)"
        exit 99
    fi
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"

cleanup() {
    rm -f "$LOCK_FILE"
    pkill -P $$ 2>/dev/null || true
    pkill -f "train_hylion_newton_BG" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 진행 알림 ─────────────────────────────────────────────────────────────────
notify() {
    local level="$1"; shift
    local msg="[$(date '+%F %T')] [${level}] $*"
    echo "$msg" | tee -a "$PROGRESS_LOG"
}

# ── 메트릭 추출 ────────────────────────────────────────────────────────────────
extract_fall_avg() {
    local logfile="$1" lines="$2"
    grep "base_orientation:" "$logfile" 2>/dev/null | awk '{print $NF}' \
      | tail -"$lines" | awk '{s+=$1; c++} END {if(c>0) printf "%.4f", s/c; else print "1.0"}'
}

extract_reward_avg() {
    local logfile="$1" lines="$2"
    grep "Mean reward:" "$logfile" 2>/dev/null | awk '{print $NF}' \
      | tail -"$lines" | awk '{s+=$1; c++} END {if(c>0) printf "%.2f", s/c; else print "0"}'
}

# ── MuJoCo 검증 ────────────────────────────────────────────────────────────────
mujoco_check() {
    local ckpt="$1" vx="$2" force="$3" torque="$4"
    local result
    result=$(timeout 90 "$PYTHON_BIN" "$MUJOCO_SCRIPT" \
        --ckpt "$ckpt" --mjcf "$MUJOCO_MJCF" \
        --no-viewer --vx "$vx" --duration 10.0 \
        --external-push "$force" --external-torque "$torque" \
        2>&1 | grep -oP 'Survived \K[0-9]+' | tail -1 || echo "0")
    echo "${result:-0}"
}

# ── MD 갱신 (시나리오 결과 기록) ───────────────────────────────────────────────
update_md() {
    local ckpt="$1" reward="$2" fall="$3" mj_walk="$4" mj_walk_p="$5" mj_stand="$6" mj_stand_p="$7" status="$8"
    if ! grep -q "## 시도 #4 — 발표 시나리오 학습" "$TRAINING_LOG_MD" 2>/dev/null; then
        cat >> "$TRAINING_LOG_MD" <<EOF

---

## 시도 #4 — 발표 시나리오 학습 (단일, BHL 정공법)

설정: BHL PPO + init_noise_std=0.5 + 외란 ±${PERTURB_FORCE}N (interval ${PERTURB_INTERVAL_MIN}~${PERTURB_INTERVAL_MAX}s) + mass(${MASS_RANGE_LO}~${MASS_RANGE_HI}kg) + standing ${STANDING_RATIO}

| 시각 | iter | reward | fall % | MuJoCo 보행 무외란 | MuJoCo 보행 ±${PERTURB_FORCE}N | MuJoCo 정지 무외란 | MuJoCo 정지 ±${PERTURB_FORCE}N | 결과 |
|------|-----:|-------:|-------:|:------------------:|:-----------------------------:|:------------------:|:-----------------------------:|:----:|
EOF
    fi
    local row
    row=$(printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
          "$(date '+%m-%d %H:%M')" "$MAX_ITERATIONS" \
          "${reward}" "${fall}" \
          "${mj_walk}/250" "${mj_walk_p}/250" \
          "${mj_stand}/250" "${mj_stand_p}/250" "$status")
    echo "$row" >> "$TRAINING_LOG_MD"
    notify INFO "  → ${TRAINING_LOG_MD} 갱신"
}

# ── 학습 실행 ─────────────────────────────────────────────────────────────────
notify INFO "================================================================"
notify INFO "Hylion 발표 시나리오 단일 학습 시작"
notify INFO "================================================================"
notify INFO "  외란       : ±${PERTURB_FORCE}N / ±${PERTURB_TORQUE}Nm interval ${PERTURB_INTERVAL_MIN}~${PERTURB_INTERVAL_MAX}s"
notify INFO "  mass 범위  : ${MASS_RANGE_LO} ~ ${MASS_RANGE_HI} kg"
notify INFO "  velocity   : ±${MAX_VEL} m/s, standing ${STANDING_RATIO}"
notify INFO "  iter       : ${MAX_ITERATIONS}"
notify INFO "  출발점     : ${START_CKPT}"
notify INFO "  출력       : ${CKPT_DIR}/${OUT_NAME}/best.pt"

if [[ ! -f "$START_CKPT" ]]; then
    notify FATAL "출발 체크포인트 없음: $START_CKPT"
    exit 1
fi

mkdir -p "${CKPT_DIR}/${OUT_NAME}"
[[ -f "${CKPT_DIR}/${OUT_NAME}/best.pt" ]] && \
    cp "${CKPT_DIR}/${OUT_NAME}/best.pt" "${CKPT_DIR}/${OUT_NAME}/best.pt.prev_$(date +%H%M)"

LOG_FILE="/tmp/hylion_demo_train.log"
notify INFO "  학습 로그  : ${LOG_FILE}"

# 학습 시작 marker
MARKER_FILE="/tmp/hylion_demo_train_marker"
touch "$MARKER_FILE"
sleep 1

# 진행 모니터
(
    prev_iter=0
    for _ in $(seq 1 200); do
        sleep 600
        [[ -f "$LOG_FILE" ]] || break
        cur_iter=$(grep -oP "Learning iteration \K[0-9]+" "$LOG_FILE" 2>/dev/null | tail -1)
        cur_reward=$(extract_reward_avg "$LOG_FILE" 50)
        cur_fall=$(extract_fall_avg "$LOG_FILE" 50)
        if [[ -n "$cur_iter" && "$cur_iter" != "$prev_iter" ]]; then
            notify INFO "  [진행] iter=${cur_iter} reward=${cur_reward} fall=${cur_fall}"
            prev_iter="$cur_iter"
        fi
    done
) &
disown $! 2>/dev/null || true
MONITOR_PID=$!

cd "$BHL_DIR" || exit 1
source /home/laba/env_isaaclab/bin/activate 2>/dev/null
unset PYTHONPATH PYTHONHOME 2>/dev/null

export HYLION_ENABLE_PERTURBATION=1
export HYLION_PERTURB_FORCE="$PERTURB_FORCE"
export HYLION_PERTURB_TORQUE="$PERTURB_TORQUE"
export HYLION_PERTURB_INTERVAL_MIN="$PERTURB_INTERVAL_MIN"
export HYLION_PERTURB_INTERVAL_MAX="$PERTURB_INTERVAL_MAX"
export HYLION_BASE_MASS_RANGE_LO="$MASS_RANGE_LO"
export HYLION_BASE_MASS_RANGE_HI="$MASS_RANGE_HI"
export HYLION_BASE_MASS_ADD_KG=0.0  # range 모드 우선
export HYLION_MAX_LIN_VEL_X="$MAX_VEL"
export HYLION_STANDING_RATIO="$STANDING_RATIO"

PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --num_envs 4096 --headless \
    --task "$TASK_ID" \
    --pretrained_checkpoint "$START_CKPT" \
    --max_iterations "$MAX_ITERATIONS" \
    2>&1 | tee "$LOG_FILE"
TRAIN_RC=${PIPESTATUS[0]}

kill "$MONITOR_PID" 2>/dev/null || true

if [[ $TRAIN_RC -ne 0 ]]; then
    notify FATAL "학습 비정상 종료 (rc=${TRAIN_RC})"
    rm -f "$MARKER_FILE"
    exit 1
fi

# 새 결과 디렉터리 찾기 (marker 이후 생성된 것만)
result_dir=$(find "${BHL_DIR}/logs/rsl_rl/hylion/" -maxdepth 1 -type d -newer "$MARKER_FILE" 2>/dev/null | sort | tail -1)
rm -f "$MARKER_FILE"
if [[ -z "$result_dir" || ! -d "$result_dir" ]]; then
    notify FATAL "학습 결과 디렉터리 못 찾음"
    exit 1
fi
latest_pt=$(ls "${result_dir}"/model_*.pt 2>/dev/null | sort -V | tail -1)
if [[ -z "$latest_pt" ]]; then
    notify FATAL "model_*.pt 못 찾음"
    exit 1
fi
pt_iter=$(echo "$latest_pt" | grep -oP 'model_\K[0-9]+')
if [[ "$pt_iter" -lt $((MAX_ITERATIONS / 2)) ]]; then
    notify FATAL "iter 부족 (${pt_iter} < $((MAX_ITERATIONS / 2)))"
    exit 1
fi
cp "$latest_pt" "${CKPT_DIR}/${OUT_NAME}/best.pt"
notify INFO "  체크포인트 저장: ${CKPT_DIR}/${OUT_NAME}/best.pt (model_${pt_iter}.pt)"

# ── 평가 (4가지 시나리오) ─────────────────────────────────────────────────────
notify INFO "================================================================"
notify INFO "발표 시나리오 평가"
notify INFO "================================================================"

CKPT="${CKPT_DIR}/${OUT_NAME}/best.pt"
total_iter=$(grep -c "Mean reward:" "$LOG_FILE" 2>/dev/null || echo 0)
fall_avg=$(extract_fall_avg "$LOG_FILE" 200)
reward_avg=$(extract_reward_avg "$LOG_FILE" 200)
notify INFO "  학습 완료 iter: ${total_iter}"
notify INFO "  마지막 200 reward: ${reward_avg}"
notify INFO "  마지막 200 fall  : ${fall_avg}"

notify INFO "  [1] 평지 보행 무외란 (vx=0.3)"
mj_walk=$(mujoco_check "$CKPT" 0.3 0 0)
notify INFO "      → ${mj_walk}/250 (기준 ≥ 230)"

notify INFO "  [2] 평지 보행 + 단차 외란 (vx=0.3, ±${PERTURB_FORCE}N)"
mj_walk_p=$(mujoco_check "$CKPT" 0.3 "$PERTURB_FORCE" "$PERTURB_TORQUE")
notify INFO "      → ${mj_walk_p}/250 (기준 ≥ 200)"

notify INFO "  [3] 제자리 서기 무외란 (vx=0)"
mj_stand=$(mujoco_check "$CKPT" 0.0 0 0)
notify INFO "      → ${mj_stand}/250 (기준 ≥ 230)"

notify INFO "  [4] 제자리 서기 + 외란 (vx=0, ±${PERTURB_FORCE}N)"
mj_stand_p=$(mujoco_check "$CKPT" 0.0 "$PERTURB_FORCE" "$PERTURB_TORQUE")
notify INFO "      → ${mj_stand_p}/250 (기준 ≥ 200)"

# ── 종합 판정 ─────────────────────────────────────────────────────────────────
pass=0
[[ "$mj_walk" -ge 230 ]] && ((pass++))
[[ "$mj_walk_p" -ge 200 ]] && ((pass++))
[[ "$mj_stand" -ge 230 ]] && ((pass++))
[[ "$mj_stand_p" -ge 200 ]] && ((pass++))

if (( pass >= 3 )); then
    status="✅ PASS (${pass}/4)"
    notify PASS "================================================================"
    notify PASS "✅ 발표 시나리오 학습 성공 — ${pass}/4 통과"
    notify PASS "================================================================"
else
    status="⚠ FAIL (${pass}/4)"
    notify FAIL "================================================================"
    notify FAIL "⚠ 발표 시나리오 학습 실패 — ${pass}/4 통과 (기준: 3+)"
    notify FAIL "================================================================"
fi

update_md "$CKPT" "$reward_avg" "$fall_avg" "$mj_walk" "$mj_walk_p" "$mj_stand" "$mj_stand_p" "$status"

notify INFO "최종 체크포인트: ${CKPT}"
notify INFO "재실행 시 다른 파라미터 변경: PERTURB_FORCE=N MAX_ITERATIONS=N bash $0"
