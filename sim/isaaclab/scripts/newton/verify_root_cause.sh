#!/usr/bin/env bash
# =============================================================================
# verify_root_cause.sh
# 학습 실패 근본 원인 검증 (2026-05-02)
#
# 가설: 외란 interval 모드 + 명시적 외란 회복 보상 부재 → catastrophic forgetting
#       (4번 실패 모두 interval 모드 사용)
#
# 검증 실험:
#   실험 1: D4 직접 평가 — 추가 학습 자체가 필요한지 확인 (학습 0)
#   실험 2: 외란 reset 모드 (BHL 정공법) — interval 모드가 진짜 원인인지
#   실험 3: 외란 0 (도메인 랜덤화만, BHL 가설 끝까지) — 외란 자체가 문제인지
#
# 사용법:
#   nohup bash sim/isaaclab/scripts/newton/verify_root_cause.sh \
#     > /tmp/hylion_verify.log 2>&1 &
#   tail -f /tmp/hylion_verify_progress.log
#
# 총 예상 시간: 실험 1 (~10분) + 실험 2 (~5h) + 실험 3 (~5h) ≈ 10시간
# =============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
BHL_DIR="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl"
TRAIN_SCRIPT="${REPO_ROOT}/sim/isaaclab/scripts/newton/train_hylion_newton_BG.py"
PYTHON_BIN="/home/laba/env_isaaclab/bin/python"
CKPT_DIR="${REPO_ROOT}/checkpoints/biped"
MUJOCO_SCRIPT="${REPO_ROOT}/sim/mujoco/play_mujoco.py"
MUJOCO_MJCF="${REPO_ROOT}/sim/isaaclab/robot/hylion_v6.xml"
PROGRESS_LOG="/tmp/hylion_verify_progress.log"
REPORT_MD="${REPO_ROOT}/docs/jimmy/root_cause_verification_2026-05-02.md"
LOCK_FILE="/tmp/hylion_verify.lock"
TASK_ID="Velocity-Hylion-BG-Generic-v0"

D4_CKPT="${CKPT_DIR}/stage_d4_hylion_v6/best.pt"

# 검증 시간 (학습은 6000 iter — BHL 동일)
EXP_MAX_ITER=6000
EXP_NUM_ENVS=4096

# ── 동시 실행 방지 ─────────────────────────────────────────────────────────────
if [[ -f "$LOCK_FILE" ]]; then
    pid=$(cat "$LOCK_FILE" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[ERROR] 다른 인스턴스 실행 중 (PID $pid)"
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

# ── 알림 ──────────────────────────────────────────────────────────────────────
notify() {
    local level="$1"; shift
    echo "[$(date '+%F %T')] [${level}] $*" | tee -a "$PROGRESS_LOG"
}

section() {
    {
        echo ""
        echo "================================================================"
        echo "  $1"
        echo "  $(date '+%F %T')"
        echo "================================================================"
    } | tee -a "$PROGRESS_LOG"
}

# ── 메트릭 추출 ────────────────────────────────────────────────────────────────
get_fall() { grep "base_orientation:" "$1" 2>/dev/null | awk '{print $NF}' | tail -"$2" | awk '{s+=$1;c++} END {if(c>0) printf "%.4f", s/c; else print "1.0"}'; }
get_reward() { grep "Mean reward:" "$1" 2>/dev/null | awk '{print $NF}' | tail -"$2" | awk '{s+=$1;c++} END {if(c>0) printf "%.2f", s/c; else print "0"}'; }
get_iter() { grep -c "Mean reward:" "$1" 2>/dev/null || echo 0; }

# ── MuJoCo 검증 ────────────────────────────────────────────────────────────────
mj_test() {
    local ckpt="$1" vx="$2" force="$3" torque="$4" extra_mass="$5"
    local result
    result=$(timeout 90 "$PYTHON_BIN" "$MUJOCO_SCRIPT" \
        --ckpt "$ckpt" --mjcf "$MUJOCO_MJCF" \
        --no-viewer --vx "$vx" --duration 10.0 \
        --external-push "$force" --external-torque "$torque" \
        --extra-base-mass "$extra_mass" \
        2>&1 | grep -oP 'Survived \K[0-9]+' | tail -1 || echo "0")
    echo "${result:-0}"
}

# ── 학습 실행 ─────────────────────────────────────────────────────────────────
run_experiment_training() {
    local exp_name="$1"   # exp2 or exp3
    local out_dir_name="$2"
    shift 2
    local extra_envs=("$@")  # KEY=VALUE 형식

    notify INFO "Training 시작: $exp_name → $out_dir_name"

    local out_dir="${CKPT_DIR}/${out_dir_name}"
    mkdir -p "$out_dir"
    [[ -f "$out_dir/best.pt" ]] && cp "$out_dir/best.pt" "$out_dir/best.pt.prev_$(date +%H%M)"

    local logfile="/tmp/hylion_verify_${exp_name}_train.log"
    local marker="/tmp/hylion_verify_${exp_name}_marker"
    touch "$marker"
    sleep 1

    # 진행 모니터
    (
        prev=0
        for _ in $(seq 1 80); do
            sleep 600
            [[ -f "$logfile" ]] || break
            cur=$(grep -oP "Learning iteration \K[0-9]+" "$logfile" 2>/dev/null | tail -1)
            r=$(get_reward "$logfile" 50)
            f=$(get_fall "$logfile" 50)
            if [[ -n "$cur" && "$cur" != "$prev" ]]; then
                notify INFO "  [$exp_name] iter=$cur reward=$r fall=$f"
                prev="$cur"
            fi
        done
    ) &
    local mon_pid=$!
    disown $mon_pid 2>/dev/null || true

    cd "$BHL_DIR" || return 1
    source /home/laba/env_isaaclab/bin/activate 2>/dev/null
    unset PYTHONPATH PYTHONHOME 2>/dev/null

    # 공통 환경변수 (모든 실험)
    export HYLION_BASE_MASS_RANGE_LO=-1.0
    export HYLION_BASE_MASS_RANGE_HI=3.0
    export HYLION_BASE_MASS_ADD_KG=0.0
    export HYLION_MAX_LIN_VEL_X=0.5
    export HYLION_STANDING_RATIO=0.2

    # 실험별 추가 환경변수
    for ev in "${extra_envs[@]}"; do
        [[ -n "$ev" ]] && export "$ev"
    done

    PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
        "$PYTHON_BIN" "$TRAIN_SCRIPT" \
        --num_envs "$EXP_NUM_ENVS" --headless \
        --task "$TASK_ID" \
        --pretrained_checkpoint "$D4_CKPT" \
        --max_iterations "$EXP_MAX_ITER" \
        2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}

    kill "$mon_pid" 2>/dev/null || true

    if [[ $rc -ne 0 ]]; then
        notify FATAL "$exp_name 학습 비정상 종료 (rc=$rc)"
        rm -f "$marker"
        return 1
    fi

    # 새 결과 디렉터리 (marker 이후)
    local result_dir
    result_dir=$(find "${BHL_DIR}/logs/rsl_rl/hylion/" -maxdepth 1 -type d -newer "$marker" 2>/dev/null | sort | tail -1)
    rm -f "$marker"
    [[ -z "$result_dir" || ! -d "$result_dir" ]] && { notify FATAL "$exp_name 결과 디렉터리 없음"; return 1; }
    local latest_pt
    latest_pt=$(ls "$result_dir"/model_*.pt 2>/dev/null | sort -V | tail -1)
    [[ -z "$latest_pt" ]] && { notify FATAL "$exp_name 체크포인트 없음"; return 1; }
    cp "$latest_pt" "$out_dir/best.pt"
    notify INFO "$exp_name 학습 완료. 체크포인트: $out_dir/best.pt"

    # 환경변수 정리
    unset HYLION_PERTURB_MODE HYLION_PERTURB_FORCE HYLION_PERTURB_TORQUE \
          HYLION_PERTURB_INTERVAL_MIN HYLION_PERTURB_INTERVAL_MAX \
          HYLION_ENABLE_PERTURBATION 2>/dev/null
    return 0
}

# ── 정책 평가 — 8가지 시나리오 (mass × push 조합) ────────────────────────────
evaluate_policy() {
    local label="$1" ckpt="$2"
    local results=()

    notify INFO "정책 평가 시작: $label"
    [[ ! -f "$ckpt" ]] && { notify FATAL "  체크포인트 없음: $ckpt"; echo ""; return 1; }

    # 시나리오: (vx, push, mass)
    # 평지 보행 / 정지 × 외란 0 또는 5 × mass 0,1,2,3
    declare -a scenarios=(
        "0.3 0 0|평지 보행 무외란"
        "0.3 5 0|평지 보행 ±5N"
        "0.0 0 0|정지 무외란"
        "0.0 5 0|정지 ±5N"
        "0.0 0 1|정지 +1kg"
        "0.0 0 2|정지 +2kg"
        "0.0 0 3|정지 +3kg"
        "0.3 0 2|보행 +2kg"
    )

    for entry in "${scenarios[@]}"; do
        local params="${entry%|*}"
        local desc="${entry#*|}"
        read -r vx push mass <<< "$params"
        local step
        step=$(mj_test "$ckpt" "$vx" "$push" "$([[ $push -gt 0 ]] && echo 2.0 || echo 0)" "$mass")
        notify INFO "  [$desc] (vx=$vx push=$push mass=+${mass}kg) → $step/250"
        results+=("$step")
    done

    # 결과를 콜론으로 join하여 반환
    local IFS=":"; echo "${results[*]}"
}

# ── 종합 보고서 (Markdown) ────────────────────────────────────────────────────
write_summary() {
    local exp1="$1" exp2_metrics="$2" exp2_eval="$3" exp3_metrics="$4" exp3_eval="$5"

    local IFS=":"
    local -a exp1_arr=($exp1)
    local -a exp2_eval_arr=($exp2_eval)
    local -a exp3_eval_arr=($exp3_eval)
    unset IFS

    cat > "$REPORT_MD" <<EOF
# 학습 실패 근본 원인 검증 보고서 (2026-05-02)

자동 검증 스크립트: \`sim/isaaclab/scripts/newton/verify_root_cause.sh\`
실행 시각: $(date '+%F %T')

## 가설

지금까지 4번의 학습 실패 (시도 #1~#4-B) 모두 **외란 interval 모드** 사용.
가설: **interval 모드가 catastrophic forgetting의 root cause**.

BHL 본가는 **reset 모드** (episode당 1회 외란)만 사용. 외란 robustness는 명시적으로 학습 안 하고
도메인 랜덤화 (mass, friction, gain) 으로 강건성 만듦.

## 검증 실험 설계

| # | 실험 | 외란 mode | 외란 force | 가설 검증 |
|---|------|----------|----------|----------|
| 1 | D4 직접 평가 | — (학습 X) | — | D4가 시나리오에 충분한가? |
| 2 | reset 모드 학습 | reset | ±2N (BHL 동일) | interval mode가 진짜 원인인가? |
| 3 | 외란 0 학습 | (외란 비활성) | 0 | 외란 자체가 문제인가? |

공통 설정 (BHL 정공법):
- PPO: lr=1e-3, adaptive, max_grad_norm=1.0, num_learning_epochs=5, init_noise_std=0.5
- mass: -1.0 ~ +3.0 kg startup randomization
- friction: 0.4 ~ 1.2 startup
- standing 비율: 20% (우리 시나리오)
- max_iterations: 6000 (BHL 동일)
- 출발점: stage_d4_hylion_v6/best.pt

## 평가 시나리오 (8가지, MuJoCo 무외란/외란 조합)

| # | 시나리오 | vx | push | mass +kg | 기준 |
|---|---------|----|------|----------|------|
| 1 | 평지 보행 무외란 | 0.3 | 0 | 0 | ≥230/250 |
| 2 | 평지 보행 ±5N | 0.3 | 5 | 0 | ≥200/250 |
| 3 | 정지 무외란 | 0.0 | 0 | 0 | ≥230/250 |
| 4 | 정지 ±5N | 0.0 | 5 | 0 | ≥200/250 |
| 5 | 정지 +1kg | 0.0 | 0 | 1 | ≥230/250 |
| 6 | 정지 +2kg | 0.0 | 0 | 2 | ≥230/250 |
| 7 | 정지 +3kg | 0.0 | 0 | 3 | ≥230/250 |
| 8 | 보행 +2kg | 0.3 | 0 | 2 | ≥230/250 |

---

## 결과

### 실험 1 — D4 정책 직접 평가

| 시나리오 | MuJoCo step |
|---------|:-----------:|
| 평지 보행 무외란 | ${exp1_arr[0]:-?}/250 |
| 평지 보행 ±5N | ${exp1_arr[1]:-?}/250 |
| 정지 무외란 | ${exp1_arr[2]:-?}/250 |
| 정지 ±5N | ${exp1_arr[3]:-?}/250 |
| 정지 +1kg | ${exp1_arr[4]:-?}/250 |
| 정지 +2kg | ${exp1_arr[5]:-?}/250 |
| 정지 +3kg | ${exp1_arr[6]:-?}/250 |
| 보행 +2kg | ${exp1_arr[7]:-?}/250 |

### 실험 2 — 외란 reset 모드 학습

학습 결과: $exp2_metrics

| 시나리오 | MuJoCo step |
|---------|:-----------:|
| 평지 보행 무외란 | ${exp2_eval_arr[0]:-?}/250 |
| 평지 보행 ±5N | ${exp2_eval_arr[1]:-?}/250 |
| 정지 무외란 | ${exp2_eval_arr[2]:-?}/250 |
| 정지 ±5N | ${exp2_eval_arr[3]:-?}/250 |
| 정지 +1kg | ${exp2_eval_arr[4]:-?}/250 |
| 정지 +2kg | ${exp2_eval_arr[5]:-?}/250 |
| 정지 +3kg | ${exp2_eval_arr[6]:-?}/250 |
| 보행 +2kg | ${exp2_eval_arr[7]:-?}/250 |

### 실험 3 — 외란 0, 도메인 랜덤화만

학습 결과: $exp3_metrics

| 시나리오 | MuJoCo step |
|---------|:-----------:|
| 평지 보행 무외란 | ${exp3_eval_arr[0]:-?}/250 |
| 평지 보행 ±5N | ${exp3_eval_arr[1]:-?}/250 |
| 정지 무외란 | ${exp3_eval_arr[2]:-?}/250 |
| 정지 ±5N | ${exp3_eval_arr[3]:-?}/250 |
| 정지 +1kg | ${exp3_eval_arr[4]:-?}/250 |
| 정지 +2kg | ${exp3_eval_arr[5]:-?}/250 |
| 정지 +3kg | ${exp3_eval_arr[6]:-?}/250 |
| 보행 +2kg | ${exp3_eval_arr[7]:-?}/250 |

---

## 결론 도출

(아래 매트릭스로 가설 검증)

| 가설 | 실험 1 결과 | 실험 2 결과 | 실험 3 결과 | 결론 |
|------|------------|------------|------------|------|
| H0: D4 충분 | 모두 통과 | — | — | 추가 학습 불필요 |
| H1: interval 모드 문제 | D4 일부 실패 | 잘 학습됨 | 잘 학습됨 | interval mode가 원인 → reset/0 권장 |
| H2: 외란 자체 문제 | D4 일부 실패 | 실패 | 잘 학습됨 | 외란 자체가 문제 → 외란 0 권장 |
| H3: 학습 자체가 어려움 | D4 일부 실패 | 실패 | 실패 | hylion 환경 자체 재검토 |

## 권장 다음 단계

(스크립트가 결과 매트릭스 기반으로 자동 결정)

EOF
    notify INFO "보고서 작성: $REPORT_MD"
}

# ============================================================================
# MAIN
# ============================================================================

section "Hylion 학습 실패 근본 원인 검증 시작"
notify INFO "예상 시간: 실험 1 (~10분) + 실험 2 (~5h) + 실험 3 (~5h) = 약 10시간"
notify INFO "보고서: $REPORT_MD"

# ── 실험 1: D4 직접 평가 ─────────────────────────────────────────────────────
section "실험 1: D4 정책 직접 평가 (학습 X)"
EXP1_RESULTS=$(evaluate_policy "D4 baseline" "$D4_CKPT")
notify INFO "실험 1 완료. 결과: $EXP1_RESULTS"

# ── 실험 2: 외란 reset 모드 (BHL 정공법) ──────────────────────────────────────
section "실험 2: 외란 reset 모드 학습 (BHL 정공법)"
notify INFO "  외란: ±2N reset mode (episode reset 시 1회만)"

run_experiment_training "exp2" "stage_exp2_reset_hylion_v6" \
    "HYLION_ENABLE_PERTURBATION=1" \
    "HYLION_PERTURB_FORCE=2.0" \
    "HYLION_PERTURB_TORQUE=2.0" \
    "HYLION_PERTURB_MODE=reset"
EXP2_RC=$?

if [[ $EXP2_RC -eq 0 ]]; then
    EXP2_LOG="/tmp/hylion_verify_exp2_train.log"
    EXP2_ITER=$(get_iter "$EXP2_LOG")
    EXP2_REWARD=$(get_reward "$EXP2_LOG" 200)
    EXP2_FALL=$(get_fall "$EXP2_LOG" 200)
    EXP2_METRICS="iter=$EXP2_ITER, reward=$EXP2_REWARD, fall=$EXP2_FALL"
    notify INFO "실험 2 학습 메트릭: $EXP2_METRICS"
    EXP2_EVAL=$(evaluate_policy "exp2 reset" "${CKPT_DIR}/stage_exp2_reset_hylion_v6/best.pt")
    notify INFO "실험 2 평가: $EXP2_EVAL"
else
    EXP2_METRICS="학습 실패"
    EXP2_EVAL="0:0:0:0:0:0:0:0"
fi

# ── 실험 3: 외란 0 (도메인 랜덤화만) ──────────────────────────────────────────
section "실험 3: 외란 0, 도메인 랜덤화만"
notify INFO "  외란 비활성, mass + friction + gain 만으로 학습"

run_experiment_training "exp3" "stage_exp3_no_perturb_hylion_v6" \
    "HYLION_ENABLE_PERTURBATION=0"
EXP3_RC=$?

if [[ $EXP3_RC -eq 0 ]]; then
    EXP3_LOG="/tmp/hylion_verify_exp3_train.log"
    EXP3_ITER=$(get_iter "$EXP3_LOG")
    EXP3_REWARD=$(get_reward "$EXP3_LOG" 200)
    EXP3_FALL=$(get_fall "$EXP3_LOG" 200)
    EXP3_METRICS="iter=$EXP3_ITER, reward=$EXP3_REWARD, fall=$EXP3_FALL"
    notify INFO "실험 3 학습 메트릭: $EXP3_METRICS"
    EXP3_EVAL=$(evaluate_policy "exp3 no_perturb" "${CKPT_DIR}/stage_exp3_no_perturb_hylion_v6/best.pt")
    notify INFO "실험 3 평가: $EXP3_EVAL"
else
    EXP3_METRICS="학습 실패"
    EXP3_EVAL="0:0:0:0:0:0:0:0"
fi

# ── 종합 보고서 ──────────────────────────────────────────────────────────────
section "종합 보고서 작성"
write_summary "$EXP1_RESULTS" "$EXP2_METRICS" "$EXP2_EVAL" "$EXP3_METRICS" "$EXP3_EVAL"

section "검증 완료"
notify INFO "보고서: $REPORT_MD"
notify INFO "결과 확인 후 다음 학습 전략 결정"
