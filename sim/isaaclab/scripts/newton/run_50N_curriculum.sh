#!/usr/bin/env bash
# =============================================================================
# run_50N_curriculum.sh
# Hylion biped — D4 → ±50N robustness 자동 커리큘럼 학습 (2026-04-30 v2)
#
# 4-게이트 통과 시 다음 스테이지 진행:
#   (1) 마지막 200 iter 평균 fall %  < 0.15
#   (2) 마지막 1000 iter fall slope ≤ 0
#   (3) 마지막 200 iter 평균 reward > 8
#   (4) MuJoCo 무외란 vx=0.3 보행 ≥ 230/250 step
#   → 4개 중 3개 이상 만족 → PASS
#
# 진행상황 알림:
#   /tmp/hylion_progress.log         (실시간 — tail -f 로 확인)
#   docs/jimmy/training_log_2026-04-30.md  (영구 기록)
#   README.md 의 학습 상태 표 (스테이지 완료 시 자동 갱신)
#
# 사용법:
#   nohup bash sim/isaaclab/scripts/newton/run_50N_curriculum.sh \
#     > /tmp/hylion_50N_orchestrator.log 2>&1 &
#
#   # 특정 스테이지부터 재개:
#   START_STAGE=E0.3 bash sim/isaaclab/scripts/newton/run_50N_curriculum.sh
#
#   # 진행상황 확인:
#   tail -f /tmp/hylion_progress.log
# =============================================================================

set -uo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
BHL_DIR="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl"
TRAIN_SCRIPT="${REPO_ROOT}/sim/isaaclab/scripts/newton/train_hylion_newton_BG.py"
PYTHON_BIN="/home/laba/env_isaaclab/bin/python"
CKPT_DIR="${REPO_ROOT}/checkpoints/biped"
MUJOCO_SCRIPT="${REPO_ROOT}/sim/mujoco/play_mujoco.py"
MUJOCO_MJCF="${REPO_ROOT}/sim/isaaclab/robot/hylion_v6.xml"
PROGRESS_LOG="/tmp/hylion_progress.log"
TRAINING_LOG_MD="${REPO_ROOT}/docs/jimmy/training_log_2026-04-30.md"
README="${REPO_ROOT}/README.md"
STATE_FILE="/tmp/hylion_orchestrator_state.txt"

START_STAGE="${START_STAGE:-D4.5}"
TASK_ID="${TASK_ID:-Velocity-Hylion-BG-Generic-v0}"

# ── 동시 실행 방지 lock ─────────────────────────────────────────────────────────
LOCK_FILE="/tmp/hylion_50N_orchestrator.lock"
if [[ -f "$LOCK_FILE" ]]; then
    existing_pid=$(cat "$LOCK_FILE" 2>/dev/null)
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "[ERROR] 다른 인스턴스가 실행 중 (PID $existing_pid)"
        echo "         중단하려면: kill $existing_pid"
        exit 99
    else
        echo "[WARN] stale lock 파일 정리 (이전 PID $existing_pid)"
        rm -f "$LOCK_FILE"
    fi
fi
echo $$ > "$LOCK_FILE"

# 종료 시 자식 프로세스 + lock 정리
cleanup() {
    rm -f "$LOCK_FILE"
    pkill -P $$ 2>/dev/null || true
    pkill -f "train_hylion_newton_BG" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 게이트 기준 ───────────────────────────────────────────────────────────────
GATE_FALL_THRESHOLD="0.15"
GATE_REWARD_THRESHOLD="8"
GATE_MUJOCO_THRESHOLD=230            # /250 (무외란)
GATE_MUJOCO_PERTURB_THRESHOLD=180    # /250 (학습 외란 동등 조건)
HARD_FAIL_FALL="0.50"                # 이 이상이면 즉시 중단 (정책 붕괴)
GATES_REQUIRED=4                     # 5개 중 4개 이상 PASS

# ── 공통 학습 인수 ─────────────────────────────────────────────────────────────
COMMON_ARGS="--num_envs 4096 --headless"

# ── 진행상황 알림 함수 ─────────────────────────────────────────────────────────
notify() {
    local level="$1"; shift
    local msg="[$(date '+%F %T')] [${level}] $*"
    echo "$msg" | tee -a "$PROGRESS_LOG"
}

notify_section() {
    local title="$1"
    {
        echo ""
        echo "================================================================"
        echo "  $title"
        echo "  $(date '+%F %T')"
        echo "================================================================"
    } | tee -a "$PROGRESS_LOG"
}

# ── 학습 로그에서 메트릭 추출 ──────────────────────────────────────────────────
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
# 학습 외란과 동일한 force/torque로 MuJoCo sim2sim 검증
mujoco_check() {
    local ckpt="$1" force="$2" torque="$3"
    local result
    result=$(timeout 90 "$PYTHON_BIN" "$MUJOCO_SCRIPT" \
        --ckpt "$ckpt" --mjcf "$MUJOCO_MJCF" \
        --no-viewer --vx 0.3 --duration 10.0 \
        --external-push "$force" --external-torque "$torque" \
        2>&1 | grep -oP 'Survived \K[0-9]+' | tail -1 || echo "0")
    echo "${result:-0}"
}

mujoco_check_noperturb() {
    # 무외란 평지 보행 추가 검증 (기본 보행 능력 확인)
    local ckpt="$1"
    local result
    result=$(timeout 90 "$PYTHON_BIN" "$MUJOCO_SCRIPT" \
        --ckpt "$ckpt" --mjcf "$MUJOCO_MJCF" \
        --no-viewer --vx 0.3 --duration 10.0 \
        2>&1 | grep -oP 'Survived \K[0-9]+' | tail -1 || echo "0")
    echo "${result:-0}"
}

# ── 5-게이트 평가 (학습 메트릭 3 + MuJoCo 2) ──────────────────────────────────
evaluate_stage() {
    local stage="$1" logfile="$2" out_ckpt="$3" force="$4" torque="$5"

    notify INFO "Stage ${stage} 평가 시작..."

    # 학습 로그에서 충분한 iter가 있는지 먼저 확인
    local total_iter
    total_iter=$(grep -c "Mean reward:" "$logfile" 2>/dev/null || echo 0)
    if [[ "$total_iter" -lt 1000 ]]; then
        notify FATAL "  ❌ 학습 iter 부족 (${total_iter} < 1000) — 학습 비정상 종료"
        return 2
    fi
    notify INFO "  학습 완료 iter: ${total_iter}"

    local fall_avg fall_first_half fall_last_half reward_avg
    fall_avg=$(extract_fall_avg "$logfile" 200)
    fall_first_half=$(grep "base_orientation:" "$logfile" 2>/dev/null | awk '{print $NF}' | tail -1000 | head -500 | awk '{s+=$1; c++} END {if(c>0) printf "%.4f", s/c; else print "1.0"}')
    fall_last_half=$(extract_fall_avg "$logfile" 500)
    reward_avg=$(extract_reward_avg "$logfile" 200)

    notify INFO "  학습 로그 분석:"
    notify INFO "    fall % (last 200)        : ${fall_avg}  (기준 < ${GATE_FALL_THRESHOLD})"
    notify INFO "    fall slope (1k→0.5k→0)   : ${fall_first_half} → ${fall_last_half}  (감소 추세 필요)"
    notify INFO "    reward (last 200)        : ${reward_avg}  (기준 > ${GATE_REWARD_THRESHOLD})"

    # MuJoCo 검증 — 두 가지 모드
    notify INFO "  MuJoCo 무외란 보행 검증..."
    local mj_step_noperturb
    mj_step_noperturb=$(mujoco_check_noperturb "$out_ckpt")
    notify INFO "    MuJoCo 무외란            : ${mj_step_noperturb}/250  (기준 ≥ ${GATE_MUJOCO_THRESHOLD})"

    notify INFO "  MuJoCo ±${force}N 외란 보행 검증 (학습 조건 일치)..."
    local mj_step_perturb
    mj_step_perturb=$(mujoco_check "$out_ckpt" "$force" "$torque")
    notify INFO "    MuJoCo ±${force}N         : ${mj_step_perturb}/250  (기준 ≥ ${GATE_MUJOCO_PERTURB_THRESHOLD})"

    # 게이트 평가 (5개)
    local pass=0
    awk "BEGIN {exit !(${fall_avg} < ${GATE_FALL_THRESHOLD})}"   2>/dev/null && ((pass++))
    awk "BEGIN {exit !(${fall_last_half} <= ${fall_first_half})}" 2>/dev/null && ((pass++))
    awk "BEGIN {exit !(${reward_avg} > ${GATE_REWARD_THRESHOLD})}" 2>/dev/null && ((pass++))
    [[ "$mj_step_noperturb" -ge "$GATE_MUJOCO_THRESHOLD" ]] && ((pass++))
    [[ "$mj_step_perturb" -ge "$GATE_MUJOCO_PERTURB_THRESHOLD" ]] && ((pass++))

    notify INFO "  게이트 통과: ${pass}/5 (필요: ${GATES_REQUIRED}+)"

    # 결과 저장 (MD 업데이트용)
    cat > /tmp/hylion_stage_metrics.txt <<EOF
fall_avg=${fall_avg}
fall_slope=${fall_first_half}→${fall_last_half}
reward_avg=${reward_avg}
mj_step_noperturb=${mj_step_noperturb}/250
mj_step_perturb=${mj_step_perturb}/250
pass=${pass}/5
EOF

    # 즉시 중단 조건 (fall > 0.50 = 정책 붕괴)
    if awk "BEGIN {exit !(${fall_avg} > ${HARD_FAIL_FALL})}" 2>/dev/null; then
        notify FATAL "  ❌ Stage ${stage} 정책 붕괴 (fall=${fall_avg} > ${HARD_FAIL_FALL})"
        notify FATAL "     보강 단계 추가 또는 이전 단계 롤백 필요"
        return 2
    fi

    if (( pass >= GATES_REQUIRED )); then
        notify PASS "  ✅ Stage ${stage} PASS — 다음 스테이지 진행"
        return 0
    else
        notify FAIL "  ⚠ Stage ${stage} FAIL — 추가 학습 또는 보강 단계 필요"
        return 1
    fi
}

# ── MD/README 갱신 ────────────────────────────────────────────────────────────
update_md() {
    local stage="$1" force="$2" status="$3" out_ckpt="$4"
    local fall_avg reward_avg mj_noperturb mj_perturb
    fall_avg=$(grep "^fall_avg=" /tmp/hylion_stage_metrics.txt | cut -d= -f2)
    reward_avg=$(grep "^reward_avg=" /tmp/hylion_stage_metrics.txt | cut -d= -f2)
    mj_noperturb=$(grep "^mj_step_noperturb=" /tmp/hylion_stage_metrics.txt | cut -d= -f2)
    mj_perturb=$(grep "^mj_step_perturb=" /tmp/hylion_stage_metrics.txt | cut -d= -f2)

    # training_log_2026-04-30.md 에 한 줄 append
    local row
    row=$(printf "| %s | ±%sN | %s | %s | %s | %s | %s | %s |\n" \
          "$(date '+%m-%d %H:%M')" "$force" "$stage" \
          "${reward_avg:-?}" "${fall_avg:-?}" \
          "${mj_noperturb:-?}" "${mj_perturb:-?}" "$status")

    if [[ ! -f "$TRAINING_LOG_MD" ]] || ! grep -q "MuJoCo 무외란" "$TRAINING_LOG_MD"; then
        cat > "$TRAINING_LOG_MD" <<EOF
# 학습 진행 로그 (2026-04-30 v2 자동)

오케스트레이터: \`sim/isaaclab/scripts/newton/run_50N_curriculum.sh\`
시작: $(date '+%F %T')

| 시각 | 외력 | Stage | reward (200) | fall % (200) | MuJoCo 무외란 | MuJoCo 외란 | 결과 |
|------|-----:|-------|-------------:|-------------:|:-------------:|:-----------:|:----:|
EOF
    fi
    echo "$row" >> "$TRAINING_LOG_MD"

    notify INFO "  → ${TRAINING_LOG_MD} 갱신 완료"
}

# ── 학습 실행 함수 ────────────────────────────────────────────────────────────
run_stage() {
    local stage="$1" force="$2" torque="$3" mass_add="$4" max_iter="$5" ckpt_in="$6" ckpt_out_dir="$7"

    notify_section "Stage ${stage}  (외력 ±${force}N, mass+${mass_add}kg, ${max_iter} iter)"
    notify INFO "  Resume from: ${ckpt_in}"
    notify INFO "  Output     : ${ckpt_out_dir}/best.pt"

    if [[ ! -f "$ckpt_in" ]]; then
        notify FATAL "  체크포인트 없음: ${ckpt_in}"
        return 1
    fi
    mkdir -p "$ckpt_out_dir"

    # 백업 (이전 best.pt가 있으면)
    if [[ -f "${ckpt_out_dir}/best.pt" ]]; then
        cp "${ckpt_out_dir}/best.pt" "${ckpt_out_dir}/best.pt.prev_$(date +%H%M)"
    fi

    local logfile="/tmp/hylion_50N_${stage//./p}.log"
    notify INFO "  로그       : ${logfile}"

    cd "$BHL_DIR" || return 1
    source /home/laba/env_isaaclab/bin/activate 2>/dev/null
    unset PYTHONPATH PYTHONHOME 2>/dev/null

    # 환경변수 설정
    export HYLION_ENABLE_PERTURBATION=1
    export HYLION_PERTURB_FORCE="$force"
    export HYLION_PERTURB_TORQUE="$torque"
    export HYLION_BASE_MASS_ADD_KG="$mass_add"
    export HYLION_MAX_LIN_VEL_X=0.5
    export HYLION_STANDING_RATIO=0.05

    # 학습 시작 직전 marker — stale 체크포인트 디렉터리 식별 방지
    local marker_file="/tmp/hylion_train_marker_${stage//./p}"
    touch "$marker_file"
    sleep 1  # 파일 시각이 다음 mkdir 보다 먼저

    # 진행 모니터 백그라운드 실행 (10분마다 reward/fall 보고)
    # log file이 아직 없을 수 있으므로 먼저 빈 파일 생성
    : > "$logfile"
    (
        prev_iter=0
        for _ in $(seq 1 200); do  # 최대 200회 (200 × 10분 = 33시간, 안전 한도)
            sleep 600
            [[ -f "$logfile" ]] || break
            cur_iter=$(grep -oP "Learning iteration \K[0-9]+" "$logfile" 2>/dev/null | tail -1)
            cur_reward=$(extract_reward_avg "$logfile" 50)
            cur_fall=$(extract_fall_avg "$logfile" 50)
            if [[ -n "$cur_iter" && "$cur_iter" != "$prev_iter" ]]; then
                notify INFO "  [${stage} 진행] iter=${cur_iter} reward=${cur_reward} fall=${cur_fall}"
                prev_iter="$cur_iter"
            fi
        done
    ) &
    local monitor_pid=$!
    disown $monitor_pid 2>/dev/null || true

    PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
        "$PYTHON_BIN" "$TRAIN_SCRIPT" \
        $COMMON_ARGS \
        --task "$TASK_ID" \
        --pretrained_checkpoint "$ckpt_in" \
        --max_iterations "$max_iter" \
        2>&1 | tee "$logfile"
    local train_rc=${PIPESTATUS[0]}

    kill "$monitor_pid" 2>/dev/null

    if [[ $train_rc -ne 0 ]]; then
        notify FATAL "  학습 프로세스 비정상 종료 (rc=${train_rc})"
        return 1
    fi

    # 최신 체크포인트 찾기 → 복사
    # ★ 중요: marker 파일보다 NEWER 한 디렉터리만 고려 (stale 디렉터리 방지)
    local result_ckpt
    result_ckpt=$(find "${BHL_DIR}/logs/rsl_rl/hylion/" -maxdepth 1 -type d -newer "$marker_file" 2>/dev/null | sort | tail -1)
    rm -f "$marker_file"
    if [[ -z "$result_ckpt" || ! -d "$result_ckpt" ]]; then
        notify FATAL "  학습 결과 디렉터리 못 찾음 (marker 이후 생성된 디렉터리 없음)"
        notify FATAL "    학습이 비정상 종료된 것으로 보임"
        return 1
    fi
    notify INFO "  학습 결과 디렉터리: ${result_ckpt}"
    local latest_pt
    latest_pt=$(ls "${result_ckpt}"/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [[ -z "$latest_pt" ]]; then
        notify FATAL "  체크포인트 파일 못 찾음 (${result_ckpt})"
        return 1
    fi
    # iter 수가 max_iter의 절반 이상은 되어야 정상 학습 인정
    local pt_iter
    pt_iter=$(echo "$latest_pt" | grep -oP 'model_\K[0-9]+')
    local min_iter=$((max_iter / 2))
    if [[ "$pt_iter" -lt "$min_iter" ]]; then
        notify FATAL "  학습이 너무 일찍 종료 (model_${pt_iter}.pt < ${min_iter})"
        return 1
    fi
    cp "$latest_pt" "${ckpt_out_dir}/best.pt"
    notify INFO "  체크포인트 저장: ${ckpt_out_dir}/best.pt (from model_${pt_iter}.pt)"

    return 0
}

# ── 스테이지 정의 ─────────────────────────────────────────────────────────────
# 형식: "stage|force|torque|mass_add|max_iter|input_dir"
# 학습 실패 시 보강 외력은 9-2 표 참조 (현재 구현은 강제 진행 없음).
STAGES=(
    # 옵션 A (2026-05-01): BHL 설정 (lr=1e-3, adaptive) — 학습 빠르므로 max_iter 짧게
    # Phase 2: D4 → ±30N (12 단계, 각 4000 iter)
    "D4.5|7.0|2.0|0.75|4000|stage_d4_hylion_v6"
    "D5|10.0|3.0|1.0|4000|stage_d4_5_hylion_v6"
    "E0.3|12.0|3.5|1.0|4000|stage_d5_hylion_v6"
    "E0.6|13.5|4.0|1.25|4000|stage_e0_3_hylion_v6"
    "E1|15.0|4.5|1.25|4000|stage_e0_6_hylion_v6"
    "E1.4|17.0|5.0|1.5|4000|stage_e1_hylion_v6"
    "E1.8|19.0|5.5|1.5|4000|stage_e1_4_hylion_v6"
    "E2.3|21.0|5.5|1.5|4000|stage_e1_8_hylion_v6"
    "E2.7|23.0|6.0|2.0|4000|stage_e2_3_hylion_v6"
    "E3|25.0|6.5|2.0|4000|stage_e2_7_hylion_v6"
    "E3.5|27.5|7.0|2.0|4000|stage_e3_hylion_v6"
    "E4|30.0|8.0|2.0|5000|stage_e3_5_hylion_v6"
    # Phase 3: ±30N → ±50N (7 단계)
    "E5|33.0|9.0|2.0|4000|stage_e4_hylion_v6"
    "E5.5|36.0|10.0|2.5|4000|stage_e5_hylion_v6"
    "E6|39.0|11.0|2.5|4000|stage_e5_5_hylion_v6"
    "E6.5|42.0|12.0|2.5|4000|stage_e6_hylion_v6"
    "E7|45.0|13.0|3.0|4000|stage_e6_5_hylion_v6"
    "E7.5|47.5|14.0|3.0|5000|stage_e7_hylion_v6"
    "E8|50.0|14.0|3.0|6000|stage_e7_5_hylion_v6"
)

stage_to_dir() {
    local s="$1"; echo "stage_${s//./_}_hylion_v6" | tr '[:upper:]' '[:lower:]'
}

stage_should_run() {
    local stage="$1"
    local started=false
    for entry in "${STAGES[@]}"; do
        local s="${entry%%|*}"
        [[ "$s" == "$START_STAGE" ]] && started=true
        [[ "$s" == "$stage" && "$started" == true ]] && return 0
    done
    return 1
}

# ── 메인 실행 ─────────────────────────────────────────────────────────────────
notify_section "Hylion 50N Curriculum 시작"
notify INFO "시작 스테이지: ${START_STAGE}"
notify INFO "총 ${#STAGES[@]} 스테이지, 예상 ~10일"
notify INFO "진행상황: tail -f ${PROGRESS_LOG}"
notify INFO "영구기록: ${TRAINING_LOG_MD}"

# 디스크 공간 체크
df_avail=$(df "$REPO_ROOT" | awk 'NR==2 {print $4}')
notify INFO "사용 가능 공간: $((df_avail / 1024 / 1024)) GB (스테이지당 ~100MB)"

start_time=$(date +%s)

for entry in "${STAGES[@]}"; do
    IFS='|' read -r stage force torque mass_add max_iter input_dir <<< "$entry"

    if ! stage_should_run "$stage"; then
        continue
    fi

    out_dir_name=$(stage_to_dir "$stage")
    ckpt_in="${CKPT_DIR}/${input_dir}/best.pt"
    ckpt_out_dir="${CKPT_DIR}/${out_dir_name}"

    # 학습 실행
    if ! run_stage "$stage" "$force" "$torque" "$mass_add" "$max_iter" "$ckpt_in" "$ckpt_out_dir"; then
        notify FATAL "Stage ${stage} 학습 실패 — 자동 진행 중단"
        echo "$stage" > "$STATE_FILE"
        exit 1
    fi

    # 평가 (5-게이트, 학습 외란 == MuJoCo 외란)
    logfile="/tmp/hylion_50N_${stage//./p}.log"
    eval_rc=0
    evaluate_stage "$stage" "$logfile" "${ckpt_out_dir}/best.pt" "$force" "$torque"
    eval_rc=$?

    # MD 갱신
    case $eval_rc in
        0) update_md "$stage" "$force" "✅ PASS" "${ckpt_out_dir}/best.pt" ;;
        1) update_md "$stage" "$force" "⚠ FAIL"  "${ckpt_out_dir}/best.pt" ;;
        2) update_md "$stage" "$force" "❌ 붕괴"  "${ckpt_out_dir}/best.pt" ;;
    esac

    # PASS가 아니면 중단
    if [[ $eval_rc -ne 0 ]]; then
        notify FATAL "Stage ${stage} 게이트 실패 — 자동 진행 중단"
        notify INFO  "  현재 best.pt: ${ckpt_out_dir}/best.pt"
        notify INFO  "  재개 명령: START_STAGE=${stage} bash $0"
        echo "$stage" > "$STATE_FILE"
        exit $eval_rc
    fi

    # 누적 시간 보고
    elapsed=$(( $(date +%s) - start_time ))
    notify INFO "  누적 학습 시간: $((elapsed / 3600))시간 $(((elapsed % 3600) / 60))분"
done

notify_section "🎉 Hylion 50N Curriculum 완료"
notify INFO "총 시간: $((($(date +%s) - start_time) / 3600))시간"
notify INFO "최종 체크포인트: ${CKPT_DIR}/stage_e8_hylion_v6/best.pt"
notify INFO "발표용 MuJoCo 영상 캡처 권장:"
notify INFO "  DISPLAY=:0 python3 ${MUJOCO_SCRIPT} \\"
notify INFO "    --ckpt ${CKPT_DIR}/stage_e8_hylion_v6/best.pt \\"
notify INFO "    --vx 0.3 --duration 15.0"
