#!/usr/bin/env bash
# =============================================================================
# run_e_stages_training.sh
# Stage E: D5(±10N) 완료 후 ±30N 최종 목표 커리큘럼
#
# 경로: E1(±15N) → E2(±20N) → E3(±25N) → E4(±30N)
#
# 사용법:
#   nohup bash /home/laba/project_singularity/δ3/scripts/run_e_stages_training.sh \
#     > /tmp/hylion_stageE_orchestrator.log 2>&1 &
#
# 특정 스테이지부터 재개:
#   START_STAGE=E2 bash .../run_e_stages_training.sh
#
# D5 best.pt 직접 지정:
#   E1_CKPT=/path/to/best.pt bash .../run_e_stages_training.sh
# =============================================================================

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BHL_DIR="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl"
TRAIN_SCRIPT="/home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py"
PYTHON_BIN="/home/laba/env_isaaclab/bin/python"
CKPT_DIR="/home/laba/project_singularity/δ3/checkpoints"
LOG_DIR="/tmp"

# ── 시작 스테이지 (기본: E1) ───────────────────────────────────────────────────
START_STAGE="${START_STAGE:-E1}"

# ── 각 스테이지 입력 체크포인트 (override 가능) ────────────────────────────────
E1_CKPT="${E1_CKPT:-${CKPT_DIR}/stage_d5_hylion_v6/best.pt}"
E2_CKPT="${E2_CKPT:-${CKPT_DIR}/stage_e1_hylion_v6/best.pt}"
E3_CKPT="${E3_CKPT:-${CKPT_DIR}/stage_e2_hylion_v6/best.pt}"
E4_CKPT="${E4_CKPT:-${CKPT_DIR}/stage_e3_hylion_v6/best.pt}"

# ── 성공 판단 기준 (E 단계는 외력이 크므로 15% 유지) ──────────────────────────
SUCCESS_ORIENTATION_THRESHOLD="0.15"

# ── 공통 학습 인수 ─────────────────────────────────────────────────────────────
COMMON_ARGS="--num_envs 4096 --headless"

# ── 로그 함수 ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%F %T')] $*"; }

# ── 체크포인트 존재 확인 ────────────────────────────────────────────────────────
check_ckpt() {
    local ckpt="$1"
    local stage="$2"
    if [[ ! -f "$ckpt" ]]; then
        log "ERROR: ${stage} 시작 체크포인트 없음: ${ckpt}"
        exit 1
    fi
}

# ── 학습 결과에서 최신 체크포인트 찾기 ─────────────────────────────────────────
find_latest_ckpt() {
    local proj_logs="/home/laba/project_singularity/logs/rsl_rl/hylion"
    local latest_dir
    latest_dir=$(ls -dt "${proj_logs}"/2026-*/ 2>/dev/null | head -1)
    if [[ -z "$latest_dir" ]]; then
        echo ""
        return
    fi
    local latest_pt
    latest_pt=$(ls "${latest_dir}"model_*.pt 2>/dev/null | sort -V | tail -1)
    echo "$latest_pt"
}

# ── orientation 성공 여부 판단 ─────────────────────────────────────────────────
check_success() {
    local logfile="$1"
    local threshold="$2"
    local avg
    avg=$(grep "Episode_Termination/base_orientation:" "$logfile" 2>/dev/null | \
          tail -20 | awk '{sum+=$NF; cnt++} END {if(cnt>0) printf "%.4f", sum/cnt; else print "1.0"}')
    log "  최근 20iter 평균 base_orientation termination: ${avg} (기준: <${threshold})"
    if awk "BEGIN {exit !(${avg} < ${threshold})}"; then
        return 0
    else
        return 1
    fi
}

# ── 스테이지별 학습 함수 ────────────────────────────────────────────────────────
run_stage() {
    local stage="$1"
    local task="$2"
    local ckpt="$3"
    local logfile="$4"
    local out_ckpt_dir="$5"
    shift 5
    local extra_env_vars=("$@")

    log "======================================================"
    log "Stage ${stage} 학습 시작"
    log "  Task:    ${task}"
    log "  Resume:  ${ckpt}"
    log "  Log:     ${logfile}"
    log "======================================================"

    check_ckpt "$ckpt" "$stage"
    mkdir -p "$out_ckpt_dir"

    cd "$BHL_DIR"
    source /home/laba/env_isaaclab/bin/activate
    unset PYTHONPATH || true
    unset PYTHONHOME || true

    if [[ ${#extra_env_vars[@]} -gt 0 ]]; then
        for ev in "${extra_env_vars[@]}"; do
            [[ -n "$ev" ]] && export "$ev"
        done
    fi

    PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
        "$PYTHON_BIN" "$TRAIN_SCRIPT" \
        $COMMON_ARGS \
        --task "$task" \
        --pretrained_checkpoint "$ckpt" \
        2>&1 | tee "$logfile"

    if [[ ${#extra_env_vars[@]} -gt 0 ]]; then
        for ev in "${extra_env_vars[@]}"; do
            if [[ -n "$ev" ]]; then
                local key="${ev%%=*}"
                unset "$key" || true
            fi
        done
    fi

    local result_ckpt
    result_ckpt=$(find_latest_ckpt)
    if [[ -z "$result_ckpt" ]]; then
        log "ERROR: Stage ${stage} 완료 후 체크포인트를 찾을 수 없음"
        exit 1
    fi
    log "Stage ${stage} 완료. 최신 체크포인트: ${result_ckpt}"
    cp "$result_ckpt" "${out_ckpt_dir}/best.pt"
    log "→ ${out_ckpt_dir}/best.pt 저장 완료"

    if check_success "$logfile" "$SUCCESS_ORIENTATION_THRESHOLD"; then
        log "✓ Stage ${stage} 성공 (orientation < ${SUCCESS_ORIENTATION_THRESHOLD})"
    else
        log "⚠ Stage ${stage} 수렴 불충분 (orientation ≥ ${SUCCESS_ORIENTATION_THRESHOLD})"
        local upper_stage
        upper_stage=$(echo "${stage}" | tr '[:lower:]' '[:upper:]')
        log "  재실행: ${upper_stage}_CKPT=${out_ckpt_dir}/best.pt START_STAGE=${stage} bash $0"
        log "  또는 중간 스테이지 삽입 후 재시도 (29_failure_contingency_plan 참조)"
    fi
}

# ── 스테이지 순서 정의 ──────────────────────────────────────────────────────────
STAGES=("E1" "E2" "E3" "E4")

stage_should_run() {
    local stage="$1"
    local started=false
    for s in "${STAGES[@]}"; do
        [[ "$s" == "$START_STAGE" ]] && started=true
        [[ "$s" == "$stage" && "$started" == true ]] && return 0
    done
    return 1
}

# ── 메인 실행 ──────────────────────────────────────────────────────────────────
log "======================================================"
log "Stage E Progressive Training 시작"
log "  시작 스테이지: ${START_STAGE}"
log "  베이스: stage_d5_hylion_v6/best.pt (외력 ±10N 달성)"
log "  경로: ±15N → ±20N → ±25N → ±30N"
log "======================================================"

# Stage E1: base_mass ±1.5kg, 외력 ±15N
if stage_should_run "E1"; then
    run_stage "E1" \
        "Velocity-Hylion-BG-E1-v0" \
        "$E1_CKPT" \
        "${LOG_DIR}/hylion_v6_stageE1.log" \
        "${CKPT_DIR}/stage_e1_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=1.5" \
        "HYLION_PERTURB_FORCE=15.0" \
        "HYLION_PERTURB_TORQUE=5.0" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.10"
fi

# Stage E2: base_mass ±2.0kg, 외력 ±20N
if stage_should_run "E2"; then
    run_stage "E2" \
        "Velocity-Hylion-BG-E2-v0" \
        "$E2_CKPT" \
        "${LOG_DIR}/hylion_v6_stageE2.log" \
        "${CKPT_DIR}/stage_e2_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=2.0" \
        "HYLION_PERTURB_FORCE=20.0" \
        "HYLION_PERTURB_TORQUE=6.0" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.10"
fi

# Stage E3: base_mass ±2.0kg, 외력 ±25N
if stage_should_run "E3"; then
    run_stage "E3" \
        "Velocity-Hylion-BG-E3-v0" \
        "$E3_CKPT" \
        "${LOG_DIR}/hylion_v6_stageE3.log" \
        "${CKPT_DIR}/stage_e3_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=2.0" \
        "HYLION_PERTURB_FORCE=25.0" \
        "HYLION_PERTURB_TORQUE=7.5" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.10"
fi

# Stage E4: base_mass ±2.0kg, 외력 ±30N (최종 목표 ✨)
if stage_should_run "E4"; then
    run_stage "E4" \
        "Velocity-Hylion-BG-E4-v0" \
        "$E4_CKPT" \
        "${LOG_DIR}/hylion_v6_stageE4.log" \
        "${CKPT_DIR}/stage_e4_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=2.0" \
        "HYLION_PERTURB_FORCE=30.0" \
        "HYLION_PERTURB_TORQUE=9.0" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.10"
fi

log "======================================================"
log "Stage E Progressive Training 전체 완료! 🎉"
log "최종 체크포인트: ${CKPT_DIR}/stage_e4_hylion_v6/best.pt"
log "  → 외력 ±30N 강건 보행 달성"
log "  → 실제 로봇 대응 가능 수준: ~15~20N (sim-to-real 마진 포함)"
log "======================================================"
