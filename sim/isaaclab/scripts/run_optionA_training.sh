#!/usr/bin/env bash
# =============================================================================
# run_optionA_training.sh
# Option A: C1 → D1(±1N) → D2(±2N) → D3(±3N) → D4(±5N) → D5(±10N)
#
# 배경:
#   C2(±3N, 3000iter)에서 orientation이 0.002 → 0.52로 악화됨
#   원인: 외력 첫 경험인데 3N은 너무 큰 충격
#   해법: ±1N → ±2N → ±3N 으로 매우 세밀하게 올림
#
# 사용법:
#   nohup bash /home/laba/project_singularity/δ3/scripts/run_optionA_training.sh \
#     > /tmp/hylion_optionA_orchestrator.log 2>&1 &
#
# 특정 스테이지부터 재개:
#   START_STAGE=D2 bash .../run_optionA_training.sh
#
# 특정 체크포인트 지정:
#   D1_CKPT=/path/to/model.pt START_STAGE=D1 bash .../run_optionA_training.sh
# =============================================================================

set -euo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BHL_DIR="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl"
TRAIN_SCRIPT="/home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py"
PYTHON_BIN="/home/laba/env_isaaclab/bin/python"
CKPT_DIR="/home/laba/project_singularity/δ3/checkpoints"
LOG_DIR="/tmp"

# ── 시작 스테이지 (기본: D1) ──────────────────────────────────────────────────
START_STAGE="${START_STAGE:-D1}"

# ── 각 스테이지 입력 체크포인트 (override 가능) ────────────────────────────────
# D1은 C1 best.pt에서 시작 (orientation 3.84% — 검증된 최고 수준)
D1_CKPT="${D1_CKPT:-${CKPT_DIR}/stage_c1_hylion_v6/best.pt}"
D1_5_CKPT="${D1_5_CKPT:-${CKPT_DIR}/stage_d1_hylion_v6/best.pt}"
D2_CKPT="${D2_CKPT:-${CKPT_DIR}/stage_d1_5_hylion_v6/best.pt}"
D2_5_CKPT="${D2_5_CKPT:-${CKPT_DIR}/stage_d2_hylion_v6/best.pt}"
D3_CKPT="${D3_CKPT:-${CKPT_DIR}/stage_d2_5_hylion_v6/best.pt}"
D4_CKPT="${D4_CKPT:-${CKPT_DIR}/stage_d3_hylion_v6/best.pt}"
D4_5_CKPT="${D4_5_CKPT:-${CKPT_DIR}/stage_d4_hylion_v6/best.pt}"
D5_CKPT="${D5_CKPT:-${CKPT_DIR}/stage_d4_5_hylion_v6/best.pt}"

# ── 성공 판단 기준 ─────────────────────────────────────────────────────────────
SUCCESS_ORIENTATION_THRESHOLD="0.15"   # C보다 엄격 — 외력 환경에서도 15% 이하

# ── 공통 학습 인수 ─────────────────────────────────────────────────────────────
COMMON_ARGS="--num_envs 4096 --headless"

# ── 로그 함수 ─────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%F %T')] $*"; }

# ── 체크포인트 존재 확인 ───────────────────────────────────────────────────────
check_ckpt() {
    local ckpt="$1"
    local stage="$2"
    if [[ ! -f "$ckpt" ]]; then
        log "ERROR: ${stage} 시작 체크포인트 없음: ${ckpt}"
        exit 1
    fi
}

# ── 학습 결과에서 최신 체크포인트 찾기 ───────────────────────────────────────
find_latest_ckpt() {
    local bhl_logs="/home/laba/project_singularity/logs/rsl_rl/hylion"
    local latest_dir
    latest_dir=$(ls -dt "${bhl_logs}"/2026-*/ 2>/dev/null | head -1)
    if [[ -z "$latest_dir" ]]; then
        echo ""
        return
    fi
    local latest_pt
    latest_pt=$(ls "${latest_dir}"model_*.pt 2>/dev/null | sort -V | tail -1)
    echo "$latest_pt"
}

# ── orientation 성공 여부 판단 ────────────────────────────────────────────────
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

# ── 스테이지별 학습 함수 ───────────────────────────────────────────────────────
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
        log "  현재 스테이지 재실행 방법:"
        local upper_stage
        upper_stage=$(echo "${stage}" | tr '[:lower:]' '[:upper:]')
        log "  ${upper_stage}_CKPT=${out_ckpt_dir}/best.pt START_STAGE=${stage} bash $0"
    fi
}

# ── 스테이지 순서 정의 ────────────────────────────────────────────────────────
STAGES=("D1" "D1.5" "D2" "D2.5" "D3" "D4" "D4.5" "D5")

stage_should_run() {
    local stage="$1"
    local started=false
    for s in "${STAGES[@]}"; do
        [[ "$s" == "$START_STAGE" ]] && started=true
        [[ "$s" == "$stage" && "$started" == true ]] && return 0
    done
    return 1
}

# ── 메인 실행 ─────────────────────────────────────────────────────────────────
log "======================================================"
log "Option A Progressive Training 시작"
log "  시작 스테이지: ${START_STAGE}"
log "  베이스: stage_c1_hylion_v6/best.pt (orientation 3.84%)"
log "  경로: ±1N → ±1.5N → ±2N → ±2.5N → ±3N → ±5N → ±7N → ±10N"
log "======================================================"

# Stage D1: base_mass ±0.5kg, 외력 ±1N (최초 외력 경험)
if stage_should_run "D1"; then
    run_stage "D1" \
        "Velocity-Hylion-BG-D1-v0" \
        "$D1_CKPT" \
        "${LOG_DIR}/hylion_v6_stageD1.log" \
        "${CKPT_DIR}/stage_d1_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=0.5" \
        "HYLION_PERTURB_FORCE=1.0" \
        "HYLION_PERTURB_TORQUE=0.3" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.02"
fi

# Stage D1.5: base_mass ±0.5kg, 외력 ±1.5N (D2 NaN 폭발 대응 — 2026-04-21 추가)
if stage_should_run "D1.5"; then
    run_stage "D1.5" \
        "Velocity-Hylion-BG-D1p5-v0" \
        "$D1_5_CKPT" \
        "${LOG_DIR}/hylion_v6_stageD1_5.log" \
        "${CKPT_DIR}/stage_d1_5_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=0.5" \
        "HYLION_PERTURB_FORCE=1.5" \
        "HYLION_PERTURB_TORQUE=0.45" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.02"
fi

# Stage D2: base_mass ±0.5kg, 외력 ±2N
if stage_should_run "D2"; then
    run_stage "D2" \
        "Velocity-Hylion-BG-D2-v0" \
        "$D2_CKPT" \
        "${LOG_DIR}/hylion_v6_stageD2.log" \
        "${CKPT_DIR}/stage_d2_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=0.5" \
        "HYLION_PERTURB_FORCE=2.0" \
        "HYLION_PERTURB_TORQUE=0.5" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.02"
fi

# Stage D2.5: base_mass ±0.5kg, 외력 ±2.5N (±2N→±3N 갭 완충 — 2026-04-22 추가)
if stage_should_run "D2.5"; then
    run_stage "D2.5" \
        "Velocity-Hylion-BG-D2p5-v0" \
        "$D2_5_CKPT" \
        "${LOG_DIR}/hylion_v6_stageD2_5.log" \
        "${CKPT_DIR}/stage_d2_5_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=0.5" \
        "HYLION_PERTURB_FORCE=2.5" \
        "HYLION_PERTURB_TORQUE=0.75" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.02"
fi

# Stage D3: base_mass ±0.5kg, 외력 ±3N (C2 실패 수준 재도전, 4000iter)
if stage_should_run "D3"; then
    run_stage "D3" \
        "Velocity-Hylion-BG-D3-v0" \
        "$D3_CKPT" \
        "${LOG_DIR}/hylion_v6_stageD3.log" \
        "${CKPT_DIR}/stage_d3_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=0.5" \
        "HYLION_PERTURB_FORCE=3.0" \
        "HYLION_PERTURB_TORQUE=1.0" \
        "HYLION_MAX_LIN_VEL_X=0.55" \
        "HYLION_STANDING_RATIO=0.03"
fi

# Stage D4: base_mass ±1.0kg, 외력 ±5N
if stage_should_run "D4"; then
    run_stage "D4" \
        "Velocity-Hylion-BG-D4-v0" \
        "$D4_CKPT" \
        "${LOG_DIR}/hylion_v6_stageD4.log" \
        "${CKPT_DIR}/stage_d4_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=1.0" \
        "HYLION_PERTURB_FORCE=5.0" \
        "HYLION_PERTURB_TORQUE=1.5" \
        "HYLION_MAX_LIN_VEL_X=0.6" \
        "HYLION_STANDING_RATIO=0.05"
fi

# Stage D4.5: base_mass ±1.0kg, 외력 ±7N (±5N→±10N 갭 완충 — 2026-04-22 추가)
if stage_should_run "D4.5"; then
    run_stage "D4.5" \
        "Velocity-Hylion-BG-D4p5-v0" \
        "$D4_5_CKPT" \
        "${LOG_DIR}/hylion_v6_stageD4_5.log" \
        "${CKPT_DIR}/stage_d4_5_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=1.0" \
        "HYLION_PERTURB_FORCE=7.0" \
        "HYLION_PERTURB_TORQUE=2.2" \
        "HYLION_MAX_LIN_VEL_X=0.65" \
        "HYLION_STANDING_RATIO=0.07"
fi

# Stage D5: base_mass ±1.5kg, 외력 ±10N (최종 강건성 목표)
if stage_should_run "D5"; then
    run_stage "D5" \
        "Velocity-Hylion-BG-D5-v0" \
        "$D5_CKPT" \
        "${LOG_DIR}/hylion_v6_stageD5.log" \
        "${CKPT_DIR}/stage_d5_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=1.5" \
        "HYLION_PERTURB_FORCE=10.0" \
        "HYLION_PERTURB_TORQUE=3.0" \
        "HYLION_MAX_LIN_VEL_X=0.7" \
        "HYLION_STANDING_RATIO=0.10"
fi

log "======================================================"
log "Option A Progressive Training 전체 완료!"
log "최종 체크포인트: ${CKPT_DIR}/stage_d5_hylion_v6/best.pt"
log "======================================================"
