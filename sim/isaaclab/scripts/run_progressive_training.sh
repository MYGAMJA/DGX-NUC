#!/usr/bin/env bash
# =============================================================================
# run_progressive_training.sh
# Stage B+ → C1 → C2 → C3 → C4 단계별 진행 학습 스크립트
#
# 사용법:
#   bash /home/laba/project_singularity/δ3/scripts/run_progressive_training.sh
#
# 시작 스테이지 지정 (재개 시):
#   START_STAGE=C1 bash .../run_progressive_training.sh
#
# 특정 스테이지 체크포인트로 재개:
#   BPLUS_CKPT=/path/to/model.pt START_STAGE=C1 bash .../run_progressive_training.sh
# =============================================================================

set -euo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BHL_DIR="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl"
TRAIN_SCRIPT="/home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py"
PYTHON_BIN="/home/laba/env_isaaclab/bin/python"
CKPT_DIR="/home/laba/project_singularity/δ3/checkpoints"
LOG_DIR="/tmp"

# ── 시작 스테이지 (기본: Bplus) ───────────────────────────────────────────────
START_STAGE="${START_STAGE:-Bplus}"

# ── 각 스테이지 입력 체크포인트 (override 가능) ────────────────────────────────
# Stage B+는 현재 Stage B best.pt에서 시작
BPLUS_CKPT="${BPLUS_CKPT:-${CKPT_DIR}/stage_b_hylion_v6/best.pt}"
C1_CKPT="${C1_CKPT:-${CKPT_DIR}/stage_bplus_hylion_v6/best.pt}"
C2_CKPT="${C2_CKPT:-${CKPT_DIR}/stage_c1_hylion_v6/best.pt}"
C3_CKPT="${C3_CKPT:-${CKPT_DIR}/stage_c2_hylion_v6/best.pt}"
C4_CKPT="${C4_CKPT:-${CKPT_DIR}/stage_c3_hylion_v6/best.pt}"

# ── 성공 판단 기준 ─────────────────────────────────────────────────────────────
# base_orientation termination이 이 값 이하여야 다음 스테이지로 진행
SUCCESS_ORIENTATION_THRESHOLD="0.30"

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
        log "  확인 후 환경변수로 경로를 지정하세요:"
        log "  export $(echo "${stage}" | tr '[:lower:]' '[:upper:]')_CKPT=/path/to/model.pt"
        exit 1
    fi
}

# ── 학습 결과에서 최신 best 체크포인트 찾기 ───────────────────────────────────
find_latest_ckpt() {
    local log_subdir="$1"  # e.g. "hylion"
    local bhl_logs="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/${log_subdir}"
    # 가장 최근에 수정된 폴더의 최신 모델
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

# ── 학습 종료 후 orientation 성공 여부 판단 ────────────────────────────────────
check_success() {
    local logfile="$1"
    local threshold="$2"
    # 마지막 10개 iteration의 base_orientation 평균이 threshold 미만인지 확인
    local avg
    avg=$(grep "Episode_Termination/base_orientation:" "$logfile" 2>/dev/null | \
          tail -10 | awk '{sum+=$NF; cnt++} END {if(cnt>0) printf "%.4f", sum/cnt; else print "1.0"}')
    log "  최근 10iter 평균 base_orientation termination: ${avg} (기준: <${threshold})"
    # awk로 부동소수점 비교
    if awk "BEGIN {exit !(${avg} < ${threshold})}"; then
        return 0  # 성공
    else
        return 1  # 실패
    fi
}

# ── 스테이지별 학습 함수 ───────────────────────────────────────────────────────
run_stage() {
    local stage="$1"          # e.g. "Bplus"
    local task="$2"           # e.g. "Velocity-Hylion-BG-Bplus-v0"
    local ckpt="$3"           # 시작 체크포인트 경로
    local logfile="$4"        # 로그 파일 경로
    local out_ckpt_dir="$5"   # 결과 체크포인트 저장 디렉토리
    shift 5
    local extra_env_vars=("$@")  # 환경변수 배열 (e.g. HYLION_ENABLE_PERTURBATION=1 ...)

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

    # 환경변수 적용
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

    # 환경변수 해제
    if [[ ${#extra_env_vars[@]} -gt 0 ]]; then
        for ev in "${extra_env_vars[@]}"; do
            if [[ -n "$ev" ]]; then
                local key="${ev%%=*}"
                unset "$key" || true
            fi
        done
    fi

    # 학습 완료 후 최신 체크포인트 복사
    local result_ckpt
    result_ckpt=$(find_latest_ckpt "hylion")
    if [[ -z "$result_ckpt" ]]; then
        log "ERROR: Stage ${stage} 완료 후 체크포인트를 찾을 수 없음"
        exit 1
    fi
    log "Stage ${stage} 완료. 최신 체크포인트: ${result_ckpt}"
    cp "$result_ckpt" "${out_ckpt_dir}/best.pt"
    log "→ ${out_ckpt_dir}/best.pt 저장 완료"

    # 성공 여부 평가
    if check_success "$logfile" "$SUCCESS_ORIENTATION_THRESHOLD"; then
        log "✓ Stage ${stage} 성공 (orientation termination < ${SUCCESS_ORIENTATION_THRESHOLD})"
    else
        log "⚠ Stage ${stage} 수렴 불충분 (orientation termination ≥ ${SUCCESS_ORIENTATION_THRESHOLD})"
        log "  다음 스테이지로 진행하되, 결과를 주의 깊게 모니터링하세요."
        log "  현재 스테이지를 반복하려면:"
        log "  $(echo "${stage}" | tr '[:lower:]' '[:upper:]')_CKPT=${out_ckpt_dir}/best.pt START_STAGE=${stage} bash $0"
    fi
}

# ── STAGE 순서 및 설정 ────────────────────────────────────────────────────────
declare -a STAGES=("Bplus" "C1" "C2" "C3" "C4")

stage_should_run() {
    local stage="$1"
    local started=false
    for s in "${STAGES[@]}"; do
        [[ "$s" == "$START_STAGE" ]] && started=true
        [[ "$s" == "$stage" && "$started" == true ]] && return 0
    done
    return 1
}

# ── 메인 실행 ────────────────────────────────────────────────────────────────
log "Progressive Training 시작 (시작 스테이지: ${START_STAGE})"
log "체크포인트 디렉토리: ${CKPT_DIR}"

# Stage B+: 외력 없음, base_mass 없음, LR=5e-5 고정
if stage_should_run "Bplus"; then
    run_stage "Bplus" \
        "Velocity-Hylion-BG-Bplus-v0" \
        "$BPLUS_CKPT" \
        "${LOG_DIR}/hylion_v6_stageBplus.log" \
        "${CKPT_DIR}/stage_bplus_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=0" \
        "HYLION_BASE_MASS_ADD_KG=0.0" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.02"
fi

# Stage C1: base_mass ±0.5kg (환경변수로 env_cfg_BG에서 제어), 외력 없음
if stage_should_run "C1"; then
    run_stage "C1" \
        "Velocity-Hylion-BG-C1-v0" \
        "$C1_CKPT" \
        "${LOG_DIR}/hylion_v6_stageC1.log" \
        "${CKPT_DIR}/stage_c1_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=0" \
        "HYLION_BASE_MASS_ADD_KG=0.5" \
        "HYLION_MAX_LIN_VEL_X=0.5" \
        "HYLION_STANDING_RATIO=0.02"
fi

# Stage C2: base_mass ±0.5kg, 외력 ±3N
if stage_should_run "C2"; then
    run_stage "C2" \
        "Velocity-Hylion-BG-C2-v0" \
        "$C2_CKPT" \
        "${LOG_DIR}/hylion_v6_stageC2.log" \
        "${CKPT_DIR}/stage_c2_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=0.5" \
        "HYLION_PERTURB_FORCE=3.0" \
        "HYLION_PERTURB_TORQUE=1.0" \
        "HYLION_MAX_LIN_VEL_X=0.55" \
        "HYLION_STANDING_RATIO=0.05"
fi

# Stage C3: base_mass ±1.0kg, 외력 ±5N
if stage_should_run "C3"; then
    run_stage "C3" \
        "Velocity-Hylion-BG-C3-v0" \
        "$C3_CKPT" \
        "${LOG_DIR}/hylion_v6_stageC3.log" \
        "${CKPT_DIR}/stage_c3_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=1.0" \
        "HYLION_PERTURB_FORCE=5.0" \
        "HYLION_PERTURB_TORQUE=2.0" \
        "HYLION_MAX_LIN_VEL_X=0.6" \
        "HYLION_STANDING_RATIO=0.07"
fi

# Stage C4: base_mass randomized (-0.3~1.5kg), 외력 ±10N (최종)
if stage_should_run "C4"; then
    run_stage "C4" \
        "Velocity-Hylion-BG-C4-v0" \
        "$C4_CKPT" \
        "${LOG_DIR}/hylion_v6_stageC4.log" \
        "${CKPT_DIR}/stage_c4_hylion_v6" \
        "HYLION_ENABLE_PERTURBATION=1" \
        "HYLION_BASE_MASS_ADD_KG=0.0" \
        "HYLION_PERTURB_FORCE=10.0" \
        "HYLION_PERTURB_TORQUE=3.0" \
        "HYLION_MAX_LIN_VEL_X=0.7" \
        "HYLION_STANDING_RATIO=0.10"
fi

log "======================================================"
log "Progressive Training 전체 완료!"
log "최종 체크포인트: ${CKPT_DIR}/stage_c4_hylion_v6/best.pt"
log "======================================================"
