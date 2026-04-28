#!/usr/bin/env bash
# =============================================================================
# train_biped.sh  — Stage D3 재도전 (±3N) — Newton 백엔드
#
# 상황: D3 best.pt가 NaN으로 오염됨 (2026-04-27 확인)
#       → D2.5 (±2.5N, 마지막 정상 체크포인트)에서 재시작
#
# 주요 변경점 (NaN 방지):
#   - D3 PPO config: num_learning_epochs 3 → 2 (rsl_rl_ppo_cfg_stageD_optionA.py 수정됨)
#   - resume: stage_d2_5_hylion_v6/best.pt
#   - 물리 백엔드: PhysX CPU → Newton GPU (DGX Spark aarch64 필수)
#
# 사용법:
#   bash /home/laba/DGX-NUC/dgx/train_biped.sh
#
# 로그 확인:
#   tail -f /tmp/hylion_stageD3_newton.log
#
# 500 iter 조기 판단 기준 (실패 시 즉시 Ctrl+C):
#   orientation termination > 30%  AND  mean_reward < 10 (하락 중)
#   → 즉시 중단 후 D3 force를 ±2.8N으로 줄여서 재시도
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # /home/laba/DGX-NUC
BHL_DIR="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl"
TRAIN_SCRIPT="$REPO_ROOT/sim/isaaclab/scripts/newton/train_hylion_newton_BG.py"
PYTHON_BIN="/home/laba/env_isaaclab/bin/python"

RESUME_CKPT="$REPO_ROOT/checkpoints/biped/stage_d2_5_hylion_v6/best.pt"
OUT_DIR="$REPO_ROOT/checkpoints/biped/stage_d3_hylion_v6"
LOG="/tmp/hylion_stageD3_newton.log"

if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "[ERROR] D2.5 체크포인트 없음: $RESUME_CKPT"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "[INFO] Stage D3 재도전 (±3N) — Newton 백엔드, D2.5에서 resume"
echo "[INFO] Resume: $RESUME_CKPT"
echo "[INFO] 로그:   tail -f $LOG"
echo "[INFO] 500 iter 후 orientation < 15%, reward > 20 확인 필요"

cd "$BHL_DIR"
source /home/laba/env_isaaclab/bin/activate
unset PYTHONPATH || true
unset PYTHONHOME || true

export HYLION_ENABLE_PERTURBATION=1
export HYLION_BASE_MASS_ADD_KG=0.5
export HYLION_PERTURB_FORCE=3.0
export HYLION_PERTURB_TORQUE=1.0
export HYLION_MAX_LIN_VEL_X=0.5
export HYLION_STANDING_RATIO=0.05

PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --task Velocity-Hylion-BG-D3-v0 \
    --num_envs 4096 \
    --headless \
    --pretrained_checkpoint "$RESUME_CKPT" \
    2>&1 | tee "$LOG"

# 완료 후 best.pt 저장
# Newton 스크립트는 BHL_DIR 기준 logs/ 에 저장됨
PROJ_LOGS="${BHL_DIR}/logs/rsl_rl/hylion"
LATEST_DIR=$(ls -dt "${PROJ_LOGS}"/2026-*/ 2>/dev/null | head -1)
LATEST_PT=$(ls "${LATEST_DIR}"model_*.pt 2>/dev/null | sort -V | tail -1)

if [[ -n "$LATEST_PT" ]]; then
    cp "$LATEST_PT" "${OUT_DIR}/best.pt"
    echo "[INFO] 저장 완료: ${OUT_DIR}/best.pt"
else
    echo "[WARN] 체크포인트 자동 저장 실패. 수동으로 저장하세요."
    echo "[HINT] ls ${PROJ_LOGS}/\$(ls -t ${PROJ_LOGS}/ | head -1)/model_*.pt | sort -V | tail -1"
fi

