#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:-/tmp/hylion_v6_physx_M2.log}"
INTERVAL="${2:-2}"

if [[ ! -f "$LOG_FILE" ]]; then
  echo "[ERROR] Log file not found: $LOG_FILE"
  echo "Usage: bash monitor_stageb_realtime.sh /tmp/hylion_v6_physx_M2.log 2"
  exit 1
fi

while true; do
  clear
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " HYLION v6 학습 모니터  $(date '+%F %T')"
  echo " LOG: $LOG_FILE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # 프로세스 상태
  echo ""
  echo "[ 프로세스 ]"
  pgrep -af "train_hylion_physx_BG.py" 2>/dev/null | grep -v grep \
    && echo "  → 실행 중" || echo "  → ❌ 실행 중이 아님"

  # 최신 iteration 블록 추출 (마지막 완전한 블록)
  LAST_ITER=$(grep "Learning iteration" "$LOG_FILE" | tail -1)
  LAST_VLOSS=$(grep "Mean value loss:" "$LOG_FILE" | tail -1 | xargs)
  LAST_SLOSS=$(grep "Mean surrogate loss:" "$LOG_FILE" | tail -1 | xargs)
  LAST_REWARD=$(grep "Mean reward:" "$LOG_FILE" | tail -1 | xargs)
  LAST_EPLEN=$(grep "Mean episode length:" "$LOG_FILE" | tail -1 | xargs)
  LAST_STD=$(grep "Mean action std:" "$LOG_FILE" | tail -1 | xargs)
  LAST_AIR=$(grep "feet_air_time" "$LOG_FILE" | tail -1 | xargs)
  LAST_ERRXY=$(grep "error_vel_xy" "$LOG_FILE" | tail -1 | xargs)
  LAST_ERRYAW=$(grep "error_vel_yaw" "$LOG_FILE" | tail -1 | xargs)

  echo ""
  echo "[ 학습 진행 ]"
  echo "  $LAST_ITER"

  echo ""
  echo "[ 핵심 지표 ]"
  printf "  %-28s %s\n" "value loss"      "$LAST_VLOSS"
  printf "  %-28s %s\n" "surrogate loss"  "$LAST_SLOSS"
  printf "  %-28s %s\n" "reward"          "$LAST_REWARD"
  printf "  %-28s %s\n" "episode length"  "$LAST_EPLEN"
  printf "  %-28s %s\n" "action std"      "$LAST_STD"
  printf "  %-28s %s\n" "feet_air_time"   "$LAST_AIR"
  printf "  %-28s %s\n" "error_vel_xy"    "$LAST_ERRXY"
  printf "  %-28s %s\n" "error_vel_yaw"   "$LAST_ERRYAW"

  # 판정
  echo ""
  echo "[ 판정 ]"
  NAN_COUNT=$(grep -cE "nan|NaN" "$LOG_FILE" 2>/dev/null | grep -v "^0$" || true)
  TRACE_COUNT=$(grep -c "Traceback" "$LOG_FILE" 2>/dev/null || true)

  if [[ "$LAST_VLOSS" == *"nan"* ]] || [[ "$LAST_SLOSS" == *"nan"* ]]; then
    echo "  🔴 FAIL — loss NaN 발생 즉시 개입 필요"
  elif [[ "$LAST_STD" == *"0.00"* ]]; then
    echo "  🔴 FAIL — action std 붕괴 즉시 개입 필요"
  elif [[ "$LAST_AIR" == *"0.0000"* ]]; then
    echo "  🟡 경계 — feet_air_time 0 고정 확인 필요"
  else
    echo "  🟢 정상"
  fi

  # 에러
  echo ""
  echo "[ 에러 ]"
  grep -E "Traceback|ValueError|RuntimeError|contains NaN" "$LOG_FILE" 2>/dev/null \
    | tail -5 || echo "  없음"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " ${INTERVAL}초마다 갱신  |  Ctrl+C 로 종료"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  sleep "$INTERVAL"
done
