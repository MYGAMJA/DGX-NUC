#!/usr/bin/env bash
# monitor_progressive.sh — progressive 학습 진행 상황 한눈에 보기
# 사용: bash /home/laba/project_singularity/δ3/scripts/monitor_progressive.sh

STAGES_LOGS=(
  "B+:/tmp/hylion_v6_stageBplus.log"
  "C1:/tmp/hylion_v6_stageC1.log"
  "C2:/tmp/hylion_v6_stageC2.log"
  "C3:/tmp/hylion_v6_stageC3.log"
  "C4:/tmp/hylion_v6_stageC4.log"
)
ORCH_LOG="/tmp/hylion_progressive_orchestrator.log"

echo "======================================================"
echo " Hylion Progressive Training 모니터 ($(date '+%F %T'))"
echo "======================================================"

# 오케스트레이터 상태
echo ""
echo "[ 오케스트레이터 ]"
if pgrep -f "run_progressive_training.sh" > /dev/null 2>&1; then
  echo "  상태: 실행 중"
else
  echo "  상태: 종료됨"
fi
tail -3 "$ORCH_LOG" 2>/dev/null | sed 's/^/  /'

# 각 스테이지 상태
for entry in "${STAGES_LOGS[@]}"; do
  stage="${entry%%:*}"
  logfile="${entry##*:}"

  echo ""
  echo "[ Stage ${stage} ]  로그: ${logfile}"

  if [[ ! -f "$logfile" ]]; then
    echo "  대기 중 (아직 시작 안 됨)"
    continue
  fi

  steps=$(grep "Total steps:" "$logfile" 2>/dev/null | tail -1 | awk '{print $NF}')
  reward=$(grep "Mean reward:" "$logfile" 2>/dev/null | tail -1 | awk '{print $NF}')
  orientation=$(grep "Episode_Termination/base_orientation:" "$logfile" 2>/dev/null | tail -1 | awk '{print $NF}')
  elapsed=$(grep "Time elapsed:" "$logfile" 2>/dev/null | tail -1 | awk '{print $NF}')
  eta=$(grep "ETA:" "$logfile" 2>/dev/null | tail -1 | awk '{print $NF}')

  if [[ -z "$steps" ]]; then
    echo "  초기화 중..."
    tail -2 "$logfile" 2>/dev/null | sed 's/^/  /'
    continue
  fi

  echo "  steps:       ${steps}"
  echo "  mean reward: ${reward}"
  echo "  orientation: ${orientation}  ← 목표: < 0.30"
  echo "  elapsed:     ${elapsed}"
  echo "  ETA:         ${eta}"

  # 수렴 판단
  if [[ -n "$orientation" ]]; then
    if awk "BEGIN {exit !(${orientation} < 0.15)}"; then
      echo "  ✓ 우수 (0.15 미만)"
    elif awk "BEGIN {exit !(${orientation} < 0.30)}"; then
      echo "  ✓ 양호 (0.30 미만)"
    else
      echo "  ⚠ 수렴 필요 (0.30 이상)"
    fi
  fi
done

echo ""
echo "======================================================"

# 현재 실행 중인 학습 프로세스
echo "[ 실행 중인 학습 프로세스 ]"
ps aux | grep "train_hylion" | grep -v grep | awk '{print "  PID:", $2, "| Task:", $(NF-5)}' 2>/dev/null || echo "  없음"
echo "======================================================"
