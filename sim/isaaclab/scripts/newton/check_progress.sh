#!/usr/bin/env bash
# 학습 진행 상황 빠른 조회
# 사용법: bash sim/isaaclab/scripts/newton/check_progress.sh
echo "================================================================"
echo "  Hylion 50N Curriculum — 현재 상태"
echo "  $(date '+%F %T')"
echo "================================================================"
echo ""
echo "── 최근 진행 알림 (마지막 30줄) ──"
tail -30 /tmp/hylion_progress.log 2>/dev/null || echo "  (아직 시작 안 됨)"
echo ""
echo "── 현재 학습 중인 iter (마지막 5줄) ──"
ls -t /tmp/hylion_50N_*.log 2>/dev/null | head -1 | xargs -I{} sh -c '
  echo "    log: {}"
  grep -E "Learning iteration|Mean reward:" {} | tail -10
' 2>/dev/null
echo ""
echo "── 오케스트레이터 프로세스 ──"
pgrep -af "run_50N_curriculum\|train_hylion_newton_BG" || echo "  실행 중 아님"
echo ""
echo "── 체크포인트 디스크 사용 ──"
du -sh /home/laba/DGX-NUC/checkpoints/biped/ 2>/dev/null
