#!/bin/bash
# 학습 실시간 모니터링 스크립트
# 사용법: bash δ3/scripts/monitor_training.sh
# 종료: Ctrl+C

PYTHON=/home/laba/env_isaaclab/bin/python
LOG_ROOT="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion"
INTERVAL=${1:-10}  # 갱신 주기(초), 기본 10초

# 실패 판단 임계값
ORIENT_WARN=0.15
ORIENT_CRIT=0.30
REWARD_CRIT=10.0

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

while true; do
    clear
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  Hylion v6 학습 모니터   $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

    $PYTHON -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os, glob

log_root = '$LOG_ROOT'
ORIENT_WARN = $ORIENT_WARN
ORIENT_CRIT = $ORIENT_CRIT
REWARD_CRIT = $REWARD_CRIT

RED    = '\033[0;31m'
YELLOW = '\033[1;33m'
GREEN  = '\033[0;32m'
CYAN   = '\033[0;36m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

dirs = sorted(glob.glob(f'{log_root}/2026-*/'), key=os.path.getmtime)
if not dirs:
    print('로그 없음')
    exit()

latest = dirs[-1]
ea = EventAccumulator(latest)
ea.Reload()

try:
    r = ea.Scalars('Train/mean_reward')
    o = ea.Scalars('Episode_Termination/base_orientation')
    el = ea.Scalars('Train/mean_episode_length')
except:
    print('데이터 로딩 중...')
    exit()

cur_iter   = r[-1].step
cur_reward = r[-1].value
cur_orient = o[-1].value
cur_eplen  = el[-1].value

# 파라미터 파일에서 외력/max_iter 읽기
env_yaml = os.path.join(latest, 'params/env.yaml')
agent_yaml = os.path.join(latest, 'params/agent.yaml')
max_iter = '?'
force_val = '?'
stage_name = os.path.basename(latest.rstrip('/'))

if os.path.exists(agent_yaml):
    with open(agent_yaml) as f:
        for line in f:
            if 'max_iterations' in line:
                max_iter = int(line.split(':')[1].strip())
                break

if os.path.exists(env_yaml):
    lines = open(env_yaml).readlines()
    for i, line in enumerate(lines):
        if 'force_range:' in line and i+1 < len(lines):
            force_val = lines[i+2].strip().replace('- ','').strip() if i+2 < len(lines) else '?'
            try:
                force_val = f'±{abs(float(force_val)):.0f}N'
            except:
                force_val = '없음'
            break

# 진행률
if max_iter != '?':
    start_iter = r[0].step
    done = cur_iter - start_iter
    total = max_iter
    pct = done / total * 100
    bar_len = 30
    filled = int(bar_len * done / total)
    bar = '█' * filled + '░' * (bar_len - filled)
else:
    pct = 0; bar = '?' * 30; done = 0; total = '?'

# 상태 색상
if cur_orient > ORIENT_CRIT:
    o_color = RED
    status = f'{RED}🚨 즉시 중단 검토{RESET}'
elif cur_orient > ORIENT_WARN:
    o_color = YELLOW
    status = f'{YELLOW}⚠ 경고 — 주의 필요{RESET}'
else:
    o_color = GREEN
    status = f'{GREEN}✅ 정상{RESET}'

if cur_reward < REWARD_CRIT:
    r_color = RED
else:
    r_color = GREEN

print(f'  로그 세션  : {stage_name}')
print(f'  외력 설정  : {BOLD}{force_val}{RESET}')
print()
print(f'  진행률     : [{CYAN}{bar}{RESET}] {pct:.1f}%')
print(f'             : {done}/{total} iter ({cur_iter} 절대)')
print()
print(f'  평균 보상  : {r_color}{BOLD}{cur_reward:.1f}{RESET}')
print(f'  orientation: {o_color}{BOLD}{cur_orient*100:.2f}%{RESET}  (기준: 15% 미만)')
print(f'  에피소드길이: {cur_eplen:.0f} 스텝')
print()
print(f'  상태       : {status}')
print()

# 최근 10개 추세
print(f'  {BOLD}최근 10 iter 추세{RESET}')
print(f'  {\"iter\":>7}  {\"reward\":>8}  {\"orient\":>8}')
print(f'  {\"─\"*7}  {\"─\"*8}  {\"─\"*8}')
for rv, ov in zip(r[-10:], o[-10:]):
    rc = GREEN if rv.value > REWARD_CRIT else RED
    oc = GREEN if ov.value < ORIENT_WARN else (YELLOW if ov.value < ORIENT_CRIT else RED)
    print(f'  {rv.step:>7}  {rc}{rv.value:>8.1f}{RESET}  {oc}{ov.value*100:>7.2f}%{RESET}')
" 2>/dev/null

    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "  갱신 주기: ${INTERVAL}초 | 종료: Ctrl+C"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

    sleep "$INTERVAL"
done
