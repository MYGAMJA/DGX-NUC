# δ3 — Hylion v6 보행학습 (2026-04-15 기준)

## 디렉토리 구조

```
δ3/
├── README.md
│
├── hylion/                      ← 학습 환경 설정 (핵심)
│   ├── env_cfg.py               # 기본 환경 (보상/관측/명령 정의)
│   ├── env_cfg_BG.py            # v6 오버라이드 (contact sensor 등)
│   ├── robot_cfg.py             # 기본 로봇 설정
│   ├── robot_cfg_BG.py          # v6 로봇 USD 경로, 관절 설정
│   └── agents/
│       └── rsl_rl_ppo_cfg.py    # PPO 하이퍼파라미터
│
├── scripts/                     ← 현재 사용 중인 스크립트
│   ├── train_hylion_physx_BG.py     # 메인 학습 스크립트
│   ├── train_hylion_physx_BG.sh     # 학습 실행 래퍼
│   ├── train_biped_physx.py         # Stage-A (BHL biped) 학습용
│   ├── run_v6_matrix_experiment.sh  # M1~M6 개별 런처
│   ├── run_v6_matrix_smoke_suite.sh # 공정 비교 스위트
│   ├── monitor_stageb_realtime.sh   # 실시간 로그 모니터
│   ├── auto_guard_hylion_train.sh   # 자동 NaN 감시/롤백
│   └── inspect_hylion_contact_state.py  # contact 상태 진단
│
├── docs/
│   ├── active/                  ← 현재 운영 문서
│   │   ├── 20_v6_command_capability_2026-04-15.md       # 명령 가능 범위
│   │   ├── 21_professor_report_timeline_workflow_2026-04-15.md  # 교수님 보고용
│   │   └── 22_project_structure_and_roles_2026-04-15.md # 전체 구조/역할
│   └── archive/                 ← 완료된 기록 (삭제 금지, 추후 참조용)
│       ├── 00~14_*.md           # 초기 환경 셋업, v3/v4 실험 기록
│       ├── 15~16_*.md           # Stage-B 초기 진행 / NaN 복구 보고
│       ├── 17_*.md              # contact sensor 근본 원인 분석
│       ├── 18_*.md              # v6_flat 단계 전환 전략
│       └── 19_*.md              # 전체 실험 이력 (가장 중요한 기록)
│
├── robot/                       ← 로봇 원본 소스 (URDF, STL, 수정 금지)
└── usd/                         ← δ3 초기 변환 자산 (현재 학습에 미사용)
```

---

## 현재 학습 자산 경로 (중요)

```
✅ 실제 사용 중:
   /home/laba/project_singularity/δ1 & ε2/usd/hylion_v6/hylion_v6.usda

❌ 미사용 (fixed-base 문제로 교체됨):
   /home/laba/project_singularity/δ3/usd/hylion_v6/...
```

---

## 체크포인트 경로

```
Stage-A 기점 (BHL biped):
  ~/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/biped/
    └── 2026-04-06_15-27-27/model_5999.pt   ← 학습 시작점

Stage-B 체크포인트:
  ~/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/
    └── 2026-04-15_.../                     ← 현재 진행 중
```

---

## 실험 설정 매트릭스 (M1~M6)

| 설정 | 상체질량(A) | 액추에이터게인(B) | air_threshold | 단기 결과 |
|------|------------|-----------------|---------------|-----------|
| M1 | 0.0 | 1.0 | 0.2 | PASS, air=0.0009 |
| M2 | 0.0 | 1.2 | 0.2 | PASS, air=0.0016 ← 장기 후보 |
| M3 | 0.3 | 1.0 | 0.2 | PASS, air=0.0014 |
| M4 | 0.3 | 1.2 | 0.2 | PASS, air=0.0018 ← 장기 후보 |
| M5 | 0.6 | 1.2 | 0.2 | PASS, air=0.0012 |
| M6 | 1.0 | 1.2 | 0.2 | PASS, air=0.0014 |

현재: M2 장기 관찰 완료 → **M4 장기 런 진행 중**

---

## ⚠️ 학습 실행 규칙 — 반드시 tmux 안에서 실행

**모든 학습은 tmux 세션 안에서 실행한다. nohup 단독 사용 금지.**
이유: VS Code / SSH 연결이 끊겨도 학습이 살아있어야 하고,
노트북에서 SSH 접속 후 즉시 화면을 볼 수 있어야 하기 때문.

```bash
# ── 학습 시작 (표준 방법) ─────────────────────────────────────
# 세션이 없을 때만 생성 (있으면 그냥 attach)
tmux new-session -d -s hylion_train -x 220 -y 50 2>/dev/null || true

# tmux 안에서 Option A 파이프라인 실행
tmux send-keys -t hylion_train \
  "cd /home/laba/project_singularity && unset PYTHONPATH && unset PYTHONHOME && \
  START_STAGE=D2.5 bash δ3/scripts/run_optionA_training.sh 2>&1 | tee /tmp/hylion_optionA_orchestrator.log" \
  Enter

# ── 개인 노트북에서 확인하는 법 ──────────────────────────────
# 1. 서버 SSH 접속
ssh laba@[서버IP]

# 2. 실행 중인 tmux 세션 확인
tmux ls

# 3. 세션에 붙기 (학습 화면 실시간 확인)
tmux attach -t hylion_train

# 4. 학습은 그대로 두고 세션에서 나가기 (절대 Ctrl+C 누르지 말 것!)
Ctrl+B, D

# ── 로그 파일로 확인 (세션 안 들어가지 않고) ─────────────────
tail -f /tmp/hylion_optionA_orchestrator.log

# ── 현재 어느 스테이지인지 한 줄 확인 ────────────────────────
ps aux | grep "train_hylion_physx_BG" | grep -v grep | grep -oP "\-\-task \S+"

# ── 학습 강제 중단 (응급 시만) ───────────────────────────────
tmux send-keys -t hylion_train C-c
```

---

## Option A 학습 파이프라인 (현재 진행 중 — 2026-04-22)

```
stage_d2_hylion_v6/best.pt  (±2N, orient 3.8%)  ← 시작 체크포인트
  → D2.5  ±2.5N  2000iter   /tmp/hylion_v6_stageD2_5.log
  → D3    ±3N    4000iter   /tmp/hylion_v6_stageD3.log
  → D4    ±5N    4000iter   /tmp/hylion_v6_stageD4.log
  → D4.5  ±7N    4000iter   /tmp/hylion_v6_stageD4_5.log
  → D5    ±10N   5000iter   /tmp/hylion_v6_stageD5.log

체크포인트 저장 위치:
  δ3/checkpoints/stage_d2_5_hylion_v6/best.pt
  δ3/checkpoints/stage_d3_hylion_v6/best.pt
  δ3/checkpoints/stage_d4_hylion_v6/best.pt
  δ3/checkpoints/stage_d4_5_hylion_v6/best.pt
  δ3/checkpoints/stage_d5_hylion_v6/best.pt

스테이지 재시작 (특정 단계부터):
  START_STAGE=D3 bash δ3/scripts/run_optionA_training.sh ...
  START_STAGE=D4 bash δ3/scripts/run_optionA_training.sh ...
```

---

## 자주 쓰는 명령

---

## 판정 기준 (매번 확인)

| 지표 | 정상 | 경계 | 즉시 중단 |
|------|------|------|-----------|
| value/surrogate loss | 유한값 안정 | 급등/급락 반복 | nan |
| action std | > 0.1 | 0.05~0.1 | 0.00 고정 20iter+ |
| feet_air_time | > 0.001 | 0.0001 미만 | 0.0000 고정 150iter+ |
| NaN/Traceback | 0건 | — | 1건이라도 |

---

## 현재 진행 단계 (2026-04-15)

```
완료 ✅  Stage-A 학습 (biped model_5999.pt)
완료 ✅  contact sensor 근본 수정 (USD 교체 + contact_sensor.py 패치)
완료 ✅  M1~M6 공정 비교 (전체 PASS)
완료 ✅  M2 장기 런 관찰
진행 🔄  M4 장기 런
다음 ⏳  M2 vs M4 최종 비교 → 주력 설정 확정
다음 ⏳  command tracking 수치 측정
다음 ⏳  이번 주 데모 제작
```
