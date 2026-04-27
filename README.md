# DGX-NUC — Hylion 보행 학습 & 검증

DGX(학습) ↔ NUC(sim-to-sim / sim-to-real) 역할 분리 레포지토리.

---

## 머신별 역할

| 머신 | 역할 | 사용 폴더 |
|------|------|----------|
| **DGX** | IsaacLab RL 학습 | `dgx/`, `sim/isaaclab/`, `checkpoints/` |
| **NUC** | MuJoCo sim-to-sim 검증, 실로봇 배포 | `nuc/`, `sim/mujoco/`, `checkpoints/` |
| **공유** | 체크포인트 동기화, 통신 프로토콜 | `checkpoints/`, `comm/`, `configs/` |

---

## 폴더 구조

```
DGX-NUC/
│
├── dgx/                          ← [DGX 전용] 학습 진입점
│   ├── train_biped.sh            # 보행 학습 실행 (현재: Stage D4 ±4N)
│   └── requirements_dgx.txt     # DGX 의존성
│
├── sim/
│   ├── isaaclab/                 ← [DGX 전용] IsaacLab 학습 환경
│   │   ├── hylion/               # 환경 설정 (env_cfg, robot_cfg, PPO)
│   │   ├── scripts/              # 학습/평가 스크립트
│   │   └── robot/                # URDF, XML (학습 로봇 정의)
│   │
│   └── mujoco/                   ← [NUC 전용] sim-to-sim 검증
│       ├── play_mujoco.py        # policy 검증 실행
│       ├── run_sim2sim.sh        # 실행 래퍼
│       └── SIM2SIM_PROGRESS.md  # 검증 진행 기록
│
├── checkpoints/biped/            ← [공유] DGX가 쓰고 NUC가 읽음
│   ├── stage_d3_hylion_v6/best.pt   ← 실제 마지막 학습 완료
│   ├── stage_d4_hylion_v6/best.pt   ← (d3 복사본, 재학습 필요)
│   └── stage_d5_hylion_v6/best.pt   ← (d3 복사본, 재학습 필요)
│
├── nuc/bhl/                      ← [NUC 전용] 실로봇 보행 인터페이스
│
├── comm/                         ← [공유] NUC-Orin 통신 프로토콜
├── configs/                      ← [공유] 환경별 설정
│
├── tests/                        ← [NUC] HW 연결/단위/통합 테스트
└── docs/                         ← 개발 문서
```

---

## 현재 학습 상태 (2026-04-27)

| 스테이지 | 외력 | 실제 학습 여부 |
|---------|------|-------------|
| B+ ~ D3 | ±3N | ✅ 완료 |
| **D4** | **±4N** | 🔄 **진행 예정** (`dgx/train_biped.sh`) |
| D5 ~ E4 | ±5N → ±30N | ⏳ 대기 |

---

## 빠른 시작

### DGX — D4 학습 시작
```bash
tmux new-session -d -s hylion_train
tmux send-keys -t hylion_train \
  "bash /home/laba/DGX-NUC/dgx/train_biped.sh" Enter
tail -f /tmp/hylion_stageD4_4N.log
```

### NUC — sim-to-sim 검증
```bash
DISPLAY=:0 python3 /home/laba/DGX-NUC/sim/mujoco/play_mujoco.py \
  --ckpt /home/laba/DGX-NUC/checkpoints/biped/stage_d3_hylion_v6/best.pt \
  --urdf /home/laba/DGX-NUC/sim/isaaclab/robot/hylion_v6.urdf \
  --vx 0.0 --duration 10.0
```
