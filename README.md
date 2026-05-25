# DGX-NUC — Hylion 보행 학습 & 검증

DGX(학습) ↔ NUC(sim-to-sim / sim-to-real) 역할 분리 레포지토리.

---

## 머신별 역할

| 머신 | 역할 | 사용 폴더 |
|------|------|----------|
| **DGX** | IsaacLab RL 학습 (Newton 백엔드) | `dgx/`, `sim/isaaclab/`, `checkpoints/` |
| **NUC** | MuJoCo sim-to-sim 검증, 실로봇 배포 | `nuc/`, `sim/mujoco/`, `checkpoints/` |
| **공유** | 체크포인트 동기화, 통신 프로토콜 | `checkpoints/`, `comm/`, `configs/` |

---

## 폴더 구조

```
DGX-NUC/
│
├── dgx/                              ← [DGX 전용] 학습 진입점
│   ├── train_biped.sh                # 보행 학습 실행
│   └── requirements_dgx.txt
│
├── sim/
│   ├── isaaclab/                     ← [DGX 전용] IsaacLab 학습 환경
│   │   ├── hylion/                   # env_cfg, robot_cfg, PPO config
│   │   ├── scripts/
│   │   │   ├── newton/               # Newton 백엔드 학습 (현재 사용)
│   │   │   └── physx/                # PhysX 백엔드 (레거시)
│   │   └── robot/
│   │       ├── hylion_v6.urdf        # URDF (레거시)
│   │       ├── hylion_v6.xml         # MJCF — sim2sim 검증용 (compiler radian)
│   │       └── hylion_v7.xml         # MJCF — BHL biped 호환 + SO-ARM
│   │
│   └── mujoco/                       ← [NUC 전용] sim-to-sim 검증
│       ├── play_mujoco.py            # 메인 검증 (MJCF 기반)
│       ├── run_sim2sim.sh            # 래퍼
│       └── README.md                 # 사용법 + 트러블슈팅
│
├── checkpoints/biped/                ← [공유] DGX가 쓰고 NUC가 읽음
│   ├── stage_a/b/bplus/c1~c4/        # 초기 학습 단계
│   ├── stage_d1~d5/                  # 외란 ±1~10N 커리큘럼
│   └── stage_e1~e4/                  # 외란 ±15~30N 커리큘럼
│
├── nuc/bhl/                          ← [NUC 전용] 실로봇 보행 인터페이스
├── comm/                             ← [공유] NUC-Orin 통신 프로토콜
├── configs/                          ← [공유] 환경별 설정
├── tests/                            ← [NUC] HW 연결/단위/통합 테스트
└── docs/
    ├── ETHAN/active/                 # 진행 중 plan / snapshot
    ├── ETHAN/archive/                # outdated 문서 보관
    └── jimmy/                        # Jimmy 작업 기록
```

---

## 현재 학습 상태 (2026-04-30)

### 체크포인트 검증 결과

| Stage | 외란 (학습) | IsaacLab 학습 reward | 학습 fall % | MuJoCo 무외란 보행 |
|-------|----------:|---------------------:|----------:|:----------------:|
| D2.5  | ±2.5 N   | (정상 수렴) | <15% | ✅ |
| D3    | ±3 N     | (정상 수렴) | <15% | ✅ |
| D4    | ±5 N     | **+20.20** | **13%** | ✅ |
| D4.5  | ±7 N     | +14.14 | 25% | ✅ |
| D5    | ±10 N    | +11.42 | 32% | ✅ |
| E1    | ±15 N    | +3.04  | 81% | ✅ (Jimmy 4-29) |
| E2    | ±20 N    | -0.06  | 100% | ✅ (Jimmy 4-29) |
| E3    | ±25 N    | -0.03  | 100% | ✅ (Jimmy 4-29) |
| E3.5  | ±27 N    | -0.04  | 100% | (재학습본) |
| E4    | ±30 N    | -0.15  | 100% | ✅ (Jimmy 4-29) |

**해석:**
- **D4까지가 외란 환경에서도 안정**적으로 학습됨
- E1 부터 ±10→±15N 50% 점프로 외란 환경에서 회복 불가, 100% 넘어짐
- 그러나 **무외란 평지 보행은 모든 단계에서 가능** (4-29 MuJoCo 검증)
- 즉 "정책이 망가진" 게 아니라 "외란 robustness 학습이 D5에서 멈춤"

### 다음 목표

**±50N 외란 robustness + MuJoCo sim2sim 보행** 을 발표 마감(2026-06-01)까지 달성.  
계획서: [`docs/ETHAN/active/32_training_plan_50N_2026-04-30.md`](docs/ETHAN/active/32_training_plan_50N_2026-04-30.md)

---

## 빠른 시작

### DGX — 학습 (Newton 백엔드)

```bash
# 전체 커리큘럼 자동 실행 (D3 ~ E4)
nohup bash sim/isaaclab/scripts/newton/run_newton_training.sh \
  > /tmp/hylion_newton_orchestrator.log 2>&1 &

# 특정 스테이지부터 재개
START_STAGE=D5 bash sim/isaaclab/scripts/newton/run_newton_training.sh
```

### NUC — sim-to-sim 검증

```bash
# 헤드리스
python3 sim/mujoco/play_mujoco.py --no-viewer --vx 0.3 --duration 10.0

# GUI
DISPLAY=:0 python3 sim/mujoco/play_mujoco.py --vx 0.3 --duration 15.0
```

자세한 사용법: [`sim/mujoco/README.md`](sim/mujoco/README.md)

---

## 핵심 사실 (2026-04-30 검증된 것만)

1. **MuJoCo sim2sim 작동함** — 4-29에 `<compiler angle="radian">` 누락 버그 수정 후 모든 체크포인트 보행 확인
2. **백엔드는 Newton** — 4-28부터 PhysX → Newton 전환 (DGX Spark aarch64 호환성)
3. **D4가 안전한 마지막 정책** — 외란 환경에서도 검증됨 (fall 13%)
4. **E1 이상은 외란 학습이 부족** — 무외란 보행은 가능
