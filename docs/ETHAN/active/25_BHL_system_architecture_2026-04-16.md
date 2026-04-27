# BHL 시스템 전체 구조 및 신호체계 (2026-04-16)

---

## 1. 전체 개요

**BHL(Berkeley Humanoid Lite)**은 경량 2족 보행 로봇이며,  
**Hylion v6**은 BHL 다리에 SO-ARM 팔 2개를 붙인 파생 로봇이다.

이 문서는 BHL/Hylion v6의 하드웨어 구조, 소프트웨어 레이어, 신호 흐름, 명령 주체와 전달 방식을 통합 정리한다.

---

## 2. 하드웨어 구조

### 2-1. 관절 구성

```
BHL biped (다리 전용)
  ├── 좌측 다리 (6 DOF)
  │   ├── left_hip_yaw_joint
  │   ├── left_hip_roll_joint
  │   ├── left_hip_pitch_joint
  │   ├── left_knee_pitch_joint
  │   ├── left_ankle_pitch_joint
  │   └── left_ankle_roll_joint
  └── 우측 다리 (6 DOF)
      ├── right_hip_yaw_joint
      ├── right_hip_roll_joint
      ├── right_hip_pitch_joint
      ├── right_knee_pitch_joint
      ├── right_ankle_pitch_joint
      └── right_ankle_roll_joint

  총 12 DOF (정책 제어 대상)

Hylion v6 = BHL biped + SO-ARM × 2
  └── SO-ARM joints: 좌우 각 12개 = 24개 추가
      → 정책에서 제외 (stiffness=50으로 고정 자세 유지)
  총 revolute joints: 36개 (다리 12 + 팔 24)
  정책 제어 joints: 12개 (다리만)
```

### 2-2. 모터 사양

| 위치 | 모터 모델 | KV | 토크 상수 | 연속 전류 | 노력 한계 | 속도 한계 |
|------|----------|-----|----------|----------|----------|----------|
| 고관절/무릎 (6개) | MAD M6C12 | 150 KV | 0.0919 Nm/A | 20.0 A | 6.0 Nm | 10.0 rad/s |
| 발목 (4개) | MAD 5010 | 110 KV | 0.1176 Nm/A | 20.0 A | 6.0 Nm | 10.0 rad/s |

### 2-3. 기어박스

```
사이클로이드 감속기
  기어비: -15.0  (음수 = 방향 반전)
  적용 범위: 모든 관절 동일
```

### 2-4. IsaacLab 액추에이터 파라미터 (시뮬 기준)

| 관절 부위 | stiffness | damping | armature |
|----------|-----------|---------|----------|
| 고관절/무릎 | 20.0 Nm/rad | 2.0 Nm·s/rad | 0.007 kg·m² (hip), 0.002 (knee) |
| 발목 | 20.0 Nm/rad | 2.0 Nm·s/rad | 0.002 kg·m² |

---

## 3. 소프트웨어 레이어 구조

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 4  실기 배포 (미구현)                              │
│           sim-to-real 보정, 안전 게이트, 속도 제한         │
├─────────────────────────────────────────────────────────┤
│  LAYER 3  명령 추종 검증 / 데모                           │
│           "N초 동안 앞으로/옆으로/회전" 시나리오            │
│           DEMO_SEQUENCE 변수로 시나리오 정의               │
├─────────────────────────────────────────────────────────┤
│  LAYER 2  강화학습 정책 (현재 운용 중)                     │
│           입력: (lin_vel_x, lin_vel_y, ang_vel_z)        │
│           출력: 12 DOF 관절 목표 각도                      │
│           알고리즘: PPO (Actor-Critic)                    │
├─────────────────────────────────────────────────────────┤
│  LAYER 1  시뮬레이터 / 물리 환경                           │
│           IsaacLab + PhysX (데스크탑) / Newton (DGX)     │
│           Hylion v6 USD 자산                              │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 신호 체계 상세

### 4-1. 정책 입력 (관측값, obs)

정책이 매 스텝 받는 관측 벡터 (총 45차원):

| 신호 | 차원 | 설명 | 노이즈 |
|------|------|------|--------|
| `velocity_commands` | 3 | 목표 속도 명령 (lin_vel_x, lin_vel_y, ang_vel_z) | 없음 |
| `base_ang_vel` | 3 | base 각속도 (roll/pitch/yaw rate) | ±0.3 rad/s |
| `projected_gravity` | 3 | 중력 방향 투영 (기울기 신호) | ±0.05 |
| `joint_pos` | 12 | 12개 다리 관절 상대 위치 | ±0.05 rad |
| `joint_vel` | 12 | 12개 다리 관절 속도 | ±2.0 rad/s |
| `last_action` | 12 | 직전 스텝 출력 행동 | 없음 |

> Critic만 추가로 `base_lin_vel` (3차원, 노이즈 없음) 수신.  
> Policy 관측은 학습 중 노이즈 추가 (enable_corruption=True).

### 4-2. 정책 출력 (행동, action)

```
출력: 12개 관절 목표 각도 (delta from default pose)
  스케일: × 0.25 (작은 delta로 안정 제어)
  제어 방식: JointPositionAction (목표 각도 → PD 제어기)

  PD 게인:
    stiffness = 20.0 Nm/rad
    damping   = 2.0 Nm·s/rad
```

### 4-3. 명령 입력 (Command)

```
명령 종류: UniformVelocityCommand (속도 기반)

범위 (학습 시):
  lin_vel_x = (-0.5, +0.5) m/s   ← 전후 이동 속도
  lin_vel_y = (-0.25, +0.25) m/s ← 좌우 이동 속도
  ang_vel_z = (-1.0, +1.0) rad/s ← 제자리 회전 속도
  heading   = (-π, +π)

resampling_time: 10초마다 새 명령 샘플링
rel_standing_envs: 2% (정지 서기 훈련)
```

### 4-4. 제어 주기 (timing)

```
IsaacLab physics step  → 200 Hz (0.005 s)
Policy 제어 (decimation=8)  → 25 Hz (0.04 s)
실로봇 저수준 제어 (ESC)     → 250 Hz
CAN 버스                    → 1 Mbps
```

---

## 5. 보상 함수 구조

정책이 학습하는 보상 신호 구성:

| 신호 | 종류 | 가중치 | 역할 |
|------|------|--------|------|
| `track_lin_vel_xy_exp` | + | 2.0 | 전후좌우 속도 명령 추종 |
| `track_ang_vel_z_exp` | + | 1.0 | yaw 속도 명령 추종 |
| `feet_air_time` | + | 1.5 | 발 들기 신호 (핵심 보행 신호) |
| `termination_penalty` | − | 10.0 | 넘어짐 페널티 |
| `lin_vel_z_l2` | − | 0.1 | 위아래 튀기 억제 |
| `ang_vel_xy_l2` | − | 0.05 | 기울어짐 억제 |
| `flat_orientation_l2` | − | 2.0 | 직립 자세 유지 |
| `action_rate_l2` | − | 0.01 | 급격한 동작 억제 |
| `dof_torques_l2` | − | 0.002 | 관절 토크 최소화 |
| `dof_acc_l2` | − | 2.5e-7 | 관절 가속도 억제 |
| `dof_pos_limits` | − | 1.0 | 관절 범위 이탈 억제 |
| `feet_slide` | − | 0.1 | 발 미끄러짐 억제 |
| `joint_deviation_hip` | − | 0.2 | hip 관절 기본 자세 유지 |
| `joint_deviation_ankle_roll` | − | 0.2 | 발목 롤 기본 자세 유지 |

---

## 6. 명령 주체와 전달 방식

### 6-1. 명령 주체 계층

```
┌────────────────────────────────────────────────────────┐
│  L3: 사람 (또는 상위 경로 계획기)                        │
│      "어떤 시나리오로 걷게 할 것인가" 결정               │
│      예: DEMO_SEQUENCE = [(5.0, 0.2, 0, 0), ...]       │
├────────────────────────────────────────────────────────┤
│  L2: 속도 명령기 (Command Governor)                     │
│      lin_vel_x / lin_vel_y / ang_vel_z 값 생성          │
│      → 학습 중: IsaacLab UniformVelocityCommand         │
│      → 데모 중: DEMO_SEQUENCE 타임라인 재생              │
│      → 실기:   ROS2 /gait/cmd 토픽                      │
├────────────────────────────────────────────────────────┤
│  L1: 보행 정책 (Locomotion Policy)                      │
│      속도 명령 + 센서 관측 → 관절 목표 각도 출력          │
│      25 Hz 실행                                         │
├────────────────────────────────────────────────────────┤
│  L0: 액추에이터 제어기 (PD Controller + ESC)             │
│      관절 목표 각도 → 모터 전류 명령                      │
│      250 Hz 실행 / CAN 버스 전달                         │
└────────────────────────────────────────────────────────┘
```

### 6-2. 시뮬레이터 내 신호 흐름

```
[명령 샘플러]
  UniformVelocityCommand
  → (lin_vel_x, lin_vel_y, ang_vel_z) 생성
  → 관측 벡터에 포함

[IsaacLab 환경]
  매 physics step (200 Hz)
    → 물리 시뮬레이션 업데이트
    → 센서값 갱신 (관절 위치/속도, IMU, contact)

  매 제어 step (25 Hz, decimation=8)
    → obs 조합 → Policy 입력
    → Policy 추론 → action 출력 (12 관절 delta)
    → JointPositionAction → PD → 토크 적용

[contact_forces 센서]
  prim_path: {ENV_REGEX_NS}/robot/.*
  history_length: 3
  track_air_time: True
  → feet_air_time 보상 계산에 사용
```

### 6-3. 실로봇 신호 흐름 (설계안)

```
[Orin (상위 컴퓨터)] → [NUC (하위 컴퓨터)] → [ESC × 10] → [모터 × 10]

Orin → NUC  (ROS2 토픽)
  /gait/cmd       geometry_msgs/Twist   보행 속도 명령 (vx, vy, wz)   25 Hz
  /gait/mode      std_msgs/String       보행 모드 (WALK/STAND/STOP)    on-change
  /arm/cmd        sensor_msgs/JointState 팔 관절 목표 위치              30 Hz

NUC → Orin  (ROS2 토픽)
  /gait/status    std_msgs/String        보행 상태 피드백               25 Hz
  /joint/state    sensor_msgs/JointState 관절 위치/속도/토크            250 Hz
  /imu/data       sensor_msgs/Imu        가속도/자이로/쿼터니언          200 Hz
  /contact/state  std_msgs/Bool[4]       발바닥 접촉 여부               200 Hz

NUC ↔ ESC  (CAN 버스)
  프로토콜: CANopen 유사 (PDO/SDO)
  비트레이트: 1 Mbps
  디바이스 ID: 0~127 (7비트)
  워치독 타임아웃: 1000 ms
  제어 모드: CURRENT(0x10) / TORQUE(0x11) / POSITION(0x13)
```

---

## 7. 학습 파이프라인 (Stage-A → Stage-B)

```
[Stage-A] BHL biped 단독 학습
  로봇: berkeley_humanoid_lite_biped.usd
  관절: 다리 12개
  Task ID: Velocity-Berkeley-Humanoid-Lite-Biped-v0
  결과: model_5999.pt (6000 iter, ~5h 46m, Newton)

        ↓ 전이학습 (pretrained_checkpoint)

[Stage-B] Hylion v6 보행학습
  로봇: hylion_v6.usda (δ1 & ε2 경로)
  관절: 다리 12개 (SO-ARM 고정)
  Task ID: Velocity-Hylion-BG-v0
  물리: PhysX (데스크탑)
  현재: M4 장기 런 진행 중

        ↓ (예정)

[Stage-C] 외력 섭동 + 넓은 명령 범위
  외력 주입: ±5~10 N force, ±1~3 Nm torque
  명령 범위 확대: lin_vel_x ±0.7 m/s 등
  게인 랜덤화: ±15%
  (환경변수 HYLION_ENABLE_PERTURBATION=1로 활성화)
```

---

## 8. PPO 알고리즘 설정

```
네트워크
  Actor:  [256 → 128 → 128], activation=elu
  Critic: [256 → 128 → 128], activation=elu
  초기 noise std: 0.5

알고리즘 파라미터 (Stage-B 기준)
  clip_param          = 0.15
  entropy_coef        = 0.01
  num_learning_epochs = 5
  num_mini_batches    = 4
  learning_rate       = 1.0e-4
  schedule            = adaptive (KL 기반)
  gamma               = 0.99
  lam                 = 0.95
  desired_kl          = 0.01
  max_grad_norm       = 0.1
  num_steps_per_env   = 24
  max_iterations      = 6000
  save_interval       = 100
```

---

## 9. 도메인 랜덤화 (sim-to-real 대비)

학습 시 적용되는 랜덤화 항목:

| 항목 | 범위 | 시점 |
|------|------|------|
| 마찰계수 (static/dynamic) | 0.4 ~ 1.2 | startup |
| base 질량 추가 | 설정값 (M1~M6에 따라) | startup |
| 관절 기본 자세 | ±0.05 rad | startup |
| 액추에이터 게인 (stiffness/damping) | ×0.8 ~ ×1.2 | startup |
| base 리셋 위치 | x/y ±0.2 m, yaw ±0.4 rad | reset |
| 관절 리셋 위치 | ×0.9 ~ ×1.1 | reset |
| 외력 주입 (Stage-C) | ±5~10 N, ±1~3 Nm | interval 1.5~3.0초 |

---

## 10. 주요 파일 경로 인덱스

```
/home/laba/project_singularity/δ3/
│
├── hylion/
│   ├── env_cfg.py              # 명령/관측/보상/이벤트 정의 (기반)
│   ├── env_cfg_BG.py           # v6 전용 오버라이드 (contact, 랜덤화)
│   ├── robot_cfg.py            # v3 로봇 설정
│   ├── robot_cfg_BG.py         # v6 USD 경로, 관절 설정
│   └── agents/
│       └── rsl_rl_ppo_cfg.py   # PPO 하이퍼파라미터
│
├── scripts/
│   ├── train_hylion_physx_BG.py        # 메인 학습 스크립트
│   ├── play_hylion_v6_BG.py            # 데모 시각화 (DEMO_SEQUENCE)
│   ├── run_v6_matrix_experiment.sh     # M1~M6 개별 런처
│   ├── run_v6_matrix_smoke_suite.sh    # 공정 비교 스위트
│   ├── monitor_stageb_realtime.sh      # 실시간 로그 모니터
│   └── auto_guard_hylion_train.sh      # 자동 NaN 감시/롤백
│
/home/laba/project_singularity/δ1 & ε2/usd/hylion_v6/hylion_v6.usda
                                         ← 현재 사용 중인 로봇 USD 자산

/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/
├── biped/2026-04-06_15-27-27/model_5999.pt   ← Stage-A 기점
└── hylion/2026-04-15_13-36-12/model_11998.pt ← Stage-B 최신 체크포인트
```

---

## 11. 현재 정책의 명령 능력 요약

| 명령 종류 | 가능 여부 | 비고 |
|----------|----------|------|
| N초 동안 앞으로 0.2 m/s | ✅ 가능 | 가장 잘 됨 |
| N초 동안 좌우 이동 0.1 m/s | ✅ 구조상 가능 | 품질 검증 필요 |
| N초 동안 제자리 회전 0.5 rad/s | ✅ 구조상 가능 | 품질 검증 필요 |
| 전진+회전 동시 명령 | ✅ 가능 | |
| "(x, y) 좌표까지 가라" | ❌ 불가 | outer-loop planner 필요 |
| 장애물 회피 경로 | ❌ 불가 | 별도 경로 계획기 필요 |
| 자연어 명령 직접 입력 | ❌ 불가 | LLM 변환기 별도 필요 |

> 현재 정책은 **보행 엔진(locomotion policy)** 레이어만 구현된 상태.  
> "저 위치까지 가라"는 그 위에 **command governor → velocity converter** 레이어를 추가해야 한다.

---

## 12. 모니터링 판정 기준

| 지표 | 정상 ✅ | 경계 ⚠️ | 즉시 중단 🛑 |
|------|--------|---------|------------|
| value/surrogate loss | 유한값 안정 | 급등/급락 반복 | nan |
| action std | > 0.1 | 0.05 ~ 0.1 | 0.00 고정 20 iter+ |
| feet_air_time | > 0.001 | 0.0001 미만 | 0.0000 고정 150 iter+ |
| NaN/Traceback | 0건 | — | 1건이라도 |

---

## 13. 강건성 학습 진행 현황 (2026-04-17 업데이트)

### 13-1. 이전 진행 요약 (Stage B → C)

4/16 ~ 4/17에 걸쳐 Stage B (기본 보행) → C (외력/질량 변동 강건성) 방향으로 학습을 진행했으나,  
C2 이후 단계에서 전부 실패함.

| 스테이지 | 조건 | 반복 | 보상 | orientation 종료율 | 결과 |
|---------|------|------|------|-------------------|------|
| Stage B+ | 외력 없음, base_mass 없음 | 3000 | 39.18 | 3.27% | ✅ 완벽 |
| Stage C1 | base_mass ±0.5kg, 외력 없음 | 3000 | 36.95 | 3.84% | ✅ 완벽 |
| Stage C2 | base_mass ±0.5kg, 외력 ±3N | 3000 | 12.70 | 46.53% | ❌ 실패 |
| Stage C3 | base_mass ±1.0kg, 외력 ±5N | 4000 | 7.46 | 62.00% | ❌ 실패 |
| Stage C4 | base_mass ±1.5kg, 외력 ±10N | 4000 | -0.40 | 99.97% | ❌ 완전 붕괴 |

**실패 원인 분석:**  
C2에서 orientation 종료율이 학습 내내 0.002 → 0.52로 꾸준히 악화됨.  
외력을 한 번도 본 적 없는 정책에 ±3N을 한 번에 주입한 것이 원인.  
3000 iteration으로는 회복 불가능했고, 이후 C3/C4는 실패한 C2에서 출발해 연쇄 붕괴.

### 13-2. 대응 전략: Option A (세밀한 외력 단계별 도입)

4/17부터 C1 best.pt (orientation 3.84%)를 베이스로, 외력을 훨씬 세밀하게 올리는 방향으로 재학습 시작.

| 스테이지 | 외력 | base_mass | 반복 | 상태 |
|---------|------|----------|------|------|
| D1 | ±1N | ±0.5kg | 3000 | 🔄 학습 중 |
| D2 | ±2N | ±0.5kg | 3000 | ⏳ 대기 |
| D3 | ±3N | ±0.5kg | 4000 | ⏳ 대기 (C2 실패 수준 재도전) |
| D4 | ±5N | ±1.0kg | 4000 | ⏳ 대기 |
| D5 | ±10N | ±1.5kg | 5000 | ⏳ 대기 |

**PPO 전략 변경 핵심:**
- LR 절대 올리지 않음 (5e-5 고정) — catastrophic forgetting 방지 최우선
- D1~D2: epochs=2 (C1과 동일, 보수적)
- D3~D4: epochs=3 (소폭 증가)
- D5: epochs=4 (최종 강화)

**예상 소요시간:** 약 20~22시간 (D1부터 D5까지 전부)

### 13-3. 체크포인트 현황

```
δ3/checkpoints/
  stage_b_hylion_v6/best.pt      → model_11998.pt (Stage B 원본)
  stage_bplus_hylion_v6/best.pt  → Stage B+ ✅ (3.27%)
  stage_c1_hylion_v6/best.pt     → Stage C1 ✅ (3.84%) ← D1 출발점
  stage_c2_hylion_v6/best.pt     → Stage C2 ❌ (46.53%, 사용 안 함)
  stage_c3_hylion_v6/best.pt     → Stage C3 ❌ (62%, 사용 안 함)
  stage_c4_hylion_v6/best.pt     → Stage C4 ❌ (99.97%, 사용 안 함)
  stage_d1_hylion_v6/best.pt     → 학습 완료 후 생성 예정
  ...
```

### 13-4. 학습 모니터링

```bash
# 현재 학습 상태 확인
bash /home/laba/project_singularity/δ3/scripts/monitor_progressive.sh

# Option A 로그 실시간 확인
tail -f /tmp/hylion_v6_stageD1.log | grep -E "Mean reward|base_orientation|Iteration"

# 오케스트레이터 로그
tail -f /tmp/hylion_optionA_orchestrator.log
```

---

*작성: 2026-04-16 | 업데이트: 2026-04-17 (강건성 학습 경과 및 Option A 재학습 현황 추가)*
