# Hylion Sim2Sim (IsaacLab → MuJoCo) 작업 기록

## 목표
IsaacLab(PhysX ImplicitActuator)로 학습된 Hylion biped policy를  
MuJoCo에서 sim-to-sim으로 검증한다.

---

## 파일 구조

| 파일 | 역할 |
|------|------|
| `sim/mujoco/play_mujoco.py` | sim2sim 실행 스크립트 |
| `sim/isaaclab/robot/hylion_v6.xml` | Hylion MuJoCo MJCF (작업 중) |
| `sim/isaaclab/hylion/robot_cfg.py` | 학습 로봇 설정 (kp=20, kd=2, effort=6 Nm) |
| `sim/isaaclab/hylion/env_cfg.py` | 학습 환경 설정 (obs 45-dim, action 12-dim) |
| `checkpoints/biped/stage_*/best.pt` | RSL-RL 체크포인트 |

---

## 학습 설정 요약

```
obs 45-dim: [cmd(3), ang_vel(3), proj_gravity(3), joint_pos_rel(12), joint_vel(12), last_action(12)]
action 12-dim: 다리 joint (left/right: hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll)
kp=20, kd=2, effort_limit=6 Nm, armature: hip/knee=0.007 / ankle=0.002
sim_dt=1/200 Hz, decimation=8 → control 25 Hz
default joint pos: hip_pitch=-0.2, knee=+0.4, ankle_pitch=-0.3, others=0
```

---

## 발견된 버그 및 수정 이력

### 1. BHL timestep 불일치 (해결)
- **문제**: BHL biped MJCF timestep=0.002s, 학습=0.005s
  → BHL은 500 Hz physics로 실행, 체감 8 substep = 0.016s (학습 0.04s와 다름)
- **수정**: `play_mujoco.py`에 `model.opt.timestep = SIM_DT(0.005)` 강제 설정

### 2. base `<inertial>` 누락 (해결)
- **문제**: base body에 inertial 미정의 → MuJoCo가 base_col box에서 자동계산  
  → body_ipos = [0, 0, 0.71] (틀린 위치)
- **수정**: `hylion_v6.xml`에 BHL 기준 CoM 명시
  ```xml
  <inertial pos="-0.00462 0.000372 0.6379" mass="4.83"
            fullinertia="0.0569 0.0432 0.0270 -6.2e-6 -0.00110 -2.3e-5"/>
  ```
  - mass=4.83 kg (BHL 3.16 kg vs Hylion 학습 4.83 kg)

### 3. 오른발 geom z 부호 오류 (해결)
- **문제**: rfoot_col `pos="0.02 0.023 -0.02"` (z=-0.02, 비대칭)
- **수정**: z=+0.02로 수정

### 4. 발 geom quat 누락 (해결 - 핵심!)
- **문제**: BHL biped foot geom에는 `quat="0.707107 0.707107 0 0"` (90° Rx)이 있어  
  box의 긴 축(0.11m)이 world-z가 아닌 world-y로 배치됨.  
  hylion_v6.xml에 이 quat을 복사할 때 빠뜨림  
  → world_half_z = 0.1114m (발이 옆으로 선 상태), base_z = +0.046m (8cm 너무 높음)
- **수정**: 양발 geom에 quat 추가
  ```xml
  <geom name="lfoot_col" type="box" size="0.036 0.11 0.02"
        pos="0.0 0.03 0.006" quat="0.707107 0.707107 0 0" material="ankle"/>
  ```
  결과: world_half_z = 0.031m, base_z = -0.034m (BHL biped와 동일)

### 6. `<compiler angle="radian">` 누락 (해결 — 근본 원인!)
- **문제**: hylion_v6/v7 MJCF에 `<compiler>` 태그 없음 → MuJoCo 기본값(degree)으로 joint range 해석
  - `range="0 2.443"` → 0°~2.44° = 0~0.043 rad (본래 의도: 0~2.44 rad = 0~140°)
  - DEFAULT_JOINT_POS(knee=0.4 rad)이 joint limit(0.043 rad) 밖 → constraint force → 관절 고정
- **증상**: 로봇이 stand posture에 freeze, 걷지 않음
- **수정**: 양 MJCF 최상단에 추가
  ```xml
  <compiler angle="radian" autolimits="true"/>
  ```
- **결과**: 보행 동작 즉시 복원 (250/250 스텝 생존)

### 5. SO-ARM 팔 질량 누락 (해결)
- **문제**: Hylion 학습에는 SO-ARM 팔(×2, 총 6.88 kg)이 있지만 MJCF에 없음  
  → `play_mujoco.py`가 6.88 kg을 base body CoM에 lumped mass로 보정 (위치 부정확)
- **수정**: hylion_v6.xml에 static body 2개 추가 (관절 없음, 질량만)
  ```xml
  <body name="soarm_left" pos="0.0 0.12 0.75">
    <inertial pos="0 0 -0.1" mass="3.44" diaginertia="0.025 0.025 0.005"/>
  </body>
  <body name="soarm_right" pos="0.0 -0.12 0.75">
    <inertial pos="0 0 -0.1" mass="3.44" diaginertia="0.025 0.025 0.005"/>
  </body>
  ```
  결과: 총 질량 **19.890 kg** (학습 목표 19.89 kg, 오차 0), mass correction = 0

---

## 현재 상태 (2026-04-29) — 보행 확인 완료

### 달성한 것
- hylion_v6.xml + compiler radian 수정 → 250/250 스텝 생존 + 보행 동작
- hylion_v7.xml (BHL collision geometry + SO-ARM + compiler radian) → 75/75 스텝 생존 + 보행
- 총 질량 정확히 일치 (19.89 kg)
- base_z = -0.034m (BHL biped와 동일)
- stage_bplus ~ stage_e4 모든 체크포인트 보행 확인

### 핵심 발견 (최종)
- **BHL biped MJCF + Hylion policy** → 걸음 (BHL은 `<compiler angle="radian">` 있어서 정상)
- **hylion_v6/v7 + Hylion policy (수정 전)** → 고정 (compiler 없어서 joint range 극소)
- **hylion_v6/v7 + Hylion policy (수정 후)** → 걸음

---

## hylion_v7.xml 생성 (완료)

`sim/isaaclab/robot/hylion_v7.xml`:
- BHL biped와 동일한 kinematic chain + collision geometry (hip/knee cylinder)
- base inertial: mass=4.83 kg (학습 값), CoM 동일
- SO-ARM 팔 body 추가 (mass=3.44 × 2, y=±0.12, z=0.75)
- mesh 절대경로 사용 (어디서나 로드 가능)
- 총 질량 19.89 kg → play_mujoco.py mass correction = 0

`play_mujoco.py` 기본값 업데이트:
- DEFAULT_MJCF → hylion_v7.xml
- DEFAULT_CKPT → stage_e4_hylion_v6/best.pt

---

---

## 근본 원인 발견 (2026-04-29) — `<compiler angle="radian">` 누락

### 진짜 원인 (해결!)
`hylion_v6.xml` / `hylion_v7.xml` 모두 `<compiler>` 태그 없음  
→ MuJoCo 기본값 = **각도 단위: degree**  
→ XML의 joint range (라디안으로 작성됨) 가 degree로 해석됨  
→ range "0 2.443" → 0° ~ 2.443° = 0 ~ 0.0426 rad  
→ DEFAULT_JOINT_POS (knee=0.4 rad) 가 joint limit 밖 → constraint force 발생 → 관절 고정!

```
BHL biped (올바름): range 0 ~ 2.443 rad = 0 ~ 140° (올바른 무릎 범위)
hylion_v7 (수정 전): range 0 ~ 2.443 deg = 0 ~ 0.043 rad (극히 작은 범위)
```

### 수정
```xml
<!-- hylion_v6.xml, hylion_v7.xml 모두 추가 -->
<compiler angle="radian" autolimits="true"/>
```

### 결과
- hylion_v7.xml + stage_e4 → 75/75 스텝 생존 + 보행 동작 확인
- hylion_v6.xml + stage_e4 → 250/250 스텝 생존 + 보행 동작 확인
- 총 질량 19.89 kg, SO-ARM y=±0.12 m 올바른 위치, base 4.83 kg

### 이전 오진 (허위 근거)
- ~~Sim2Sim Gap (PhysX vs MuJoCo)~~: 진짜 원인이 아니었음. 관절 범위가 잘못 해석된 것이 원인.
- ~~inertia/mass ratio 차이~~: BHL이 걷고 hylion이 안 걷던 이유도 모두 이 버그 때문이었음.

---

## 개선 방향

1. **IsaacLab 시각화 검증**: MuJoCo 대신 IsaacLab play_script로 보행 확인 (참고용)
2. **더 나은 체크포인트**: stage_e4_hylion_v6 이후 추가 훈련

---

## play_mujoco.py 주요 파라미터

```bash
python3 sim/mujoco/play_mujoco.py \
  --ckpt checkpoints/biped/stage_e4_hylion_v6/best.pt \
  --mjcf sim/isaaclab/robot/hylion_v6.xml \
  --vx 0.3 --no-viewer --diag 25
```

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--effort-limit` | 6 | 토크 한도 Nm |
| `--kp` / `--kd` | 20 / 2 | PD 게인 |
| `--zero-action` | off | policy 무시, default 자세 유지 |
| `--no-viewer` | off | headless 실행 |
| `--diag N` | 0 | N 스텝마다 상태 출력 |
