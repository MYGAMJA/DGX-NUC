# Hylion v6 Biped — MuJoCo Sim-to-Sim 검증

> 작성: 2026-04-27  
> 목적: IsaacLab(PhysX)에서 학습한 biped 정책을 MuJoCo에서 교차 검증하는 전체 프로세스 정리

---

## 1. 프로젝트 맥락

| 항목 | 내용 |
|------|------|
| 로봇 | Hylion v6 (biped, 12-DOF 다리, 총 질량 19.89 kg) |
| 학습 환경 | IsaacLab (PhysX 백엔드), Berkeley Humanoid Lite 기반 |
| 검증 환경 | MuJoCo 3.6.0 (MJCF 기반, `play_mujoco.py`) |
| 실행 머신 | NUC (SSH headless 또는 DISPLAY=:0 GUI) |
| 발표 마감 | 2026-06-01 |
| 현재 유효 최고 체크포인트 | `stage_d2_5_hylion_v6/best.pt` |

---

## 2. 체크포인트 현황 (2026-04-27 기준)

```
checkpoints/biped/
  stage_bplus_hylion_v6/best.pt    ✅ 정상
  stage_c1_hylion_v6/best.pt       ✅ 정상
  stage_c2_hylion_v6/best.pt       ✅ 정상
  stage_c3_hylion_v6/best.pt       ✅ 정상
  stage_c4_hylion_v6/best.pt       ✅ 정상
  stage_d1_hylion_v6/best.pt       ✅ 정상   w0_mean=0.2617
  stage_d1_5_hylion_v6/best.pt     ✅ 정상   w0_mean=0.2649
  stage_d2_hylion_v6/best.pt       ✅ 정상   w0_mean=0.2697
  stage_d2_5_hylion_v6/best.pt     ✅ 정상   w0_mean=0.2709  ← 현재 최선
  stage_d3_hylion_v6/best.pt       ❌ NaN 오염 (D3 학습 폭발)
  stage_d4_hylion_v6/best.pt       ❌ NaN (D3 복사본)
  stage_d5_hylion_v6/best.pt       ❌ NaN (D3 복사본)
```

### NaN 체크 방법 (30초)

```python
import torch, glob

for path in sorted(glob.glob("checkpoints/biped/stage_*/best.pt")):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    w = ckpt.get("actor_state_dict", {}).get("mlp.0.weight")
    if w is None:
        print(f"{path}: actor_state_dict 없음")
        continue
    nan_n = w.isnan().sum().item()
    print(f"{path}  iter={ckpt.get('iter','?')}  NaN={nan_n}/{w.numel()}")
```

> NaN인 체크포인트는 `nan_to_num`으로 0이 되어 policy 출력이 항상 0. 검증 의미 없음.

---

## 3. 로봇 물리 특성

| 항목 | 값 |
|------|-----|
| 총 질량 | 19.89 kg |
| COM 높이 | z ≈ 0.473 m |
| COM 수평 오프셋 | x ≈ +0.0076 m (살짝 앞) |
| base body 질량 | 10.74 kg |
| hip_roll 높이 (직립) | z ≈ 0.498 m |
| ankle_roll 높이 (직립) | z ≈ 0.054 m |

**URDF 좌표 특이사항**: `base` 링크 frame origin이 발 아래 (z ≈ -0.045 m)에 있음.  
IsaacLab USD는 pelvis 위치(z ≈ 0.78 m)를 root로 사용. MJCF는 이 차이를 자동 보정함.

---

## 4. 학습 설정 (IsaacLab 기준)

### 물리 파라미터 (`robot_cfg_BG.py`)

```
kp = 20.0          # joint stiffness (Nm/rad)
kd = 2.0           # joint damping   (Nm·s/rad)
effort_limit = 6.0 # Nm (다리 + 발목 동일)
armature:
  legs (hip/knee)  = 0.007
  ankles           = 0.002
sim_dt = 1/200 Hz
decimation = 8   → control Hz = 25
action_scale = 0.25
```

### 관측 공간 (45-dim)

```
[0:3]   velocity_commands   (vx, vy, wz)
[3:6]   base_ang_vel        (body frame, IMU gyro)
[6:9]   projected_gravity   (body frame, quaternion → R·[0,0,-1])
[9:21]  joint_pos_rel       (12 leg joints, default 오프셋 제거)
[21:33] joint_vel           (12 leg joints)
[33:45] last_action         (12)
```

### 행동 공간 (12-dim)

```
joint 순서:
  leg_left:  hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
  leg_right: hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll

default 자세:
  hip_pitch  = -0.2 rad
  knee_pitch =  0.4 rad
  ankle_pitch= -0.3 rad
  others     =  0.0 rad

target_pos = default_pos + action * 0.25
torque = kp * (target_pos - q) - kd * qdot   (clip ±6 Nm)
```

### 학습 curriculum (외력 강화 단계)

| 스테이지 | 외력 | PPO epochs | iter | 상태 |
|---------|------|-----------|------|------|
| D1 | ±1 N | 2 | 3000 | ✅ 완료 |
| D1.5 | ±1.5 N | 2 | 2000 | ✅ 완료 |
| D2 | ±2 N | 2 | 3000 | ✅ 완료 |
| D2.5 | ±2.5 N | 2 | 2000 | ✅ **현재 최선** |
| D3 | ±3 N | 2 (수정됨) | 4000 | 재학습 중 |
| D4 | ±5 N | 3 | 4000 | 예정 |
| D5 | ±10 N | 4 | 5000 | 예정 |
| E1~E4 | ±15~30 N | 4 | 5000~7000 | 예정 |

---

## 5. 모델 구조 (Actor MLP)

```
Input  → Linear(45, 256) → ELU
       → Linear(256, 128) → ELU
       → Linear(128, 128) → ELU
       → Linear(128, 12)
```

checkpoint 키 변환: `mlp.X.*` → `net.X.*` (`load_policy()` 내부 자동 처리)

---

## 6. MuJoCo 검증 실행

### 파일 위치

```
sim/mujoco/
  play_mujoco.py       ← 메인 검증 스크립트 (MJCF 기반)
  run_sim2sim.sh       ← 모드별 래퍼 스크립트
  MUJOCO_VALIDATION.md ← 원칙 모음
  SIM2SIM_PROGRESS.md  ← 세션별 진행 기록
sim/isaaclab/robot/
  hylion_v6.xml        ← MJCF (검증 기준, armature/sensor 포함)
  hylion_v6.urdf       ← URDF (레거시, 현재 MJCF 권장)
```

### 기본 실행 명령

```bash
# headless (SSH 환경)
python3 sim/mujoco/play_mujoco.py \
  --ckpt checkpoints/biped/stage_d2_5_hylion_v6/best.pt \
  --no-viewer \
  --vx 0.0 --duration 10.0

# GUI 뷰어 포함
DISPLAY=:0 python3 sim/mujoco/play_mujoco.py \
  --ckpt checkpoints/biped/stage_d2_5_hylion_v6/best.pt \
  --vx 0.3 --duration 10.0

# 진단 모드 (5스텝마다 obs/action/torque 출력)
python3 sim/mujoco/play_mujoco.py \
  --ckpt checkpoints/biped/stage_d2_5_hylion_v6/best.pt \
  --no-viewer --diag 5 --vx 0.0 --duration 10.0

# run_sim2sim.sh 래퍼 사용
./sim/mujoco/run_sim2sim.sh walk          # GUI, vx=0.3
./sim/mujoco/run_sim2sim.sh headless      # headless, vx=0.3
./sim/mujoco/run_sim2sim.sh baseline      # zero-action 기준선
./sim/mujoco/run_sim2sim.sh walk_hard     # effort_limit=20 검증
./sim/mujoco/run_sim2sim.sh diag          # 진단 출력
```

### CLI 옵션 전체

```
--ckpt PATH        체크포인트 경로 (default: stage_d5)
--mjcf PATH        MJCF .xml 경로 (default: hylion_v6.xml)  ← 권장
--urdf PATH        URDF 경로 (레거시)
--vx/--vy/--wz     속도 명령 (default: 0.0)
--duration N       시뮬레이션 시간 초 (default: 10.0)
--no-viewer        headless 실행
--diag N           N스텝마다 진단 출력
--zero-action      policy 끄고 PD만 (기준선 측정)
--kp / --kd        PD 게인 (default: 20 / 2, 변경 금지)
--effort-limit N   토크 한도 (default: 6, 실험 시에만 변경)
--armature         dof_armature 적용 (hip/knee=0.007, ankle=0.002)
--device cpu/cuda  torch device
```

---

## 7. 결과 해석 기준

| 결과 | 판정 | 의미 |
|------|------|------|
| vx=0.0, **250/250 steps (10s)** | ✅ 서있기 성공 | 기본 자세 유지 |
| vx=0.0, **< 50 steps** | ❌ 서있기 실패 | 체크포인트 불량 또는 물리 설정 오류 |
| vx=0.3, **> 100 steps (4s+)** | ✅ 걷기 성공 | 동적 보행 가능 |
| vx=0.3, **< 60 steps** | ⚠️ 걷기 미흡 | 중간 스테이지에선 정상 허용 범위 |
| action 전 step에서 0.0 | ❌ NaN 체크포인트 | 섹션 2의 NaN 검사 먼저 |

> D1~D2.5 기대치: 서있기 OK, 걷기 불완전해도 정상  
> D5 이상 기대치: 서있기 + vx=0.3 걷기 모두 성공

### 현재 검증 결과 (세션 3, 2026-04-27 — MJCF 기반)

#### kp 스위프 요약

| kp | kd | effort (Nm) | 체크포인트 | 생존 | 비고 |
|----|----|------------|----------|------|------|
| 20 | 2 | 6 | zero-action | 27 steps / 1.1s | 학습 기준값 |
| 20 | 2 | 6 | d2_5 ✅ | 26 steps / 1.0s | policy ≈ zero-action |
| 20 | 2 | 20 | zero-action | 26 steps / 1.0s | effort 올려도 악화 |
| 50 | 3 | 15 | d2_5 ✅ | 25 steps / 1.0s | |
| 80 | 4 | 25 | d2_5 ✅ | 26 steps / 1.0s | |
| 150 | 8 | 60 | d2_5 ✅ | 24 steps / 1.0s | |
| **200** | **10** | **200** | d2_5 ✅ | **45 steps / 1.8s** | **현재 최선** |
| 200 | 10 | 50 | d2_5 ✅ | 30 steps / 1.2s | effort 줄이면 효과 반감 |
| 200 | 10 | 100 | d2_5 ✅ | 19 steps / 0.8s | 중간값이 오히려 불안정 |
| 250 | 12 | 100 | d2_5 ✅ | 25 steps / 1.0s | |
| 300 | 15 | 150 | d2_5 ✅ | 8 steps / 0.3s | kp 너무 높으면 완전 불안정 |

#### 스테이지별 비교 (kp=20/kd=2/effort=6, vx=0.0 및 vx=0.3)

| 스테이지 | vx=0.0 | vx=0.3 |
|---------|--------|--------|
| d1 ✅ | 25 steps / 1.0s | 25 steps / 1.0s |
| d1_5 ✅ | 25 steps / 1.0s | 25 steps / 1.0s |
| d2 ✅ | 25 steps / 1.0s | 25 steps / 1.0s |
| d2_5 ✅ | 26 steps / 1.0s | 25 steps / 1.0s |

> D1~D2.5 모두 동일한 결과 → curriculum 효과가 MuJoCo에서는 측정 불가  
> 학습 진척도는 IsaacLab play 스크립트로 검증해야 함

---

## 8. 낙상 패턴 및 근본 원인 (2026-04-27 세션 3 검증 완료)

### 낙상 데이터 (d2.5, vx=0.0, kp=20/effort=6)

```
step=  0   pitch= +0.3°  qpos_rel=0  (초기 완벽 자세)
step=  1   qpos_rel: Lhp=+0.105, Lk=-0.226, Lap=+0.168  ← 40ms만에 대규모 deflection
step=  2   qpos_rel: Lhp=+0.147, Lk=-0.317, Lap=+0.228
step=  5   qpos_rel: Lhp=+0.164, Lk=-0.349, Lap=+0.260  ← 이 패턴에서 LOCK
step= 25   pitch=+58°  hip_z=0.219
step= 26   FALL (hip_z=0.071)
```

### policy가 있어도 없어도 동일한 이유

step 5에서의 qpos_rel 비교:
- **zero-action**: Lhp=0.163, Lk=-0.351, Lap=0.256
- **policy+d2_5**: Lhp=0.167, Lk=-0.350, Lap=0.254

두 값이 거의 동일 → **PD 토크가 지면반력에 압도되어 policy 명령이 전달되지 않음.**

### 근본 원인: PhysX ImplicitActuator vs MuJoCo explicit PD

| 항목 | IsaacLab (PhysX) | MuJoCo |
|------|-----------------|--------|
| 방식 | velocity-level constraint (solver 통합) | 외력으로 토크 주입 |
| effort_limit | **soft clip** (순간 초과 가능) | **hard clip** (절대 초과 불가) |
| 지면반력 저항 | constraint solve가 관절 위치 '잠금' | 토크가 지면반력과 경쟁 |
| 학습 때 knee 자세 유지 | implicit constraint가 0.4 rad 유지 | 6 Nm으로는 불가 (0.35 rad 오차 → 7 Nm 필요) |

**결론**: kp=20/±6 Nm는 PhysX에서는 soft limit 덕분에 자세 유지가 되지만,  
MuJoCo hard clip에서는 초기 접촉 충격(40ms, 단 1 control step) 안에  
Lk=-0.35 rad deflection이 발생하고 복구 불가.

### kp 임계값 현상

- kp < 200: 모두 ~25 steps (1.0s) — deflection이 PD를 압도
- kp = 200 / effort = 200: **45 steps (1.8s)** — 임계점, 초기 deflection을 일부 저항
- kp > 200 또는 effort 부분 증가: 오히려 악화 (contact jitter 증가)
- **최적값: kp=200 / kd=10 / effort=200** (학습 파라미터와 10배 차이)

---

## 9. PhysX ↔ MuJoCo 물리 차이 (참고)

| 항목 | IsaacLab (PhysX) | MuJoCo |
|------|-----------------|--------|
| Actuator | implicit joint drive (solver 통합) | explicit PD → motor torque |
| effort_limit | soft clip (순간 초과 가능) | hard clip |
| PD 계산 타이밍 | 200 Hz constraint solve와 동시 | substep 루프 내 200 Hz 수동 계산 |
| self-collision | disabled | contype/conaffinity로 비활성화 |
| armature | `ImplicitActuatorCfg.armature` | `dof_armature` (MJCF default에 포함) |

> MuJoCo에서 서있기는 가능하지만 동적 보행에서 gap이 더 크게 나타남.

---

## 10. MJCF 내부 구조 요점

파일: `sim/isaaclab/robot/hylion_v6.xml`

- `integrator="RK4"` → `play_mujoco.py`에서 `mjINT_IMPLICITFAST`로 교체 (contact 안정성)
- `<joint armature="0.007"/>` (default): 학습 config와 일치
- `<motor ctrlrange="-6 6"/>` (default): effort_limit=6 Nm
- IMU 센서: `imu_gyro` (body-frame angular velocity), `imu_quat` (orientation w,x,y,z)
- 질량 보정: MJCF는 SO-ARM 팔 없이 ~13 kg → base body에 lumped mass 추가 (자동)
- ground: `contype=0 conaffinity=1`, robot geom: `contype=1 conaffinity=0`

---

## 11. 전 스테이지 일괄 검증 스크립트

```bash
for stage in d1 d1_5 d2 d2_5; do
  ckpt="checkpoints/biped/stage_${stage}_hylion_v6/best.pt"
  [ -f "$ckpt" ] || continue

  result=$(python3 sim/mujoco/play_mujoco.py \
    --ckpt "$ckpt" --no-viewer --vx 0.0 --duration 10.0 2>&1 | tail -1)
  echo "[${stage} vx=0.0] $result"

  result=$(python3 sim/mujoco/play_mujoco.py \
    --ckpt "$ckpt" --no-viewer --vx 0.3 --duration 10.0 2>&1 | tail -1)
  echo "[${stage} vx=0.3] $result"
done
```

> d3~d5는 NaN이므로 현재 제외.

---

## 12. 새 체크포인트 도착 시 체크리스트

```
[ ] 1. NaN 검사 (섹션 2) → NaN=0 확인
[ ] 2. vx=0.0, 10s → 250/250 steps 확인 (서있기)
[ ] 3. vx=0.3, 10s → 생존 시간 기록 (걷기)
[ ] 4. --diag 5로 진단 → action이 non-zero인지 확인
[ ] 5. D5 이상이면 vx=0.5도 테스트
[ ] 6. 결과를 이 문서 섹션 7에 추가
```

---

## 13. 알려진 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| action 전 step에서 0.0 | NaN 체크포인트 | 섹션 2 NaN 검사 |
| `--effort-limit 20` 시 더 빨리 쓰러짐 | MuJoCo contact impact에 더 공격적으로 반응 | 학습 기준값(6 Nm) 유지 |
| `AttributeError: MjOption has no attribute 'noslip_iter'` | MuJoCo 버전 차이 | `noslip_iter` → `noslip_iterations` |
| Viewer 종료 시 exit code 139 | MuJoCo 3.6.0 segfault 버그 | 결과 출력 후 발생이므로 무시 |
| step 0에서 knee/ankle 토크 포화 | 초기 접촉 충격 | 바로 사라지면 정상, 계속되면 물리 불안정 |

---

## 14. 검증 결론 및 향후 방향

### MuJoCo 검증 결론 (2026-04-27 세션 3)

```
검증 완료 사항:
  ✅ policy가 non-zero action 출력 (d2_5 체크포인트 정상 작동)
  ✅ NaN 체크포인트 없음 (d1~d2_5 모두 정상)
  ✅ obs 구성 정확 (grav=[0,0,-1], qpos_rel=0, ang_vel=0 at t=0)
  ✅ 초기 낙상 원인 특정: PhysX implicit vs MuJoCo explicit PD 불일치
  ✅ kp 스위프로 임계값 발견 (kp=200/effort=200 → 1.8s 최선)

한계:
  ❌ MuJoCo에서 자립 불가 (최선 1.8s, 학습 환경은 10s+ 유지)
  ❌ curriculum 진척도 (D1→D2.5) 측정 불가 (모두 동일 결과)
  ❌ 걷기 검증 불가 (서있기도 안 됨)
```

### 향후 검증 방향

| 방법 | 목적 | 실행 조건 |
|------|------|---------|
| **IsaacLab play 스크립트** | 학습 환경에서 policy 행동 확인 | DGX 또는 Orin (IsaacLab 설치) |
| MuJoCo kp=200/effort=200 | 새 체크포인트 간 상대 비교 | 지금 바로 가능, 절대 수치는 무의미 |
| 실기 테스트 | 진짜 sim-to-real 검증 | D5/E4 완료 + 하드웨어 준비 후 |

### 향후 체크포인트 도착 시

| 시점 | 체크포인트 | 목적 |
|------|----------|------|
| D3 완료 후 | D3 | kp=200/effort=200으로 D2.5 대비 생존 시간 비교 |
| D5 완료 후 | D5 | IsaacLab play 검증 우선, MuJoCo는 참고용 |
| E4 완료 후 | E4 | IsaacLab play + 실기 테스트 |

> **MuJoCo에서 1.8s 미만이어도 체크포인트 불량이 아님**  
> → IsaacLab에서는 정상 작동 가능. 두 환경의 물리 차이가 본질적으로 큼.

---

*담당: Jimmy | 갱신: 2026-04-27 세션 3 | 기반: SIM2SIM_PROGRESS.md, MUJOCO_VALIDATION.md*
