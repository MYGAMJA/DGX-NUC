# Sim-to-Sim 진행 현황 (MuJoCo 검증)

> 최초 작성: 2026-04-25 | 최근 갱신: 2026-04-27  
> 목표: IsaacLab(PhysX) 학습 체크포인트를 MuJoCo에서 sim-to-sim 검증

---

## 환경

| 항목 | 내용 |
|------|------|
| 실행 머신 | NUC (SSH from VS Code) |
| Python | 3.10.12 (system) |
| mujoco | 3.6.0 |
| torch | 2.11.0+cpu |
| DISPLAY | :0 (NUC에 모니터 직접 연결) |

실행 명령 (기준):
```bash
DISPLAY=:0 python3 /home/laba/Hylion/sim/mujoco/play_mujoco.py \
  --ckpt /home/laba/Hylion/checkpoints/biped/stage_d5_hylion_v6/best.pt \
  --urdf /home/laba/Hylion/sim/isaaclab/robot/hylion_v6.urdf \
  --vx 0.0 --duration 10.0

# 가설 B 검증: effort limit 20Nm
DISPLAY=:0 python3 ... --zero-action --effort-limit 20 --diag 5 --duration 10.0

# 가설 B + armature 동시 적용
DISPLAY=:0 python3 ... --zero-action --effort-limit 20 --armature --diag 5 --duration 10.0

# policy + 20Nm (가설 B가 원인이면 이게 버텨야 함)
DISPLAY=:0 python3 ... --effort-limit 20 --armature --diag 5 --vx 0.0 --duration 10.0
```

---

## 체크포인트

경로: `/home/laba/Hylion/checkpoints/biped/`

```
stage_d1_hylion_v6/    ← 검증 대상
stage_d1_5_hylion_v6/  ← 검증 대상
stage_d2_hylion_v6/    ← 검증 대상
stage_d2_5_hylion_v6/  ← 검증 대상
stage_d3_hylion_v6/    ← 검증 대상
stage_d4_hylion_v6/    ← 검증 대상
stage_d5_hylion_v6/    ← 검증 대상 (최신)
# stage_a/b/bplus/c* 는 망한 테스트라 제외
```

체크포인트 구조:
```python
{
  'actor_state_dict': {'mlp.0.weight': [256,45], ..., 'mlp.6.weight': [12,128]},
  'infos': None   # obs normalization 없음
}
# mlp.X.* → net.X.* 변환 후 로드 (play_mujoco.py load_policy() 내부 처리)
```

학습 config (`robot_cfg_BG.py`):
- kp=20, kd=2, effort_limit=6 Nm (legs + ankles 동일)
- armature=0.007 (legs), armature=0.002 (ankles)
- obs_dim=45, act_dim=12
- decimation=8, sim_dt=1/200Hz → control_Hz=25
- action_scale=0.25 (`JointPositionActionCfg`)

---

## 로봇 물리 특성 (측정값)

```
총 질량:  19.89 kg
중력:     195 N
COM 위치: x=+0.0076m (살짝 앞), z=0.473m (지면 위)
base body:  mass=10.74 kg, COM z=0.627m  (frame origin은 z≈-0.045m으로 발 아래이지만
                                            inertial origin이 0.627m에 있음 — 정상)
hip_roll:   z≈0.498m (정상 서 있을 때)
ankle_roll: z≈0.054m
```

URDF "base" 링크 특이사항:
- **frame origin이 발 아래 (z≈-0.045m)**에 있음 — IsaacLab USD와 다른 좌표 기준
- IsaacLab: `init_state.pos=(0,0,0.78)` → USD 루트 바디를 0.78m에 배치
- URDF freejoint qpos[2]=-0.045m일 때 발이 지면(z≈0)에 닿음
- COM은 정상적으로 0.47m에 있어 물리적으로 안정된 구조

---

## play_mujoco.py 누적 수정 내역

### 세션 1에서 해결한 것들

| 번호 | 문제 | 해결 |
|------|------|------|
| 1 | STL 메시 경로 오류 | assets dict + basename 키로 로드 |
| 2 | checkpoint 키 미스매치 (`actor.*` vs `mlp.*`) | `load_policy()`에서 변환 |
| 3 | freejoint 없음 | `base_body.add_freejoint()` |
| 4 | 지면 없음 | `mjGEOM_PLANE` 추가 |
| 5 | base_z 하드코딩 오류 | FK로 자동 계산 |
| 6 | quaternion 순서 오류 | `(w,x,y,z)` 직접 사용 |
| 7 | angular velocity world→body 변환 누락 | `R.T @ omega_world` |
| 8 | fall detection `base_z<0.3` 즉시 트리거 | `hip_z<0.15` 로 변경 |
| 9 | kp=100 (5배 과도) | kp=20 (training config 기준) |
| 10 | position servo → motor actuator | 수동 PD 계산 |

### 세션 2에서 해결한 것들

| 번호 | 문제 | 해결 | 효과 |
|------|------|------|------|
| 11 | STL mesh + URDF primitive collision 이중 활성 | mesh geom만 비활성화 (`_disable_mesh_geoms`) | - |
| 12 | collision primitive 중복 추가 (URDF에 이미 있음) | 커스텀 추가 코드 제거 | - |
| 13 | PD를 25Hz에서만 계산 (IsaacLab: 200Hz) | PD 계산을 substep 루프 안으로 이동 | 23→35 steps 개선 |
| 14 | self-collision 활성 (IsaacLab: 비활성) | `contype/conaffinity` 분리 설정 | 효과 미미 |
| 15 | foot_z가 mesh geom 기준 → 잘못된 초기 높이 | active collision geom 기준 + half_z 보정 | 초기 위치 정확화 |

### 세션 3에서 추가한 것들 (코드 변경만, 결과 미수집)

| 번호 | 변경 | 내용 |
|------|------|------|
| 16 | `--effort-limit N` CLI arg | 기본 6Nm, 가설 B 테스트용 20Nm 지원 |
| 17 | `--kp / --kd` CLI arg | PD 게인 런타임 오버라이드 |
| 18 | `--armature` 플래그 | `dof_armature` 적용: hip/knee=0.007, ankle=0.002 |
| 19 | `--zero-action` 플래그 | policy 비활성화, 순수 PD 기준선 측정 |
| 20 | `--diag N` 플래그 | N 스텝마다 grav/angvel/qpos/action/torque/포화 출력 |
| 21 | MuJoCo solver params | `iterations=50, noslip_iter=4` (PhysX 설정 대응) |

---

## URDF collision 구조 (중요)

URDF(`hylion_v6.urdf`)에는 이미 collision primitive가 정의되어 있음:

| body | collision geom |
|------|----------------|
| base | box (0.15×0.14×0.23m), z=+0.71m offset |
| leg_*_hip_pitch | cylinder (r=0.05, h=0.13m) |
| leg_*_knee_pitch | cylinder (r=0.04, h=0.15m) |
| leg_*_ankle_roll | box (0.072×0.22×0.04m), rpy=[π/2,0,0] |
| soarm_*_shoulder_link | box (0.08×0.05×0.15m) |

MuJoCo URDF 로더는 `<visual>`과 `<collision>` 모두 collision 활성화.  
→ mesh visual geom만 `contype=0, conaffinity=0`으로 비활성화해야 함.  
→ 커스텀 primitive 별도 추가 **불필요** (오히려 이중 충돌 발생).

---

## 현재 결과 (세션 2 기준)

| 설정 | d5, vx=0.0 생존 |
|------|----------------|
| 이전 (kp=100, 25Hz PD, URDF+custom중복) | 18 steps / 0.72s |
| 중간 (kp=20, 25Hz PD, URDF+custom중복) | 45 steps / 1.8s |
| 현재 (kp=20, 200Hz PD, URDF only, no self-col) | **35 steps / 1.4s** |

---

## 핵심 미해결 문제 (가장 중요)

### 발견: policy가 zero-action과 동일한 결과

```python
# Zero action (PD가 default 자세만 유지)으로 테스트 → 35 steps / 1.4s
# policy 사용 시에도 → 35 steps / 1.4s
# → policy가 안정화 기여를 전혀 못 하고 있음
```

### 낙상 패턴 (진단 데이터)

```
step=  0  pitch=-0.3°  vx=+0.166  ← 초기 접촉 충격으로 이미 움직임
step=  5  pitch=-2.9°  vx=+0.033
step= 10  pitch=-0.7°  vx=-0.080
step= 15  pitch=+4.8°  vx=-0.178  ← 뒤로 기울기 시작
step= 20  pitch=+15.5° vx=-0.327  ← 가속
step= 25  pitch=+35.4° vx=-0.344
step= 30  pitch=+71.9°           ← 완전히 뒤로 넘어짐
step= 35  FALL (hip_z=0.146)
```

낙상 시 joint 상태: ankle_pitch, hip_pitch 모두 **±6Nm 포화** → effort limit 한계 도달

### 원인 가설 (두 가지 중 하나)

**가설 A: Obs mismatch** (더 가능성 높음)
- IsaacLab USD의 root body와 URDF "base" body가 다른 물리적 위치를 기준으로 함
- IsaacLab: root body at 0.78m (아마도 pelvis 위치)
- URDF MuJoCo: root body at -0.045m (발 아래 참조점)
- 관측값 계산 시 body frame 방향이 다를 가능성 (특히 angular velocity, projected_gravity)
- 관절 각도 부호 규칙 불일치 (일부 joint에 `rpy=[0,0,-π]` 등 180° 플립 있음)

**가설 B: 6Nm effort limit이 MuJoCo 동역학에 불충분** ← 정량적으로 유력
- 19.89kg 로봇, COM z=0.473m: ankle에서 필요한 복원 토크 ≈ 91.7 × sin(θ) Nm
- 5° 피칭만 해도 ≈ 8 Nm 필요 → 6Nm 한도 초과
- PhysX ImplicitActuator의 effort_limit은 soft clip이라 순간적으로 초과 가능
  → MuJoCo hard clip과 다름 (같은 6Nm이라도 실제 적용 토크 다름)
- 낙상 시 모든 ankle/hip_pitch가 포화 상태 → 직접적 증거

**가설 A URDF joint sign 정적 분석 결과:**
- 모든 다리 joint `<axis xyz="0 0 1">` (로컬 z축 회전)
- rpy=[0,0,-π]는 z축을 회전시키지 않음 → 실제 회전축은 변하지 않음
- 즉 left knee/ankle_pitch의 rpy=-π는 x/y frame을 뒤집지만 joint 회전 방향은 유지
- **결론: 정적 분석으로는 joint sign 반전 없음** (하지만 USD→URDF 변환 과정의
  암묵적 부호 차이는 실제 실행 없이 확인 불가)

---

## 다음 할 일 (우선순위 순)

### 1순위: effort limit 20Nm 테스트 ← 지금 바로 실행 가능

```bash
# 1a. zero-action + 20Nm: 6Nm 포화가 원인인지 확인
DISPLAY=:0 python3 sim/mujoco/play_mujoco.py \
  --zero-action --effort-limit 20 --armature --diag 5 --duration 10.0

# 결과 해석:
#   10초 버팀 → 가설 B 확정 (6Nm 포화가 문제)
#   여전히 빨리 쓰러짐 → obs mismatch (가설 A)

# 1b. policy + 20Nm
DISPLAY=:0 python3 sim/mujoco/play_mujoco.py \
  --effort-limit 20 --armature --diag 5 --vx 0.0 --duration 10.0
```

### 2순위: 가설 B 확정 시 → 적절한 effort_limit 찾기

```bash
# 이론적으로 필요한 ankle torque: 19.89kg * 9.81 * 0.47m = 91.7 Nm (최대, 90° 기울기)
# 실제 보행 중 max 필요 토크 → IsaacLab 학습 로그에서 joint_torques_l2 확인
# 실용 테스트: 40Nm, 60Nm 순서로 시도
```

### 3순위: obs/action 규칙 검증 (가설 A 검증용, --diag 활용)

```bash
# zero-action 상태에서 --diag 1로 step 0~5 obs 확인
# 특히 qpos_rel이 0에 가까운지 (default 자세에서 시작이므로 ~0이어야 함)
# ang_vel이 초기에 0에 가까운지
# grav_body가 [0, 0, -1]에 가까운지
DISPLAY=:0 python3 sim/mujoco/play_mujoco.py \
  --zero-action --effort-limit 20 --diag 1 --duration 0.5
```

### 4순위: IsaacLab play 스크립트로 USD에서 obs 비교

USD에서 obs를 직접 뽑아 우리 obs와 비교 (IsaacLab 환경 있을 때):
```bash
python sim/isaaclab/scripts/play_hylion_v6_BG.py \
  --ckpt_path .../stage_d5_hylion_v6/best.pt --lin_vel_x 0.0
# 첫 5 step의 obs[3:21] (ang_vel + gravity + qpos_rel)을 캡처해 비교
```

---

## 현재 play_mujoco.py 상태 요약

```python
# CLI 파라미터 (training defaults)
# --kp 20 --kd 2 --effort-limit 6  ← 학습 config 기준
# --armature                         ← dof_armature 적용 (hip/knee=0.007, ankle=0.002)
# --zero-action                      ← policy 없이 PD만 (기준선)
# --diag N                           ← N 스텝마다 상세 출력
ACTION_SCALE = 0.25
SIM_DT = 1/200, N_SUBSTEPS = 8, CONTROL_HZ = 25

# build_mujoco_model(urdf_path, effort_limit):
# 1. URDF 로드 (STL assets basename 키)
# 2. ground plane 추가
# 3. base_body.add_freejoint()
# 4. mesh geom collision 비활성화 (URDF 기존 box/cylinder는 유지)
# 5. self-collision 비활성화 (robot: contype=1/conaffinity=0, ground: contype=0/conaffinity=1)
# 6. 12개 leg joint에 motor actuator 추가 (forcerange=±effort_limit)
# 7. solver: iterations=50, noslip_iter=4 (PhysX 대응)

# run():
# - armature: --armature 시 dof_armature 적용
# - 초기 base_z: active collision geom 중 가장 낮은 geom bottom이 z=+5mm가 되도록 자동 계산
# - PD 계산: substep 루프 안에서 200Hz로 계산 (IsaacLab ImplicitActuator 매칭)
# - obs: [cmd(3), ang_vel_body(3), proj_grav(3), joint_pos_rel(12), joint_vel(12), last_action(12)]
# - fall: hip_z < 0.15
# - diag: torque 포화 관절 이름 출력 포함
```

---

## 기타

### Segfault
viewer 종료 시 exit code 139 (MuJoCo 3.6.0 버그)  
→ 결과 출력 이후에 발생, 수치 수집에는 영향 없음

### armature (미적용)
학습 config에 armature=0.007/0.002 있음, MuJoCo에는 미적용  
적용하려면: `model.dof_armature[dofadr] = 0.007` (컴파일 후)  
효과가 있는지는 미확인
