# Hylion v6 MuJoCo Sim-to-Sim 검증 가이드

> 이 문서는 새 학습 체크포인트가 도착했을 때 **시행착오 없이** MuJoCo 검증을 수행하기 위한 원칙 모음이다.  
> 배경: 2026-04-27 세션에서 쌓인 경험 기반.

---

## 1. 체크포인트 도착 시 제일 먼저 할 것

### 1-1. NaN 검사 (필수, 30초)

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

**판정:**
- `NaN=0` → 정상. 검증 진행.
- `NaN>0` → 학습 미완료 또는 폭발. **MuJoCo 실행해도 policy가 항상 0 출력됨.** 건너뜀.

> **왜 중요한가**: NaN 가중치는 `nan_to_num`으로 0이 되어 policy 출력이 항상 0이 된다.  
> zero-action과 동일한 결과가 나오므로 아무리 실험해도 의미없는 숫자만 얻는다.

---

## 2. 검증 명령어

### 2-1. 기본 실행 (SSH headless)

```bash
python3 sim/mujoco/play_mujoco.py \
  --ckpt checkpoints/biped/stage_XXX_hylion_v6/best.pt \
  --urdf sim/isaaclab/robot/hylion_v6.urdf \
  --no-viewer \
  --vx 0.0 --duration 10.0
```

### 2-2. 전 스테이지 일괄 확인

```bash
for stage in d1 d1_5 d2 d2_5 d3 d4 d5; do
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

### 2-3. 진단 모드 (넘어지는 이유 조사)

```bash
python3 sim/mujoco/play_mujoco.py \
  --ckpt checkpoints/biped/stage_XXX_hylion_v6/best.pt \
  --no-viewer --diag 5 --vx 0.0 --duration 10.0
```

---

## 3. 결과 해석 기준

| 결과 | 판정 | 의미 |
|------|------|------|
| vx=0.0 에서 **250/250 steps (10s)** | ✅ **서있기 성공** | 기본 자세 유지 가능 |
| vx=0.0 에서 **< 50 steps** | ❌ 서있기 실패 | 체크포인트 문제 또는 학습 부족 |
| vx=0.3 에서 **> 100 steps (4s+)** | ✅ 걷기 성공 | 동적 보행 가능 |
| vx=0.3 에서 **< 60 steps** | ⚠️ 걷기 미흡 | 중간 학습 스테이지에선 정상일 수 있음 |
| action이 모두 0.0 | ❌ NaN 체크포인트 | 1-1 검사 먼저 |

> **중간 스테이지(D1~D2) 기대치**: 서있기는 되어야 함. 걷기는 불완전해도 정상.  
> **최종 스테이지(D5+) 기대치**: 서있기 + vx=0.3 걷기 모두 성공해야 함.

---

## 4. 고정 물리 파라미터 (변경 금지)

학습 config(`robot_cfg_BG.py`)와 반드시 일치해야 한다.

```
KP = 20.0          # stiffness (Nm/rad)
KD = 2.0           # damping   (Nm·s/rad)
EFFORT_LIMIT = 6.0 # Nm  ← hard clip
ACTION_SCALE = 0.25
SIM_DT = 1/200     # physics Hz
N_SUBSTEPS = 8     # decimation → control 25Hz
```

> **EFFORT_LIMIT**: IsaacLab PhysX의 `ImplicitActuatorCfg`는 soft limit이고  
> MuJoCo는 hard clip이다. 물리적으로 다르지만 학습 config 값(6Nm)을 기준으로 삼는다.

---

## 5. 알려진 PhysX ↔ MuJoCo 물리 차이 (이해용)

### 5-1. Actuator 차이 (핵심)

| | IsaacLab (PhysX) | MuJoCo (우리 구현) |
|--|--|--|
| 방식 | implicit joint drive (solver 통합) | explicit PD → motor torque |
| effort_limit | soft (순간 초과 가능) | hard clip |
| 안정성 | constraint solve와 동시 계산 | 한 step 지연 |

→ MuJoCo에서 서있기는 되지만 동적 보행에서 gap이 더 크게 나타남.

### 5-2. Root body 기준점 차이

- IsaacLab USD: root body = pelvis (`init_state.pos=(0,0,0.78)`)
- URDF MuJoCo: "base" link origin = 발 아래 (z≈-0.045m)
- `play_mujoco.py`는 FK로 foot 높이를 자동 계산해서 보정함. 별도 조작 불필요.

### 5-3. sim-to-sim gap 범위

2026-04-27 기준 검증 결과:

| 스테이지 | 서있기(vx=0) | 걷기(vx=0.3) |
|----------|-------------|-------------|
| d1~d2_5  | 10s ✅      | ~2s ⚠️      |
| d3~d5    | 학습 미완료  | -           |

---

## 6. 문제별 트러블슈팅

### action이 모든 step에서 0.0

→ **NaN 체크포인트**. 섹션 1-1 검사.

### step 0에서 이미 knee/ankle 토크 포화

→ 정상. 초기 접촉 충격 때문. 바로 사라지면 문제없음. 계속 포화되면 물리 불안정.

### `--effort-limit 20`으로 올렸는데 더 빨리 넘어짐

→ 정상 현상. MuJoCo에서 higher gain은 contact impact에 더 공격적으로 반응해 역효과.  
학습 기준값(6Nm)에서 벗어나지 말 것.

### `AttributeError: MjOption has no attribute 'noslip_iter'`

→ `noslip_iter` → `noslip_iterations`로 수정. (MuJoCo 버전마다 다름)

### Viewer 종료 시 exit code 139 (segfault)

→ MuJoCo 3.6.0 버그. 결과 출력 이후에 발생하므로 수치에 영향 없음. 무시.

---

## 7. 새 체크포인트 검증 체크리스트

```
[ ] 1. NaN 검사 → NaN=0 확인
[ ] 2. vx=0.0, 10s → 250/250 steps 확인
[ ] 3. vx=0.3, 10s → 생존 시간 기록
[ ] 4. --diag 5로 진단 → action이 실제로 non-zero인지 확인
[ ] 5. 최종 스테이지(D5+)라면 vx=0.5도 테스트
```

---

## 8. play_mujoco.py CLI 옵션 요약

```
--ckpt PATH        체크포인트 경로 (default: stage_d5)
--urdf PATH        URDF 경로 (default: hylion_v6.urdf)
--vx / --vy / --wz 속도 명령 (default: 0.0)
--duration N       시뮬레이션 시간 초 (default: 10.0)
--no-viewer        headless 실행 (SSH 환경 필수)
--diag N           N스텝마다 obs/action/torque 출력 (기본 0=off)
--zero-action      policy 끄고 PD만 → 기준선 측정용
--kp / --kd        PD 게인 오버라이드 (기본 20/2, 바꾸지 말 것)
--effort-limit N   토크 한도 (기본 6, 바꾸지 말 것)
--armature         dof_armature 적용 (hip/knee=0.007, ankle=0.002)
--device cpu/cuda  torch device
```
