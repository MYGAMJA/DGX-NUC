# Hylion MuJoCo Sim-to-Sim

> 갱신: 2026-04-30  
> 권위 있는 진행 기록은 [`docs/jimmy/sim2sim_progress.md`](../../docs/jimmy/sim2sim_progress.md)

---

## 현재 상태 (한 줄 요약)

**MuJoCo sim2sim 작동함.** 무외란 평지 보행 250/250 스텝 통과. 적용 체크포인트: stage_bplus ~ stage_e4 모두.

근본 원인 발견 및 수정: hylion_v6.xml / hylion_v7.xml 에 `<compiler angle="radian" autolimits="true"/>` 누락 → 무릎 joint range가 도(°)로 해석되어 약 0.043 rad 한계로 잠김 → 보행 불가. 4-29에 추가하여 해결.

---

## 빠른 실행

```bash
# 기본 (E4 + hylion_v7.xml + 평지 vx=0.3, headless)
python3 sim/mujoco/play_mujoco.py --no-viewer --vx 0.3 --duration 10.0

# 다른 체크포인트로
python3 sim/mujoco/play_mujoco.py \
  --ckpt checkpoints/biped/stage_d4_hylion_v6/best.pt \
  --no-viewer --vx 0.3 --duration 10.0

# GUI 뷰어 포함 (NUC 모니터)
DISPLAY=:0 python3 sim/mujoco/play_mujoco.py --vx 0.3 --duration 15.0
```

`run_sim2sim.sh` 래퍼:
```bash
./sim/mujoco/run_sim2sim.sh walk        # GUI, vx=0.3
./sim/mujoco/run_sim2sim.sh headless    # SSH용
./sim/mujoco/run_sim2sim.sh baseline    # zero-action 기준선
./sim/mujoco/run_sim2sim.sh diag        # --diag 5
```

---

## 새 체크포인트 도착 시 검증

```bash
# 1. NaN 체크 (30초)
python3 -c "
import torch, glob
for p in sorted(glob.glob('checkpoints/biped/stage_*/best.pt')):
    c = torch.load(p, map_location='cpu', weights_only=False)
    w = c.get('actor_state_dict', {}).get('mlp.0.weight')
    if w is None: continue
    print(f'{p}  iter={c.get(\"iter\",\"?\")}  NaN={w.isnan().sum().item()}')
"

# 2. 평지 서있기 (vx=0)
python3 sim/mujoco/play_mujoco.py --ckpt $CKPT --no-viewer --vx 0.0 --duration 10.0

# 3. 평지 보행 (vx=0.3)
python3 sim/mujoco/play_mujoco.py --ckpt $CKPT --no-viewer --vx 0.3 --duration 10.0

# 4. 빠른 보행 (vx=0.5)  — 최종 스테이지(D5+)에서만
python3 sim/mujoco/play_mujoco.py --ckpt $CKPT --no-viewer --vx 0.5 --duration 10.0
```

판정 (`docs/jimmy/sim2sim_progress.md` 기준):
| 결과 | 판정 |
|------|------|
| vx=0.0, 250/250 step | ✅ 서있기 OK |
| vx=0.3, > 200 step | ✅ 보행 OK |
| action이 항상 0 | ❌ NaN 체크포인트 |
| 30 step 이내 fall | ❌ 다른 원인 (XML/joint range 등 의심) |

---

## 학습 설정 동기화 (변경 금지)

```python
# sim/isaaclab/hylion/robot_cfg_BG.py 와 일치해야 함
KP = 20.0
KD = 2.0
EFFORT_LIMIT = 6.0  # Nm  ← MuJoCo는 hard clip
ACTION_SCALE = 0.25
SIM_DT = 1/200      # physics 200 Hz
N_SUBSTEPS = 8      # → control 25 Hz
ARMATURE_legs = 0.007
ARMATURE_ankles = 0.002

# default joint pos
hip_pitch  = -0.2 rad
knee_pitch =  0.4 rad
ankle_pitch = -0.3 rad
others = 0
```

> **중요**: MJCF 새로 작성 시 반드시 `<compiler angle="radian" autolimits="true"/>` 포함. 누락 시 모든 joint range가 도 단위로 해석됨.

---

## MJCF 파일

| 파일 | 용도 | 비고 |
|------|------|------|
| `sim/isaaclab/robot/hylion_v6.xml` | 주력 검증용 | compiler radian 적용. 250/250 통과. |
| `sim/isaaclab/robot/hylion_v7.xml` | BHL biped collision geometry 호환 | compiler radian 적용. 75/75 통과. |
| `sim/isaaclab/robot/hylion_v6.urdf` | 레거시 | URDF 경로는 사용하지 말 것. MJCF 사용. |

총 질량 19.89 kg (base 4.83 + SO-ARM × 2 = 6.88 + biped 다리 + 기타). v7은 SO-ARM 정확한 위치까지 반영.

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| 무릎이 즉시 잠겨 보행 안 됨 | `<compiler angle="radian">` 누락 | MJCF 최상단에 추가 |
| action 전 step 0.0 | NaN 체크포인트 | NaN 체크 먼저 |
| viewer exit 139 | MuJoCo 3.6.0 segfault | 결과 출력 후 발생, 무시 |
| 발이 회전된 상태로 박힘 | foot geom quat 누락 | quat="0.707107 0.707107 0 0" 확인 |
| base_z 어긋남 | base inertial 자동계산 | `<inertial>` 명시 |

---

## 관련 문서

- [`docs/jimmy/sim2sim_progress.md`](../../docs/jimmy/sim2sim_progress.md) — 모든 작업 기록과 버그 수정 이력
- [`docs/ETHAN/active/`](../../docs/ETHAN/active/) — 학습 계획 및 plan 문서
- [`docs/ETHAN/archive/`](../../docs/ETHAN/archive/) — outdated 문서 보관

---

## CLI 옵션

```
--ckpt PATH          체크포인트 (default: stage_e4_hylion_v6/best.pt)
--mjcf PATH          MJCF (default: hylion_v7.xml)
--vx / --vy / --wz   속도 명령 (default: 0)
--duration N         시뮬레이션 시간 초 (default: 10)
--kp / --kd          PD 게인 (default 20 / 2 — 변경 금지)
--effort-limit N     토크 한도 (default 6 — 학습값과 일치 유지)
--zero-action        policy 끄고 PD만 (기준선)
--diag N             N step마다 obs/action/torque 출력
--no-viewer          headless (SSH 환경 필수)
--device cpu/cuda    torch device
```
