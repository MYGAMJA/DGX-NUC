# D4 실배포 설계 — 코드 레벨 (2026-05-20)

> 작성: 2026-05-20
> 목적: [34_sim2real_plan](34_sim2real_plan_2026-05-02.md) 의 단계 0~3 을 **코드 레벨**로 구체화
> 정책: `checkpoints/biped/stage_d4_hylion_v6/best.pt` ([33_d4_validation_report](33_d4_validation_report_2026-05-02.md))
> 결정: **BHL lowlevel 스택을 이식**하고 D4 를 ONNX 로 export 한다. 처음부터 작성하지 않는다.

---

## 0. 요약

BHL 본가(`/home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/`)에
완성된 실배포 스택이 있고, **우리 D4 가 그 biped 러너에 사실상 drop-in** 이다.

- BHL `configs/policy_biped_25hz_a.yaml` 이 우리 D4 config 와 **완전히 동일** (obs 45, joint 12, kp20/kd2, effort 6, default pose, action_scale 0.25, history 0)
- 모터: CAN 버스 ×2 (다리당 1개), USB-CAN 어댑터 → SocketCAN `can0`/`can1`. BHL `recoil` 무수정 이식
- 정책: D4 `.pt` → **ONNX export** → BHL `OnnxPolicy` 가 그대로 로드
- 제어 루프: 25 Hz 단일 프로세스 (`run_policy.py` = BHL `run_locomotion.py`)

DGX 측 작업은 ONNX export 1건뿐. 나머지는 전부 NUC 측.

---

## 1. 아키텍처

### 데이터 흐름

```
                        NUC (단일 프로세스, 25 Hz)
  ┌─────────────────────────────────────────────────────────────┐
  │  run_policy.py                                               │
  │    while True:  (RateLimiter 25 Hz)                          │
  │      actions = controller.update(obs)   ── RlController      │
  │      obs     = robot.step(actions)      ── Humanoid          │
  │      rate.sleep()                                            │
  └───────┬─────────────────────────────────────┬───────────────┘
          │                                     │
   ┌──────┴──────┐                       ┌──────┴───────┐
   │ RlController│                       │  Humanoid    │
   │  D4 ONNX    │                       │  HW 인터페이스│
   │  obs45→act12│                       └──┬────────┬──┘
   └─────────────┘                          │        │
                              recoil.Bus ───┤        ├─── SerialImu
                            can0 / can1     │        │   /dev/ttyUSB*
                         (USB-CAN ×2)       │        │   460800 baud
                                     ┌──────┴──┐  ┌──┴──────┐
                                     │ 왼다리   │  │ IMU      │
                                     │ 6 모터   │  │ quat+gyro│
                                     │ daisy    │  └──────────┘
                                     └──────────┘
                                     ┌─────────┐
                            can1 ────┤ 오른다리 │
                                     │ 6 모터   │
                                     └─────────┘
```

### 25 Hz 루프 (BHL `run_locomotion.py` 그대로)

```python
controller = RlController(cfg); controller.load_policy()   # D4 ONNX
robot = Humanoid()
robot.enter_damping()
obs = robot.reset()
while True:                          # RateLimiter(25 Hz)
    actions = controller.update(obs) # 45-dim obs → 12 position target
    obs = robot.step(actions)        # CAN write/read + IMU read
    rate.sleep()
```

`robot.step()` 안에 IDLE → RL_INIT → RL_RUNNING 상태머신이 들어있다
(`humanoid.py`). RL_INIT 은 현재 자세 → default pose 로 100 스텝 선형 보간 →
정책이 갑자기 안 튀게 함.

### 모듈 표

| 모듈 | 출처 | 우리 작업 |
|------|------|-----------|
| `recoil/` (CAN 프로토콜) | BHL `recoil/` | **무수정 이식** |
| `imu.py` (SerialImu) | BHL `robot/imu.py` | 이식 + 포트/IMU 모델 확인 |
| `rl_controller.py` | BHL `policy/rl_controller.py` | **무수정 이식** (ONNX 경로) |
| `humanoid.py` | BHL `robot/humanoid.py` | 이식 + Hylion 적응 (3절) |
| `run_policy.py` | BHL `scripts/run_locomotion.py` | 이식, UDP 텔레메트리 제거 |
| `mock_biped.py` | 신규 | Humanoid 인터페이스 mock (HW 없이 루프 검증) |
| `factory.py` | 신규 | config 로 real/mock 선택 |
| `configs/policy_latest.yaml` | BHL `policy_biped_25hz_a.yaml` | 복사 + ONNX 경로만 변경 |

---

## 2. 파일 레이아웃

현재 비어있는(0 바이트) 파일들을 채운다:

```
nuc/bhl/
  recoil/              # BHL recoil/ 패키지 복사 (core.py, can.py, util.py, fixed16.py)
  imu.py               # BHL robot/imu.py
  rl_controller.py     # BHL policy/rl_controller.py
  humanoid.py          # BHL robot/humanoid.py + Hylion 적응
  run_policy.py        # = BHL run_locomotion.py  ← 진입점
  mock_biped.py        # 신규  ※현재 0 바이트
  factory.py           # 신규  ※현재 0 바이트
configs/
  policy_latest.yaml   # = BHL policy_biped_25hz_a.yaml, checkpoint 경로만 변경  ※현재 0 바이트
checkpoints/biped/stage_d4_hylion_v6/
  best.pt              # (있음) D4 PhysX 학습본
  d4_policy.onnx       # 단계 0 산출물 (신규)
calibration.yaml       # 단계 2 산출물, 매 전원 사이클 갱신
```

NUC 의존성: `python-can`, `pyserial`, `onnxruntime`, `omegaconf`, `loop_rate_limiters`, `numpy`.
BHL `run_locomotion.py` 의 `cc.udp` (텔레메트리) 의존성은 제거하거나 우리 `comm/` 로 대체.

`comm/` (NUC↔Orin) 는 1차 배포 범위 밖. 속도 명령은 gamepad 또는 하드코딩(vx=0).

---

## 3. D4 ↔ BHL 호환성 검증

[play_mujoco.py](../../../sim/mujoco/play_mujoco.py) 와 BHL `rl_controller.py` /
`policy_biped_25hz_a.yaml` 을 한 줄씩 대조한 결과:

| 항목 | D4 (play_mujoco) | BHL biped 러너 | |
|------|------------------|----------------|:--:|
| obs 차원 | 45 | `num_observations: 45`, `history_length: 0` | ✅ |
| obs 순서 | cmd·angvel·grav·qpos·qvel·last_act | 동일 | ✅ |
| joint 순서 (12) | L/R × hip_roll·yaw·pitch·knee·ank_p·ank_r | 동일 | ✅ |
| default pose | `[0,0,-0.2,0.4,-0.3,0]×2` | `default_joint_positions` 동일 | ✅ |
| action | `default + action*0.25` | `clip*action_scale + default` | ✅ |
| projected gravity | `projected_gravity_vec(quat)` | `quat_rotate_inverse(quat,[0,0,-1])` | ✅ 동등 |
| PD 게인 | kp 20 / kd 2 | `joint_kp 20` / `joint_kd 2` | ✅ |
| effort limit | 6 Nm | `effort_limits: 6.0` | ✅ |
| 모터 명령 | position target | `Mode.POSITION` + `transmit_pdo_2` | ✅ |
| 정책 포맷 | RSL-RL 체크포인트 dict | `nn.Module` 또는 ONNX | ⚠ **단계 0 에서 해소** |

**유일한 불일치**: BHL `TorchPolicy` 는 `torch.load()` 후 통째로 호출 — 직렬화된
`nn.Module` 가정. D4 `best.pt` 는 RSL-RL 체크포인트 dict (`model_state_dict`).
→ 단계 0 에서 ONNX 로 export 하면 `OnnxPolicy` 가 그대로 받는다.

---

## 4. 단계별 작업 (코드 레벨)

### 단계 0 — D4 → ONNX export  [DGX, 막힌 것 없음]

actor MLP 구조는 [play_mujoco.py:82-90](../../../sim/mujoco/play_mujoco.py) 와 동일:
`Linear(45,256)·ELU → Linear(256,128)·ELU → Linear(128,128)·ELU → Linear(128,12)`.

신규 스크립트 (예: `sim/isaaclab/scripts/export_onnx.py`):

```python
import torch, torch.nn as nn

CKPT = "checkpoints/biped/stage_d4_hylion_v6/best.pt"
OUT  = "checkpoints/biped/stage_d4_hylion_v6/d4_policy.onnx"

class ActorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(45,256), nn.ELU(),
            nn.Linear(256,128), nn.ELU(),
            nn.Linear(128,128), nn.ELU(),
            nn.Linear(128,12))
    def forward(self, x): return self.net(x)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
# play_mujoco.load_policy 와 동일한 prefix 처리 (actor_state_dict / model_state_dict)
actor = ActorMLP(); actor.eval()
# ... state_dict 로드 (play_mujoco.py:95-103 참조) ...

dummy = torch.zeros(1,45)
torch.onnx.export(actor, dummy, OUT,
                  input_names=["obs"], output_names=["actions"],
                  opset_version=11)
```

**검증**: export 후 onnxruntime 출력 ↔ torch 출력을 동일 obs 로 비교 (오차 < 1e-5).
산출물 `d4_policy.onnx` 를 git LFS 로 커밋 → NUC `git lfs pull`.

`OnnxPolicy` 가 입력 키 `"obs"` 를 먼저 시도하므로 `input_names=["obs"]` 필수.

### 단계 1 — recoil / imu 이식 + 연결 확인  [NUC]

1. BHL `recoil/`, `robot/imu.py` 를 `nuc/bhl/` 로 복사.
2. USB-CAN 어댑터 2개를 SocketCAN 으로 올림:
   ```bash
   sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up
   sudo ip link set can1 type can bitrate 1000000 && sudo ip link set can1 up
   ```
   (slcan 계열 어댑터면 `slcand` 경유 필요. candleLight/gs_usb 는 네이티브.)
3. **CAN 종단저항 120Ω** — 각 버스 양 끝 모터에 있는지 물리 확인. 없으면 통신 간헐 오류.
4. 12 모터 핑: BHL `check_connection.py` 이식 → `Humanoid.check_connection()` 으로
   왼다리 ID `1,3,5,7,11,13` / 오른다리 `2,4,6,8,12,14` 응답 확인.
5. IMU: `SerialImu(port="/dev/ttyUSB?", baudrate=460800)` → quaternion·angular_velocity
   출력 확인. **Hylion IMU 가 BHL 과 같은 시리얼 IMU(WIT/HiPNUC 계열, gyro deg/s)
   인지 확인** — 다르면 `imu.py` 파서 재작성.

### 단계 2 — 캘리브레이션 (3종, 혼동 주의)  [NUC]

> **device ID 굽기 ≠ 관절 캘리브레이션.** 별개의 3가지 작업이다.

| # | 작업 | 스크립트 | 산출/저장 | 주기 |
|---|------|----------|-----------|------|
| 1 | **device ID** (CAN 주소) | `motor/configure_parameter.py` | 모터 flash | 1회 |
| 2 | **전기각 캘리브레이션** (BLDC 정류) | `motor/calibrate_electrical_offset.py` | 모터 flash | 1회 (모터당) |
| 3 | **관절 영점 캘리브레이션** | `calibrate_joints.py` | `calibration.yaml` | **매 전원 사이클** |

- **#1 device ID**: 사용자가 완료. `configure_parameter.py` 로 각 컨트롤러에 ID 부여.
  이것은 CAN 주소일 뿐 — 보행 동작 정렬과 무관.
- **#2 전기각 offset**: `Mode.CALIBRATION` 으로 모터가 ~20초 회전하며 엔코더-자석
  정렬. FOC 토크 제어의 전제. **완료 여부 확인 필요** (안 됐으면 모터가 토크를 못 냄).
- **#3 관절 영점**: `calibrate_joints.py` 가 관절을 기계적 한계로 보내 `limit_readings`
  를 기록, `offsets = limit_readings - ideal_values` 로 `position_offsets`(12) 산출 →
  `calibration.yaml` 저장. **스크립트 주석상 "매 전원 사이클마다 실행"** — `humanoid.py`
  가 부팅 시 이 파일을 로드한다. 발표 당일에도 전원 켤 때마다 재실행.
- `joint_axis_directions`(부호 12개)는 `humanoid.py` 에 하드코딩 (`calibrate_joints.py`
  에도 동일 값). Hylion 조립 방향이 BHL 과 다르면 이 값을 수정해야 함 — 모터 +방향
  과 URDF +방향이 일치하는지 관절별 확인.

### 단계 3 — mock 으로 루프 검증  [NUC, HW 불필요]

`mock_biped.py`: `Humanoid` 와 동일 인터페이스(`reset`, `step`, `enter_damping`,
`stop`)를 갖되 CAN/IMU 대신 가짜 상태를 반환하는 클래스.

```python
class MockBiped:
    """HW 없이 run_policy 루프를 검증. lowlevel_states 를 zeros/MuJoCo 로 채움."""
    def reset(self): ...
    def step(self, actions): ...      # actions 받고 obs 45-dim 반환
    def enter_damping(self): pass
    def stop(self): pass
```

`factory.py`: config 의 `robot.mode` 가 `real` 이면 `Humanoid`, `mock` 이면
`MockBiped` 반환. `run_policy.py` 는 factory 만 호출하므로 코드 분기 없음.

검증 항목:
- 25 Hz RateLimiter 가 실제로 40 ms 주기를 지키는지 (계획서 34 단계 1-5)
- D4 ONNX forward 가 obs 45 → action 12 를 내는지
- obs 조립 순서가 학습과 일치하는지 (`diag` 출력 ↔ play_mujoco 비교)

여기까지 통과하면 → [34_sim2real_plan](34_sim2real_plan_2026-05-02.md) 단계 4
(매달기 → 정지 → 느린 보행 → 속도 증가) 로 진입.

---

## 5. Hylion 고유 적응 체크리스트

`humanoid.py` 이식 시 BHL 기본값을 Hylion 실물에 맞춰야 하는 항목:

- [ ] **모터 device ID 매핑** — `joints` 리스트의 (bus, id, name) 가 실제 배선과 일치
- [ ] **`joint_axis_directions`** (부호 12) — 모터 +회전 = URDF +각도 인지 관절별 확인
- [ ] **`calibration.yaml`** — 단계 2-#3, 매 전원 사이클
- [ ] **torque limit** — BHL `enter_damping()` 은 4 Nm 하드코딩. **D4 는 6 Nm 로 학습**
      ([33_d4_validation_report](33_d4_validation_report_2026-05-02.md)). RL_RUNNING 진입
      전 `write_torque_limit(6)` 으로 학습값에 맞출 것. 4 로 두면 정책이 학습 분포 밖
      토크를 명령 → 모터 클리핑 → 불안정 위험.
- [ ] **IMU 모델/포트** — 단계 1-5
- [ ] **gamepad** — BHL `Humanoid.__init__` 이 `Se2Gamepad` 스레드를 띄움. 물리
      gamepad 없으면 optional 처리하거나 mock. 1차 배포는 vx=0 하드코딩.

---

## 6. 안전 / 미해결 / 결정 필요

### 안전 (계획서 34 위험관리와 동일)
- 첫 통전은 로봇을 **매달아 놓고** (발 안 닿게). RL_INIT 보간 동작만 먼저 확인.
- e-stop 물리 버튼이 닿는 위치. `tests/3_interface/test_emergency_stop.py` (현재 빈 파일) 구현 필요.
- `enter_damping` → `RL_INIT` → `RL_RUNNING` 순서 강제. 바로 RL_RUNNING 진입 금지.

### 미해결 (확인 필요)
- 전기각 캘리브레이션(#2) 완료 여부
- USB-CAN 어댑터 칩셋 (SocketCAN 네이티브 / slcan)
- Hylion IMU 모델 (BHL 시리얼 IMU 호환?)
- 25 Hz 루프가 USB-CAN 왕복 + ONNX forward 를 40 ms 안에 끝내는지 (단계 3 에서 측정)

### 결정 필요
- torque limit: 6 (학습값, 권장) vs 4 (BHL 보수값)
- 속도 명령 입력: gamepad vs 하드코딩 vs `comm/` 경유 Orin

---

## 부록 — 진입점 비교

| | sim2sim (검증됨) | 실배포 (이 문서) |
|---|---|---|
| 진입점 | `sim/mujoco/play_mujoco.py` | `nuc/bhl/run_policy.py` |
| 환경 | MuJoCo `mj_step` | 실모터 CAN + IMU |
| 정책 | `.pt` actor MLP 추출 | `.onnx` (`OnnxPolicy`) |
| obs 조립 | `get_obs()` | `RlController.update()` |
| 제어율 | 25 Hz / 200 Hz substep | 25 Hz, 모터 on-board PD |

`play_mujoco.py` 의 obs 조립·action 스케일이 곧 실배포의 정답지 — 단계 3 에서
mock 출력을 이것과 대조해 검증한다.
