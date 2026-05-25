# WALKING — Hylion 실배포 스택

BHL Lowlevel 기반 실로봇 보행 실행 코드 (NUC에서 실행).  
조이스틱 없이 **시간 기반 시퀀스**로 동작함.

---

## 현재 상태

| 항목 | 상태 |
| --- | --- |
| ONNX 모델 | ✅ `checkpoints/stage_a_biped.onnx`, `stage_d4_hylion_v6.onnx` |
| Policy config | ✅ `configs/policy_latest.yaml` (D4 기준), `policy_stage_a.yaml`, `policy_d4.yaml` |
| 실행 코드 | ✅ `scripts/run_policy.py` (시퀀스 기반, 조이스틱 없음) |
| 관절 캘리브 | ✅ `scripts/calibrate_joints.py` (Ctrl+C로 종료) |
| NUC 이전 | ⬜ 미완료 — 이 폴더 통째로 NUC에 복사 필요 |
| joint_axis_directions | ⬜ 실물 조립 후 확인 필요 |
| IMU 포트 | ⬜ NUC에서 `/dev/ttyUSB?` 확인 필요 |

---

## 폴더 구조

```
WALKING/
  berkeley_humanoid_lite_lowlevel/
    recoil/             CAN 프로토콜 패키지
    policy/             rl_controller, config
    robot/              humanoid, imu
  scripts/
    start_can.sh        CAN 인터페이스 셋업
    stop_can_transports.sh
    calibrate_joints.py 관절 영점 캘리브 (매 전원 사이클)
    check_connection.py 12 모터 핑 테스트
    test_imu.py         IMU 연결 테스트
    run_policy.py       메인 진입점
  configs/
    policy_latest.yaml  현재 사용 정책 (D4)
    policy_d4.yaml      Hylion D4 — 현재 최고 성능
    policy_stage_a.yaml BHL 기본 biped
  checkpoints/
    stage_d4_hylion_v6.onnx   Hylion D4 (권장)
    stage_a_biped.onnx        BHL 기본 biped (참고용)
  calibration.yaml      관절 영점값 (매 전원 사이클 갱신)
  requirements.txt
```

---

## 정책 선택

```bash
# D4 Hylion (권장)
python3 scripts/run_policy.py --config configs/policy_d4.yaml

# BHL 기본 biped (참고용)
python3 scripts/run_policy.py --config configs/policy_stage_a.yaml
```

`policy_latest.yaml`은 현재 D4로 설정되어 있음. 바꾸려면 `policy_checkpoint_path`만 수정.

---

## 매 전원 사이클 실행 순서

### 1. CAN 인터페이스 올리기

```bash
sudo bash scripts/start_can.sh
```

can0 (왼다리 ID 1,3,5,7,11,13), can1 (오른다리 ID 2,4,6,8,12,14).  
CANable V2 같은 slcan 계열 어댑터는 먼저 slcand로 올려야 함:

```bash
sudo slcand -o -c -s8 /dev/ttyACM0 can0
sudo ip link set up can0
```

### 2. 모터 연결 확인

```bash
python3 scripts/check_connection.py
```

12개 모터 전부 OK가 나와야 다음 단계로.

### 3. 관절 영점 캘리브

```bash
python3 scripts/calibrate_joints.py
```

- 실행하면 모든 관절이 댐핑 모드로 전환됨 (손으로 밀 수 있음)
- 12개 관절을 기계적 한계까지 수동으로 밀어줌
- 완료 후 `Ctrl+C` → `calibration.yaml` 자동 저장
- **매 전원 사이클마다 필요** (엔코더가 절대 위치를 잃어버리므로)

### 4. 정책 실행

```bash
python3 scripts/run_policy.py --config configs/policy_latest.yaml
```

---

## 시퀀스 설정 (`run_policy.py`)

`SEQUENCE` 리스트만 수정해서 동작을 정의함.

```python
SEQUENCE = [
    # (지속시간(초), 상태,          vx,   vy,   vyaw)
    (4.0,  State.RL_INIT,    0.0,  0.0,  0.0 ),  # default pose로 이동
    (5.0,  State.RL_RUNNING, 0.3,  0.0,  0.0 ),  # 앞으로 5초
    (3.0,  State.RL_RUNNING, 0.0,  0.2,  0.0 ),  # 오른쪽으로 3초
    (3.0,  State.RL_RUNNING, 0.0,  0.0,  0.5 ),  # 제자리 회전 3초
    (0.0,  State.IDLE,       0.0,  0.0,  0.0 ),  # 정지
]
```

| 상태 | 설명 |
| --- | --- |
| `RL_INIT` | 현재 자세 → default pose 선형 보간 (100 스텝, 약 4초) |
| `RL_RUNNING` | 정책 실행. vx/vy/vyaw 명령 따름 |
| `IDLE` | 댐핑 모드 (모터 힘 빠짐) |

속도 단위: m/s (vx, vy), rad/s (vyaw). 언제든 `Ctrl+C`로 중단 가능.

---

## Hylion 적응 체크리스트

NUC에 처음 올릴 때 확인/수정해야 할 항목들.

- [ ] **`joint_axis_directions` 부호 12개** — 모터 +방향이 URDF +방향과 일치하는지 실물 확인  
  ([humanoid.py](berkeley_humanoid_lite_lowlevel/robot/humanoid.py) L30 근처)
- [ ] **IMU 포트** — `SerialImu(port="/dev/ttyUSB0")` → NUC 실제 포트 확인 후 수정
- [ ] **CAN 어댑터 타입** — SocketCAN 네이티브(gs_usb)면 `start_can.sh` 그대로, slcan 계열이면 `slcand` 먼저

---

## NUC 최초 셋업

```bash
sudo apt install build-essential net-tools can-utils python3-pip
pip install -r requirements.txt
```

의존성: `pyserial`, `python-can`, `onnxruntime`, `omegaconf`, `loop_rate_limiters`, `numpy`

---

## 안전 수칙

1. **첫 통전은 반드시 로봇을 매달아 놓고** (발이 땅에 닿지 않게)
2. RL_INIT 단계 (default pose 보간) 먼저 확인 후 RL_RUNNING 진입
3. 이상 동작 시 즉시 `Ctrl+C` → 댐핑 모드 진입
