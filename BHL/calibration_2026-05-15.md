# BHL 조인트 캘리브레이션 작업 기록 (2026-05-15)

> 대상 로봇: Berkeley-Humanoid-Lite (다리 12 DOF, 현재 팔 비활성)
> 실행 머신: NUC (`/home/laba/Berkeley-Humanoid-Lite/`)
> 본 문서의 절차는 **공식 출처 2건 + 소스 코드 교차검증**에 근거함.

## 출처

1. **공식 README** — `Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/README.md`
   ("Joint Calibration" 섹션, line 46-58)
2. **소스 코드** (README가 일부 outdated이므로 교차검증):
   - `source/berkeley_humanoid_lite_lowlevel/scripts/calibrate_joints.py`
   - `source/berkeley_humanoid_lite_lowlevel/scripts/start_can_transports.sh`
   - `source/berkeley_humanoid_lite_lowlevel/scripts/check_connection.py`
   - `source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/policy/gamepad.py`
   - `source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/robot/humanoid.py`
3. GitBook (`berkeley-humanoid-lite.gitbook.io/docs`) — 해당 페이지 404, 출처 불가.

## README ↔ 실제 코드 불일치 (반영함)

| 항목 | README 표기 | 실제 (코드 기준) | 본 문서에서 채택 |
|---|---|---|---|
| 캘리브 스크립트 경로 | `./berkeley_humanoid_lite_lowlevel/robot/calibrate_joints.py` | `source/berkeley_humanoid_lite_lowlevel/scripts/calibrate_joints.py` | 실제 경로 |
| 종료 키 | "press `q` or `B` button" | 키보드 핸들러 없음. **gamepad BTN_X** 또는 좌·우 썸스틱 클릭으로 `mode_switch=1` | gamepad 입력만 |
| CAN bringup 경로 | `./scripts/start_can_transports.sh` | `source/berkeley_humanoid_lite_lowlevel/scripts/start_can_transports.sh` | 실제 경로 |
| sudo 필요 여부 | "sudo" prefix | 스크립트 내부에 이미 `sudo ip link ...` 포함 | sudo 없이 호출 가능 (단 패스워드 프롬프트는 뜸) |

---

## 0. 사전 점검

- [ ] BHL 전원 ON, can0/can1 케이블 연결 확인
- [ ] IMU USB 연결 ([humanoid.py:65](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/robot/humanoid.py#L65) 에서 `SerialImu` 자동 시작 — 안 잡히면 `Humanoid()` 생성 단계에서 멈춤)
- [ ] Xbox 호환 게임패드 USB 연결 (XInput)
- [ ] 작업 공간 확보 — 사람이 두 다리를 limit까지 직접 밀어야 함

> ⚠️ 안전: `Humanoid()` 인스턴스를 만들면 모터 transport가 활성화되고, IMU 스레드, gamepad 스레드가 시작됨. 다리 주변에 사람·물체 없도록.

---

## 1. CAN transport 기동

**공식 (README line 24-26)** — sudo 포함이지만, 내부 스크립트가 이미 sudo를 호출하므로 둘 다 가능:

```bash
cd /home/laba/Berkeley-Humanoid-Lite
bash source/berkeley_humanoid_lite_lowlevel/scripts/start_can_transports.sh
```

스크립트 내용 (4 라인 — `can0~can3` 모두 1Mbps로 up):
```
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
sudo ip link set can2 up type can bitrate 1000000
sudo ip link set can3 up type can bitrate 1000000
```

확인:
```bash
ip -details link show can0 | grep -E "state|bitrate"
ip -details link show can1 | grep -E "state|bitrate"
```

실행 결과 / 에러:
```
(여기에 출력 붙여넣기)
```

---

## 2. 연결 점검

**공식 (README line 32-34)**:

```bash
python3 source/berkeley_humanoid_lite_lowlevel/scripts/check_connection.py
```

내부적으로 `Humanoid().check_connection()` 호출 → 12개 조인트 ([humanoid.py:50-62](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/robot/humanoid.py#L50-L62)) 모두 응답해야 정상:

| Bus | CAN ID | Joint |
|---|---|---|
| can0 (left_leg) | 1 | left_hip_roll |
| can0 | 3 | left_hip_yaw |
| can0 | 5 | left_hip_pitch |
| can0 | 7 | left_knee_pitch |
| can0 | 11 | left_ankle_pitch |
| can0 | 13 | left_ankle_roll |
| can1 (right_leg) | 2 | right_hip_roll |
| can1 | 4 | right_hip_yaw |
| can1 | 6 | right_hip_pitch |
| can1 | 8 | right_knee_pitch |
| can1 | 12 | right_ankle_pitch |
| can1 | 14 | right_ankle_roll |

실행 결과:
```
(붙여넣기)
```

---

## 3. (선택) Joystick 점검

캘리브 종료는 gamepad로만 가능하므로, 먼저 동작 확인을 권장.

```bash
python3 source/berkeley_humanoid_lite_lowlevel/scripts/test_joystick.py
```

[gamepad.py:103-117](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/policy/gamepad.py#L103-L117) 기준 mode_switch 매핑:

| 버튼 조합 | mode_switch | 의미 |
|---|---|---|
| `BTN_X` 또는 좌/우 썸스틱 클릭 | **1** | **IDLE — 캘리브 종료 트리거** |
| `BTN_A` + `BTN_BUMPER_L` | 2 | RL_INIT |
| `BTN_A` + `BTN_BUMPER_R` | 3 | RL_RUNNING |

→ 캘리브 종료 시 누를 키: **X 버튼** (또는 썸스틱 클릭).

---

## 4. 캘리브레이션 실행

**공식 (README line 52-58)** — 단, 경로는 실제 위치로 정정:

```bash
cd /home/laba/Berkeley-Humanoid-Lite
python3 source/berkeley_humanoid_lite_lowlevel/scripts/calibrate_joints.py
```

### 4-1. 동작 원리 ([calibrate_joints.py](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/scripts/calibrate_joints.py))

> "Because the joint actuators only have single encoder on the motor shaft, we need to calibrate the zero position of the joints after each power cycle." — 공식 README

- 스크립트 실행 중 12개 조인트를 **기계적 limit**까지 사람이 밀고 있음
- 루프가 `limit_readings`에 min/max 값을 누적
- 누른 자세에서 그 각도를 "limit에서의 측정값"으로 간주
- `offset = limit_reading − ideal_value` 계산
- `calibration.yaml`에 `position_offsets:` 저장

### 4-2. 목표 자세 — `ideal_values` ([calibrate_joints.py:29-43](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/scripts/calibrate_joints.py#L29-L43))

| idx | 조인트 | 좌 (deg) | 우 (deg) | 누적 방향 |
|---|---|---|---|---|
| 0/6 | hip_yaw | −10 | +10 | min / max |
| 1/7 | hip_roll | +33.75 | −33.75 | max / min |
| 2/8 | hip_pitch | +56.25 | +56.25 | max / max |
| 3/9 | knee | 0 | 0 | min / min |
| 4/10 | ankle_pitch | −45 | −45 | min / min |
| 5/11 | ankle_roll | −15 | +15 | min / max |

> ⚠️ 인덱스 순서 주의: `humanoid.py`의 joints 리스트 순서(roll-yaw-pitch-knee-ankle_pitch-ankle_roll)와 `calibrate_joints.py`의 ideal_values 순서가 같다고 가정함. 보정 후 자세가 비정상이면 이 매핑 먼저 의심.

→ 실행 후 위 표 방향으로 각 관절을 한계까지 밀고 있어야 함. 한 번이라도 min/max가 갱신되면 그 위치를 기억함.

### 4-3. 종료 — gamepad **X 버튼** (`mode_switch=1`)

### 4-4. 출력 기록

initial readings (rad):
```
(붙여넣기)
```

final readings at the limits (rad):
```
(붙여넣기)
```

offsets (rad) — `calibration.yaml`에 저장될 값:
```
(붙여넣기)
```

---

## 5. calibration.yaml 보관

[calibrate_joints.py:82-83](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/scripts/calibrate_joints.py#L82-L83): 파일은 **현재 작업 디렉토리**에 저장 (`open("calibration.yaml", "w")` 상대 경로).

따라서 `cd /home/laba/Berkeley-Humanoid-Lite` 한 상태로 실행하면 → `/home/laba/Berkeley-Humanoid-Lite/calibration.yaml`

```bash
ls -la /home/laba/Berkeley-Humanoid-Lite/calibration.yaml

# 본 작업 폴더로 백업 (날짜 태그)
cp /home/laba/Berkeley-Humanoid-Lite/calibration.yaml \
   /home/laba/DGX-NUC/BHL/calibration_2026-05-15.yaml
```

내용 (`position_offsets:` 리스트 12개):
```yaml
(붙여넣기)
```

---

## 6. 사후 검증

[run_idle.py](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/scripts/run_idle.py)는 zero action을 보내며 측정 위치를 UDP로 송신함. 다리에 가해지는 토크가 없으므로 자세 확인용으로 안전:

```bash
cd /home/laba/Berkeley-Humanoid-Lite
python3 source/berkeley_humanoid_lite_lowlevel/scripts/run_idle.py
```

- [ ] 측정 위치(`robot.joint_position_measured`) 값이 합리적인지 확인 (ideal 자세에서 0 근처)
- [ ] 비정상 관절이 있으면 표 6-1에 기록

### 6-1. 이상 관절 기록

| 관절 idx | 이름 | offset (rad) | 메모 |
|---|---|---|---|
|  |  |  |  |

---

## 7. 트러블슈팅

| 증상 | 원인 후보 | 조치 |
|---|---|---|
| `Humanoid()` 생성 단계에서 멈춤 | IMU 시리얼 포트 못 잡음 | `ls /dev/ttyUSB* /dev/ttyACM*`, baudrate 460800 |
| check_connection이 일부 조인트 응답 X | CAN ID 충돌, 모터 펌웨어, 전원 부족 | 해당 ID의 모터만 따로 ping. `motor_configuration.json` 확인 |
| 캘리브 종료 안 됨 (X 버튼 무반응) | gamepad 미인식 / 비-XInput 패드 | `test_joystick.py`로 mode_switch 값 변하는지 확인 |
| offset 값이 비정상적으로 큼 (>1 rad) | limit까지 충분히 안 밀렸음 | 재실행 후 더 확실히 끝까지 밀기 |
| `calibration.yaml`은 나왔는데 idle 자세가 이상 | joints 리스트 순서 ↔ ideal_values 순서 매핑 의심 | §4-2의 매핑 재확인 후 수동 검증 |

---

## 8. 진행 로그

- 2026-05-15 hh:mm — 시작
- (실행/이벤트 줄별 기록)
