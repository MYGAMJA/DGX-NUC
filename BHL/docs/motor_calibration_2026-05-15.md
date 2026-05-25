# BHL 모터(전기적) 캘리브레이션 작업 기록 (2026-05-15)

> **⚠️ 전제 조건**: 본 문서는 **모든 대상 모터가 `ping`에 응답하는 상태**를 가정함.
> 2026-05-15 현재 ID=1 모터부터 `Motor is offline` 상태 — [BHL_CAN_Debug_Handoff.md](BHL_CAN_Debug_Handoff.md) 의 4단계 진단(candump/cansend → BESC clean build → python-can 직접 recv → CANable candleLight 재플래싱)을 통과해야 본 문서로 진입 가능.
>
> **환경 차이 메모** (핸드오프 기준 ↔ 본 문서 초기 가정):
> - 작업 디렉토리: `~/Berkeley-Humanoid-Lite-Lowlevel` (lowlevel-only fork)
> - CAN: **slcan** (MKS CANable V2.0 Pro), `start_can_transports.sh` 안 씀 → `slcand -o -c -s8 /dev/ttyACM1 can0`
> - 현재 **단일 모터 벤치 테스트** 단계 (12개 데이지체인은 나중)
> - 본 문서의 12개 일괄 루프(§2, §3, §4)는 12개 모터 전부 ping 통과 시점에 사용.

> **이 문서는 "각 모터 단위" 전기적 캘리브레이션** (encoder flux offset / phase order 자동 검출)에 대한 것임.
> **조인트(영점) 캘리브레이션과 다름** → 조인트 캘리브는 `calibration_2026-05-15.md` 참조.

| 캘리브레이션 종류 | 스크립트 | 주기 | 결과 저장 위치 |
|---|---|---|---|
| **모터 전기적** (본 문서) | `scripts/motor/calibrate_electrical_offset.py` | 1회/모터 (펌웨어·하드웨어 교체 시 재실행) | 모터 자체 flash |
| 조인트 영점 | `scripts/calibrate_joints.py` | 매 power cycle | `calibration.yaml` |

---

## 출처

1. **소스 코드 (단일 권위)** — 본 절차에 대한 별도 README/GitBook 문서를 확인할 수 없어, 다음 스크립트들의 코드 자체가 근거:
   - `source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py`
   - `source/berkeley_humanoid_lite_lowlevel/scripts/motor/ping.py`
   - `source/berkeley_humanoid_lite_lowlevel/scripts/motor/read_configurations.py`
   - `source/berkeley_humanoid_lite_lowlevel/scripts/motor/configure_parameter.py`
   - `source/berkeley_humanoid_lite_lowlevel/scripts/motor/move_actuator.py`
   - `source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/recoil/core.py` (Mode/Parameter enum)
   - `source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/robot/humanoid.py` (조인트 ↔ CAN 매핑)
2. `motor_configuration.json` (저장소 루트, 예시 dump) — 정상값 참고용. 특히 `flux_offset: 37.83759307861328` 정상 범위 감 잡는 용.

---

## 동작 원리

[calibrate_electrical_offset.py](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py):

```python
bus.set_mode(device_id, recoil.Mode.CALIBRATION)  # 0x05
time.sleep(20)                                     # 20초 동안 모터가 자체 시퀀스 수행
bus.stop()
```

- `Mode.CALIBRATION = 0x05` ([recoil/core.py:35](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/recoil/core.py#L35))
- 모터 펌웨어가 내부적으로 로터를 천천히 돌리면서 다음을 학습:
  - **phase_order** (모터 결선 방향, ±1)
  - **encoder flux_offset** (encoder 0점 ↔ 전기각 0점 사이 오프셋)
- 사용되는 전류는 `motor.max_calibration_current` (기본 3.0 A — `motor_configuration.json` 참조)
- 캘리브 후 → 결과는 **flash에 저장하지 않으면 전원 사이클 시 사라짐**

> ⚠️ **반드시 모터가 자유롭게 회전 가능한 상태에서만 실행.** 다리에 조립된 상태에서 실행하면 기어/링크 부하 때문에 오프셋이 잘못 잡히거나 모터가 멈춤.
> 가능하면 **다리 들어 올려서 발이 공중에 뜬 상태** 또는 **모터 분리** 상태에서 실행.

---

## CAN ↔ 조인트 ↔ Device ID 매핑

[humanoid.py:50-62](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/robot/humanoid.py#L50-L62) 기준:

| # | CAN | ID | Joint | 비고 |
|---|---|---|---|---|
| 1 | can0 | 1 | left_hip_roll | |
| 2 | can0 | 3 | left_hip_yaw | |
| 3 | can0 | 5 | left_hip_pitch | |
| 4 | can0 | 7 | left_knee_pitch | |
| 5 | can0 | 11 | left_ankle_pitch | |
| 6 | can0 | 13 | left_ankle_roll | |
| 7 | can1 | 2 | right_hip_roll | |
| 8 | can1 | 4 | right_hip_yaw | |
| 9 | can1 | 6 | right_hip_pitch | |
| 10 | can1 | 8 | right_knee_pitch | |
| 11 | can1 | 12 | right_ankle_pitch | |
| 12 | can1 | 14 | right_ankle_roll | |

---

## 0. 사전 점검

- [ ] BHL 전원 ON, can0/can1 케이블 연결
- [ ] **다리가 자유 회전 가능한 상태** (들어 올려서 발이 공중)
- [ ] 다리 주변 안전 거리 (모터가 의도치 않게 회전함)
- [ ] (선택) `motor_configuration.json` 백업 — 캘리브 전 상태 비교용

```bash
cp /home/laba/Berkeley-Humanoid-Lite/motor_configuration.json \
   /home/laba/DGX-NUC/BHL/motor_configuration_before_2026-05-15.json
```

---

## 1. CAN transport 기동

```bash
cd /home/laba/Berkeley-Humanoid-Lite
bash source/berkeley_humanoid_lite_lowlevel/scripts/start_can_transports.sh
```

확인:
```bash
ip -details link show can0 | grep -E "state|bitrate"
ip -details link show can1 | grep -E "state|bitrate"
```

기대: `state UP`, `bitrate 1000000`.

---

## 2. 각 모터 ping (응답 확인)

[ping.py](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/scripts/motor/ping.py) — `-c <can채널> -i <ID>`.

```bash
cd /home/laba/Berkeley-Humanoid-Lite

# left leg (can0)
for ID in 1 3 5 7 11 13; do
  echo "=== can0 id=$ID ==="
  python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/ping.py -c can0 -i $ID
done

# right leg (can1)
for ID in 2 4 6 8 12 14; do
  echo "=== can1 id=$ID ==="
  python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/ping.py -c can1 -i $ID
done
```

기대 출력: `Motor is online`.

결과 (12행 채우기):
```
can0 id=1  : 
can0 id=3  : 
can0 id=5  : 
can0 id=7  : 
can0 id=11 : 
can0 id=13 : 
can1 id=2  : 
can1 id=4  : 
can1 id=6  : 
can1 id=8  : 
can1 id=12 : 
can1 id=14 : 
```

---

## 3. (선택) 캘리브 전 config 덤프

각 모터의 **현재 flux_offset / phase_order**를 기록 — 캘리브 후 변경 확인용.

[read_configurations.py](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/scripts/motor/read_configurations.py)는 `motor_configuration.json`에 **덮어쓰기**(line 63: `open("motor_configuration.json", "w")`)하므로 매번 백업해야 함.

```bash
cd /home/laba/Berkeley-Humanoid-Lite
mkdir -p /home/laba/DGX-NUC/BHL/motor_dumps_before_2026-05-15

for CHAN_ID in "can0:1" "can0:3" "can0:5" "can0:7" "can0:11" "can0:13" \
               "can1:2" "can1:4" "can1:6" "can1:8" "can1:12" "can1:14"; do
  CHAN="${CHAN_ID%:*}"; ID="${CHAN_ID#*:}"
  python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/read_configurations.py -c $CHAN -i $ID
  cp motor_configuration.json /home/laba/DGX-NUC/BHL/motor_dumps_before_2026-05-15/${CHAN}_id${ID}.json
done
```

---

## 4. 모터별 전기적 캘리브레이션 실행

### ⚠️ 안전 체크 (각 모터마다)

- 해당 다리/관절이 **자유 회전** 가능한가? (지면·구조물에 닿지 않게)
- 케이블이 회전축에 감기지 않게
- 캘리브 중 20초 동안 **사람이 만지지 말 것**

### 실행

```bash
cd /home/laba/Berkeley-Humanoid-Lite

# left leg
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can0 -i 1   # left_hip_roll
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can0 -i 3   # left_hip_yaw
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can0 -i 5   # left_hip_pitch
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can0 -i 7   # left_knee_pitch
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can0 -i 11  # left_ankle_pitch
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can0 -i 13  # left_ankle_roll

# right leg
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can1 -i 2   # right_hip_roll
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can1 -i 4   # right_hip_yaw
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can1 -i 6   # right_hip_pitch
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can1 -i 8   # right_knee_pitch
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can1 -i 12  # right_ankle_pitch
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/calibrate_electrical_offset.py -c can1 -i 14  # right_ankle_roll
```

각 명령:
- 모터 모드 → `CALIBRATION` 진입
- **약 20초** 동안 모터가 자체 시퀀스 (로터 회전)
- 종료 후 `bus.stop()`

진행 로그 (✓ 통과 / ✗ 에러):

| # | CAN | ID | Joint | 결과 | 비고 |
|---|---|---|---|---|---|
| 1 | can0 | 1  | left_hip_roll      |   |   |
| 2 | can0 | 3  | left_hip_yaw       |   |   |
| 3 | can0 | 5  | left_hip_pitch     |   |   |
| 4 | can0 | 7  | left_knee_pitch    |   |   |
| 5 | can0 | 11 | left_ankle_pitch   |   |   |
| 6 | can0 | 13 | left_ankle_roll    |   |   |
| 7 | can1 | 2  | right_hip_roll     |   |   |
| 8 | can1 | 4  | right_hip_yaw      |   |   |
| 9 | can1 | 6  | right_hip_pitch    |   |   |
| 10 | can1 | 8  | right_knee_pitch  |   |   |
| 11 | can1 | 12 | right_ankle_pitch |   |   |
| 12 | can1 | 14 | right_ankle_roll  |   |   |

---

## 5. 캘리브 결과 검증 (flux_offset 갱신 확인)

각 모터에서 `read_configurations.py` 다시 실행 → `encoder.flux_offset` 값이 4번 단계 전과 **달라졌는지** 확인.

```bash
cd /home/laba/Berkeley-Humanoid-Lite
mkdir -p /home/laba/DGX-NUC/BHL/motor_dumps_after_2026-05-15

for CHAN_ID in "can0:1" "can0:3" "can0:5" "can0:7" "can0:11" "can0:13" \
               "can1:2" "can1:4" "can1:6" "can1:8" "can1:12" "can1:14"; do
  CHAN="${CHAN_ID%:*}"; ID="${CHAN_ID#*:}"
  python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/read_configurations.py -c $CHAN -i $ID
  cp motor_configuration.json /home/laba/DGX-NUC/BHL/motor_dumps_after_2026-05-15/${CHAN}_id${ID}.json
done

# 차이 비교
for f in /home/laba/DGX-NUC/BHL/motor_dumps_before_2026-05-15/*.json; do
  base=$(basename $f)
  echo "=== $base ==="
  diff <(jq '.encoder.flux_offset, .motor.phase_order' $f) \
       <(jq '.encoder.flux_offset, .motor.phase_order' /home/laba/DGX-NUC/BHL/motor_dumps_after_2026-05-15/$base)
done
```

기대:
- `flux_offset`은 모터마다 다른 실수값 (참고: 저장소 예시 `37.83759307861328`)
- 캘리브 실패 시 0.0 또는 NaN
- `phase_order`는 ±1

비정상 모터 기록:
| CAN | ID | flux_offset (전) | flux_offset (후) | 메모 |
|---|---|---|---|---|
|  |  |  |  |  |

---

## 6. **Flash에 저장 (영구화) — 매우 중요**

캘리브 결과는 RAM에만 있음. 전원 사이클 시 **사라짐**.
[configure_parameter.py:66-68](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/scripts/motor/configure_parameter.py#L66-L68)의 `store_to_flash()`를 호출해야 함.

`configure_parameter.py` 자체는 함수 정의만 있고 실제 호출은 line 94의 `# store_to_flash(motor)`로 주석 처리됨. 그래서 모터별로 다음 1-liner를 직접 실행:

```bash
cd /home/laba/Berkeley-Humanoid-Lite

for CHAN_ID in "can0:1" "can0:3" "can0:5" "can0:7" "can0:11" "can0:13" \
               "can1:2" "can1:4" "can1:6" "can1:8" "can1:12" "can1:14"; do
  CHAN="${CHAN_ID%:*}"; ID="${CHAN_ID#*:}"
  echo "=== flash $CHAN id=$ID ==="
  python3 -c "
import berkeley_humanoid_lite_lowlevel.recoil as recoil
bus = recoil.Bus(channel='$CHAN', bitrate=1000000)
bus.store_setting_to_flash($ID)
bus.stop()
print('Settings stored to flash!')
"
done
```

저장 확인 — 전원 사이클 후 다시 `read_configurations.py` 실행 → `flux_offset` 값이 유지되어야 함.

---

## 7. (선택) 캘리브 후 동작 테스트

[move_actuator.py](file:///home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/scripts/motor/move_actuator.py) — 1 Hz, 1 rad 진폭의 sin파 위치 명령:

```bash
# torque_limit=0.2, kp=0.2, kd=0.005 — 안전한 저토크
python3 source/berkeley_humanoid_lite_lowlevel/scripts/motor/move_actuator.py -c can0 -i 1
# Ctrl+C로 종료
```

부드럽게 진동하면 OK. 진동·이상음·역회전이면 phase_order 의심.

---

## 8. 다음 단계

모터 전기적 캘리브 완료 후 → **조인트 영점 캘리브레이션** (`calibration_2026-05-15.md` §3-4 단계)으로 넘어가야 실제 보행 자세 일치.

---

## 9. 트러블슈팅

| 증상 | 원인 후보 | 조치 |
|---|---|---|
| `ping`은 되는데 calibrate 중 멈춤 | 기계적 부하 (조립 상태 + 외력) | 다리 들어올려 자유회전 상태로 |
| `flux_offset`이 0.0인 채로 끝남 | 캘리브 전류 부족 / 회전 막힘 | `max_calibration_current` 확인 (json 기본 3.0 A), 부하 제거 |
| 캘리브 후에도 진동 | `phase_order` 오검출 | `configure_parameter.configure_phase_order(bus, id, -X)`로 부호 반전 후 재시도 |
| 전원 사이클 후 다시 캘리브해야 함 | flash 저장 안 함 | §6 단계 누락 |

---

## 10. 진행 로그

### 2026-05-15 단일 모터 (can0, id=2) 실행 결과

- 13:56 — `read_configurations.py` (BEFORE) → `flux_offset=0.0`, `phase_order=1`, `device_id=2`, `cal_current=5.0A`
- 13:56:53 — `calibrate_electrical_offset.py -c can0 -i 2` 시작
- 13:57:13 — 정상 종료 (20s)
- 13:57:13 — `read_configurations.py` (AFTER) → **`flux_offset=34.71932601928711`**, `phase_order=1` (변동 없음)
- 백업: `BHL/motor_id2_before_cal.json`, `BHL/motor_id2_after_cal.json`
- 14:02 — Flash 저장 (`bus.store_settings_to_flash(2)`) → return cleanly
  - **주의**: `configure_parameter.py` 의 템플릿(line 67)에는 단수형 `store_setting_to_flash`로 적혀 있는데 실제 메서드명은 **복수형** `store_settings_to_flash`. 템플릿 오타.
- 14:03 — `move_actuator.py -c can0 -i 2` 4초 sin파 테스트
  - 의존성: `loop_rate_limiters` (requirements.txt 누락 → `pip install loop_rate_limiters` 별도)
  - 결과: 통신 200Hz 정상, but **position이 5.55~5.70 rad 범위에서만 진동, sin파 추종 안 됨**
  - 해석: 시작 위치가 5.6 rad이고 명령 범위는 ±1 rad → 위치 오차 항상 5 rad 내외, `torque_limit=0.2 N·m`로는 정적 마찰 못 넘김. **electrical 캘리브 성공 ↔ 별개 이슈**.
  - 후속: 본격 사용 시 `calibrate_joints.py`로 mechanical zero (position_offset) 잡거나, move_actuator 시작 전에 모터를 명령 범위 근처로 수동 회전 후 시작.
- **참고**: 핸드오프 디버깅 결론은 `BHL_CAN_Debug_Handoff.md` 상단 해결 보고 참조. 진짜 원인은 device_id=2 가정 오류였음.

### 2026-05-15 15:18:50 — id=1 (left_hip_roll)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=0.0`, `phase_order=1`
- 결과: ❌ 실패
- 백업: `motor_id1_before_cal.json` / `motor_id1_after_cal.json`

### 2026-05-15 15:30:07 — id=1 (left_hip_roll)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=-1`
- after:  `flux_offset=0.0`, `phase_order=-1`
- 결과: ❌ 실패
- 백업: `motor_id1_before_cal.json` / `motor_id1_after_cal.json`

### 2026-05-15 15:38:42 — id=1 (left_hip_roll)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=31.333097457885742`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id1_before_cal.json` / `motor_id1_after_cal.json`

### 2026-05-15 16:53:26 — id=4 (right_hip_yaw)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=-32.383766174316406`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id4_before_cal.json` / `motor_id4_after_cal.json`

### 2026-05-15 17:03:54 — id=6 (right_hip_pitch)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=0.0`, `phase_order=1`
- 결과: ❌ 실패
- 백업: `motor_id6_before_cal.json` / `motor_id6_after_cal.json`

### 2026-05-15 17:10:01 — id=6 (right_hip_pitch)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=-25.838977813720703`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id6_before_cal.json` / `motor_id6_after_cal.json`

### 2026-05-15 17:13:56 — id=14 (right_ankle_roll)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=0.0`, `phase_order=1`
- 결과: ❌ 실패
- 백업: `motor_id14_before_cal.json` / `motor_id14_after_cal.json`

### 2026-05-15 17:15:30 — id=14 (right_ankle_roll)  [calibrate_one.py]

- before: `flux_offset=-88.53892517089844`, `phase_order=1`
- after:  `flux_offset=-88.53892517089844`, `phase_order=1`
- 결과: ❌ 실패
- 백업: `motor_id14_before_cal.json` / `motor_id14_after_cal.json`

### 2026-05-15 17:24:13 — id=12 (right_ankle_pitch)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=-84.72776794433594`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id12_before_cal.json` / `motor_id12_after_cal.json`

### 2026-05-15 17:28:31 — id=8 (right_knee_pitch)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=0.0`, `phase_order=1`
- 결과: ❌ 실패
- 백업: `motor_id8_before_cal.json` / `motor_id8_after_cal.json`

### 2026-05-15 17:37:18 — id=1 (left_hip_roll)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=-41.047996520996094`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id1_before_cal.json` / `motor_id1_after_cal.json`

### 2026-05-15 17:53:05 — id=5 (left_hip_pitch)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=0.0`, `phase_order=1`
- 결과: ❌ 실패
- 백업: `motor_id5_before_cal.json` / `motor_id5_after_cal.json`

### 2026-05-15 17:59:22 — id=3 (left_hip_yaw)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=-7.464817523956299`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id3_before_cal.json` / `motor_id3_after_cal.json`

### 2026-05-15 18:03:42 — id=7 (left_knee_pitch)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=-53.864952087402344`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id7_before_cal.json` / `motor_id7_after_cal.json`

### 2026-05-15 18:22:01 — id=1 (left_hip_roll)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=0.0`, `phase_order=1`
- 결과: ❌ 실패
- 백업: `motor_id1_before_cal.json` / `motor_id1_after_cal.json`

### 2026-05-15 19:55:04 — id=12 (right_ankle_pitch)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=-3.047628164291382`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id12_before_cal.json` / `motor_id12_after_cal.json`

### 2026-05-15 19:58:11 — id=12 (right_ankle_pitch) 5A 재캘리브  [direct script]

- supply 1A → 5A capacity 업그레이드 후 재캘리브
- cal_current: 3.0A → **5.0A** 로 변경 후 저장
- before: `flux_offset=0.0` (리셋)
- after:  `flux_offset=-3.046331644058228`, `phase_order=1`
- bus_V 모니터: 캘리브 전 과정 23.49~23.81V (절대 20V 아래 안 떨어짐 ✓)
- 결과: ✅ 캘리브 + flash 저장
- 동작 테스트: **94.2°** sin파 추종 (역대 최대 진폭, flux_offset 정확도 우수)

### 2026-05-15 19:59 — 최종 12개 모터 완료

전수 캘리브 + flash 영구화 + 동작 검증 완료.

| # | ID | 조인트 | flux_offset | cal_I | 동작 진폭 | 비고 |
|---|---|---|---|---|---|---|
| 1 | 2 | right_hip_roll | +34.72 | 5.0A | ✓ | Windows 빌드 |
| 2 | 1 | left_hip_roll | +31.33 | 5.0A | ✓ | 우리 패치 (LOAD_ID=0) |
| 3 | 4 | right_hip_yaw | −32.38 | 5.0A | 47.1° | Windows 빌드 |
| 4 | 6 | right_hip_pitch | −25.84 | 5.0A | 37.5° | Windows 빌드 |
| 5 | 14 | right_ankle_roll | −82.55 | 3.0A | — | Windows 빌드 (ankle 변종) |
| **6** | **12** | **right_ankle_pitch** | **−3.05** | **5.0A** | **94.2°** | **5A 재캘리브 (가장 정확)** |
| 7 | 8 | right_knee_pitch | −41.05 | 5.0A | 37.8° | 우리 패치 (LOAD_ID=1) |
| 8 | 5 | left_hip_pitch | −23.76 | 5.0A | 30.5° | Windows 빌드 |
| 9 | 3 | left_hip_yaw | −7.46 | 5.0A | 66.8° | Windows 빌드 |
| 10 | 7 | left_knee_pitch | −53.86 | 5.0A | 36.9° | Windows 빌드 |
| 11 | 13 | left_ankle_roll | +0.79 | 3.0A | 40.0° | Windows 빌드 (ankle 변종) |
| 12 | 11 | left_ankle_pitch | −44.15 | 0.5A | 30° | 우리 패치 (LOAD_ID=1), 저전류 cal |

---

## 📅 내일 (2026-05-16) 계획: 전수 5A 재캘리브

### 동기
- id=12 (5A 재캘리브 결과) 동작 진폭 94° = 다른 모터 30~67° 대비 우월
- 다른 10개 모터는 1A 제한 supply에서 캘리브돼 정확도 의심
- 균일한 정확도로 보행 성능 최적화

### 사전 준비
1. ✅ supply 5A로 설정돼 있음 (어제 마무리)
2. 각 모터의 cal_current를 5A로 변경 필요 (Windows 빌드 모터들은 현재 3.0~5.0A)
3. 각 모터 BESC + 모터 페어 보관 상태 확인

### 절차 (모터당 ~5분)

```python
# 모터 1개당:
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel python3 <<EOF
import berkeley_humanoid_lite_lowlevel.recoil as recoil
import time
bus = recoil.Bus('can0', 1000000)
dev = <ID>  # 해당 모터 id
# 1. cal_I 5A로 강제 (현재 3A인 ankle 모터들 위해)
bus._write_parameter_f32(dev, recoil.Parameter.MOTOR_MAX_CALIBRATION_CURRENT, 5.0)
time.sleep(0.1)
# 2. flux 리셋
bus._write_parameter_f32(dev, recoil.Parameter.ENCODER_FLUX_OFFSET, 0.0)
# 3. CAL 모드 + bus_V 모니터
bus.set_mode(dev, recoil.Mode.CALIBRATION)
for i in range(44):
    bv = bus._read_parameter_f32(dev, recoil.Parameter.POWERSTAGE_BUS_VOLTAGE_MEASURED)
    if bv < 20: print(f"⚠️ bus_V drop {bv}V at t={i*0.5}s")
    time.sleep(0.5)
# 4. 결과 + flash 저장
flux = bus.read_encoder_flux_offset(dev)
print(f"flux_offset = {flux}")
if abs(flux) > 0.01:
    bus.store_settings_to_flash(dev)
    print("✓ flash stored")
bus.set_mode(dev, recoil.Mode.IDLE)
bus.stop()

---

## 📅 내일 (2026-05-16) 계획: 전수 5A 재캘리브

### 동기
- id=12 (5A 재캘리브 결과) 동작 진폭 94° = 다른 모터 30~67° 대비 우월
- 다른 10개 모터는 1A 제한 supply에서 캘리브돼 정확도 의심
- 균일한 정확도로 보행 성능 최적화

### 사전 준비
1. ✅ supply 5A로 설정돼 있음
2. 각 모터의 cal_current를 5A로 변경 필요 (ankle 모터들은 현재 3.0A)
3. 각 모터 BESC + 모터 페어 보관 상태 확인

### 모터당 절차 (~5분)

각 모터 연결 후 `BHL/recal_one.py` 또는 다음 인라인 스크립트 실행:

1. `cal_I = 5.0A` 로 강제 (`MOTOR_MAX_CALIBRATION_CURRENT` write)
2. `flux_offset = 0` 으로 리셋
3. `Mode.CALIBRATION` 진입 + 22초 sleep
4. **중요**: cal 도중 `bus_V` 모니터 → 23V 유지 = OK, 5-10V로 drop = supply 부족
5. flux_offset 변화 확인 후 `store_settings_to_flash`
6. 동작 테스트 (sin파 ±0.5 rad around start) → 진폭 1차와 비교

### 모터별 순서 (제안)

오른쪽 다리:
| # | ID | 조인트 | 1차 flux | 1차 동작 |
|---|---|---|---|---|
| 1 | 2 | right_hip_roll | +34.72 | ✓ |
| 2 | 4 | right_hip_yaw | −32.38 | 47° |
| 3 | 6 | right_hip_pitch | −25.84 | 37.5° |
| 4 | 8 | right_knee_pitch | −41.05 | 37.8° |
| 5 | 14 | right_ankle_roll | −82.55 | — |
| ~~6~~ | ~~12~~ | ~~right_ankle_pitch~~ | ~~5A 재캘리브 완료 (−3.05, 94°)~~ | |

왼쪽 다리:
| # | ID | 조인트 | 1차 flux | 1차 동작 |
|---|---|---|---|---|
| 7 | 1 | left_hip_roll | +31.33 | ✓ |
| 8 | 3 | left_hip_yaw | −7.46 | 66.8° |
| 9 | 5 | left_hip_pitch | −23.76 | 30.5° |
| 10 | 7 | left_knee_pitch | −53.86 | 36.9° |
| 11 | 11 | left_ankle_pitch | −44.15 (0.5A cal) | 30° |
| 12 | 13 | left_ankle_roll | +0.79 | 40° |

### 주의사항

- `bus_V` 모니터 필수 — supply 5A 진짜 받쳐주는지 매번 확인
- bus_V drop 발생 시 → cal_I 더 낮추거나 supply 점검
- 각 모터 후 동작 테스트로 진폭 변화 확인 (1차 vs 5A)
- 모터 12 (id=11, left_ankle_pitch)는 ID 영구화 SDO 거쳐야 함 (LOAD_ID=1 patched build)
- 모터 7 (id=8, right_knee_pitch)도 동일

### 예상 시간
12개 × 5분 = **약 1시간**

### 재캘리브 후 다음 단계
1. 본체 조립
2. 데이지체인 펌웨어 (중간 11개는 PC14=RESET 변종)
3. `calibrate_joints.py` 영점 캘리브
4. 보행 테스트

### 2026-05-22 18:00:50 — id=1 (left_hip_roll)  [calibrate_one.py]

- before: `flux_offset=0.0`, `phase_order=1`
- after:  `flux_offset=-11.7579984664917`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id1_before_cal.json` / `motor_id1_after_cal.json`

### 2026-05-22 19:52:12 — id=7 (left_knee_pitch)  [calibrate_one.py]

- before: `flux_offset=-53.864952087402344`, `phase_order=1`
- after:  `flux_offset=-79.39668273925781`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id7_before_cal.json` / `motor_id7_after_cal.json`

### 2026-05-22 22:04:34 — id=7 (left_knee_pitch)  [calibrate_one.py]

- before: `flux_offset=-79.39668273925781`, `phase_order=1`
- after:  `flux_offset=-35.416133880615234`, `phase_order=1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id7_before_cal.json` / `motor_id7_after_cal.json`

### 2026-05-22 22:16:04 — id=7 (left_knee_pitch)  [calibrate_one.py]

- before: `flux_offset=-35.416133880615234`, `phase_order=-1`
- after:  `flux_offset=86.06014251708984`, `phase_order=-1`
- 결과: ✅ 캘리브 + flash 저장
- 백업: `motor_id7_before_cal.json` / `motor_id7_after_cal.json`

---

### 2026-05-22 ⚠️ 정정 — id=7 19:52 / 22:04 / 22:16 결과 해석 변경

calibrate_one.py의 "flux_offset 변동 = ✅" 판정은 약함. id=7의 위 세 시도는 폐루프 동작 테스트(`/tmp/move_test.py`)로 다시 검증한 결과:

- **19:52** (flux=−79.40, po=+1) → 폐루프 buzz, **❌ 실제로는 실패** (캘리브 자체는 진행됐으나 동작 안 됨)
- **22:04** (flux=−35.42, po=+1) → 폐루프 buzz, **❌ 실제로는 실패**
- **22:16** (flux=86.06, po=−1) → 이동추종 9/9 통과 = **✅ 진짜 성공**. 단 phase_order=−1이 flash에 영구 저장 안 됨 — 당시 펌웨어 버그 (`motor_controller.c:252` 0xFFFFFFFF 가드). 전원 사이클 시 phase_order=+1로 복귀 → 다시 buzz.

원인 진단·해결 과정 상세: [id7_phase_order_2026-05-22.md](id7_phase_order_2026-05-22.md).

---

### 2026-05-23 — id=7 phase_order persist 버그 펌웨어 fix + 폐루프 검증

- `motor_controller.c:252`의 `0xFFFFFFFF` 가드를 phase_order=−1과 충돌 안 하게 수정 후 재빌드.
- 산출물: `BHL/firmware_id7_phasefix_termOFF_20260523.bin`, `firmware_version=0x20260523`.
- ST-LINK로 id=7에 플래시 → flash 영구 저장 검증:
  - ST-LINK reset 후 `phase_order=−1` 유지 ✓
  - USB/ST-LINK 분리 + 24V 전원 OFF/ON 후 `phase_order=−1` 유지 ✓
- 폐루프 이동추종 (`/tmp/move_test.py`):
  - 자유회전, ilim=2.5A, ±0.5 rad 9단계 sin (start +2.5648 rad 기준)
  - 결과: **✅ 9/9 통과**, 측정 이동 범위 0.988 rad (명령 1.00 rad), 최대 추종오차 0.081 rad
- 사고/복구: 작업 중 실수로 id=1에 phasefix 펌웨어가 잠깐 플래시됐으나 즉시 원본 `BHL/firmware_termOFF.bin`으로 복구. id=1 config page는 건드리지 않음 (2026-05-24 단독 점검에서 확인).
- 결론: id=7은 다른 11개 모터와 동일하게 부팅 시마다 자신의 phase_order(−1)가 안정 로드됨. **추가 전기 캘리브 불필요.**

---

### 2026-05-24 — 조립 전 사전 단독 점검 (id=1, id=13)

다리 조립 전 각 보드의 config page 무사 + 캘리브값 유지 확인. read-only.

| ID | device_id | firmware | flux_offset | phase_order | ERROR | VBUS | 판정 |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0x20250226 | −11.7580 | +1 | 0x0 | 23.72V | ✓ 정상 (2026-05-22 18:00 캘리브값 유지) |
| 7 | 7 | 0x20260523 (phasefix) | +86.0601 | **−1** | 0x0 | 23.71V | ✓ 정상 (5-23 검증 후 유지) |
| 13 | 13 | 0x20250226 | −68.3217 | +1 | 0x0 | 23.71V | ✓ 정상 (기존 캘리브값 유지) |

체인 sweep (id=1/7/13 동시 검출) + 흔들림 테스트는 왼다리 조립 시점에 재개.

---

### 2026-05-24 — id=8 SWD flash dump (`motor_id8_after_cal.json`의 `flux_offset=0.0`은 잘못된 데이터)

`motor_id8_after_cal.json` (timestamp 5-20 14:35) 의 `flux_offset=0.0`이 의심돼 ST-LINK SWD로 보드 flash를 직접 dump (`st-flash read 0x0801F800 2048`) → struct 파싱. **보드 flash의 진짜 값**:

| 필드 | SWD dump 값 | 검증 |
|---|---|---|
| device_id (offset 0x000) | 8 | ✓ |
| firmware_version (0x004) | 0x20250226 | ✓ 원본 |
| Motor.pole_pairs (0x104) | 14 | ✓ |
| Motor.torque_constant | 0.0895799994 | ✓ id=7 backup과 동일 |
| **Motor.phase_order** | **+1** | ✓ 정상 |
| Motor.max_calibration_current | 5.0 A | — |
| Encoder.cpr (0x120) | 4096 | ✓ |
| Encoder.velocity_filter_alpha | 0.7153904438 | ✓ |
| **Encoder.flux_offset (0x13C)** | **−41.04800033569336** | ✓ **5-15 최종표 −41.05와 일치** |

→ id=8은 **추가 캘리브 불필요**. `motor_id8_after_cal.json`의 flux_offset 0.0 → −41.04800033569336로 정정 완료.

**원인 추정**: 5-20 14:35에 read_configurations.py로 일괄 dump 시점에 1A supply / 통신 글리치 / 변수 미초기화 등으로 flux_offset이 0.0으로 잘못 read됨. 보드 flash 자체는 정상.

**시사점 — SWD flash dump > CAN read의 신뢰도**: 같은 5-20 백업의 `motor_id5_after_cal.json`도 `flux_offset=0.0` → 동일 글리치 의심. 실제 진실은 5-15 최종표의 −23.76일 가능성 높음. SWD 확인 권장 (또는 24V + CAN으로 정상 read).

**참고**: 어제(5-23) CAN으로 read한 id=5 값은 `flux_offset=-17.4827`였으나 그 시점 VBUS=4.19V로 비정상 → 이 값도 신뢰도 낮음. SWD dump가 최종 진실.

---

### 2026-05-24 — id=5 SWD flash dump (5-20 json 0.0 정정 + 5-15 최종표도 메모 오류 의심)

같은 방법(ST-LINK SWD → `st-flash read 0x0801F800 2048` → struct 파싱)으로 id=5 BESC 확인:

| 필드 | SWD dump 값 | 검증 |
|---|---|---|
| device_id | 5 | ✓ |
| firmware_version | 0x20250226 | ✓ 원본 |
| Motor.pole_pairs | 14 | ✓ |
| Motor.torque_constant | 0.0895799994 | ✓ |
| **Motor.phase_order** | **+1** | ✓ 정상 |
| Motor.max_calibration_current | 5.0 A | — |
| Encoder.cpr | 4096 | ✓ |
| **Encoder.flux_offset** | **−17.482736587524414** | — |

→ id=5 추가 캘리브 **불필요**. `motor_id5_after_cal.json`의 flux_offset 0.0 → −17.482736587524414로 정정 완료.

**4가지 값 비교**:

| 출처 | flux_offset | 평가 |
|---|---|---|
| 5-15 최종표 | −23.76 | ❓ **메모 오류** 또는 그 후 재캘리브됨 |
| 5-20 json | 0.0 | ❌ read 글리치 |
| 5-23 CAN read (VBUS=4.19V 비정상) | −17.48 | ✓ flash 변수 read는 정확 |
| **5-24 SWD dump** | **−17.48** | ✓ ground truth |

**시사점**:
- id=5/8 모두 5-20 json의 0.0이 글리치임을 확정.
- 5-15 최종표는 id=8에는 맞았지만 id=5에는 다른 값 → 표 전체를 액면 그대로 신뢰하기 어려움. **각 모터의 진실은 SWD dump로만 확정 가능**.
- 어제 id=5 CAN read 결과(VBUS 비정상 상태에서도 정확한 −17.48)는 flash 변수 read는 MCU만 살아있으면 정확함을 보여줌. cal 실행 등 모터를 돌리는 동작은 24V 필요하지만, **read는 5V만 있어도 됨**.

---

## 2026-05-24 종합 — 모든 BESC 현재 상태 (신뢰도 기준)

다양한 출처(SWD dump, CAN read, 5-20 일괄 json, 5-15 최종표)에서 모은 데이터를 신뢰도별로 정리. **SWD dump = ground truth**. 그 외는 검증 필요.

### 출처 신뢰도 등급

| 등급 | 출처 | 비고 |
|---|---|---|
| **A (확정)** | 5-24 SWD flash dump | MCU debug 인터페이스로 flash 직접 read. 최고 신뢰도 |
| **B (높음)** | 5-23~5-24 CAN read (VBUS 정상) | MCU 동작 정상 + 통신 안정 시 정확 |
| **C (중간)** | 5-20 일괄 json | id=5/8에서 글리치 확인됨. 의미 있는 값이라도 SWD 확인 권장 |
| **D (낮음)** | 5-15 최종표 / 5-20 json 0.0 | 메모 오류 또는 글리치. SWD로만 확정 가능 |

### 모터별 현재 상태 (12개)

| ID | 조인트 | CAN | flux_offset | phase_order | firmware | 등급 | 출처 / 메모 |
|---:|---|:-:|---:|:-:|---|:-:|---|
|  **1** | left_hip_roll   | can0 | **−11.7579984664917**   | +1 | 0x20250226           | **A** | 5-24 SWD dump (5-22 CAN cal 결과 −11.7580과 정확 일치) |
|  2 | right_hip_roll      | can1 | +34.7193                | +1 | 0x20250226           | C | 5-20 json (글리치 없음, 5-15 표와 일치) |
|  3 | left_hip_yaw        | can0 | −7.4648                 | +1 | 0x20250226           | C | 5-20 json (5-15 표와 일치) |
|  4 | right_hip_yaw       | can1 | −32.3838                | +1 | 0x20250226           | C | 5-20 json (5-15 표와 일치) |
|  **5** | left_hip_pitch  | can0 | **−17.482736587524414** | **+1** | 0x20250226       | **A** | **5-24 SWD dump** (5-15 표 −23.76은 메모 오류 추정) |
|  6 | right_hip_pitch     | can1 | −25.8390                | +1 | 0x20250226           | C | 5-20 json (5-15 표와 일치) |
|  **7** | left_knee_pitch | can0 | **+86.06014251708984**  | **−1** | **0x20260523 phasefix** | **A+** | **5-23 SWD + 5-24 CAN + 9/9 폐루프 검증** |
|  **8** | right_knee_pitch| can1 | **−41.04800033569336**  | **+1** | 0x20250226       | **A** | **5-24 SWD dump** (5-15 표 −41.05와 일치) |
| 11 | left_ankle_pitch    | can0 | (5-15 표) −44.15        | +1? | (미확인)             | D | json 백업 없음. 5-15 표 0.5A cal, 30° — SWD 확인 강력 권장 |
| 12 | right_ankle_pitch   | can1 | −3.0476                 | +1 | 0x20250226           | C | 5-20 json (5-15 표와 일치, 5A 재캘리브) |
| **13** | left_ankle_roll | can0 | **−68.3217**            | +1 | 0x20250226           | B | 5-24 CAN read (VBUS=23.71V 정상) |
| 14 | right_ankle_roll    | can1 | json: −88.5389 / 표: −82.55 ❗ | +1 | 0x20250226   | D | 두 값 불일치 → SWD 확인 필요 |

### 정리

- **확정 (A 등급) 4개**: id=1, 5, 7, 8. 5-24 SWD로 ground truth 확보.
- **높은 신뢰 (B 등급) 1개**: id=13. 5-24 정상 VBUS에서 CAN read (SWD 미확인).
- **중간 (C 등급) 5개**: id=2, 3, 4, 6, 12. 5-20 json과 5-15 표가 일치 (둘 다 의미있는 값) → 실제로 맞을 가능성 높음.
- **불확실 (D 등급) 2개**: id=11(데이터 부족), id=14(json/표 불일치).

### 권장 다음 작업

1. **id=11, id=14를 SWD로 확인** — 미확인/불일치라 진실 모름. 캘리브 필요 여부도 모름.
2. (선택) C 등급 모터들(id=2, 3, 4, 6, 12)도 SWD로 한번 훑으면 모든 보드 ground truth 확정.
3. 다른 보드에 phasefix 펌웨어가 잘못 적용된 사례 없는지 SWD 시 firmware_version 같이 확인 (지금까지 id=7만 0x20260523, 나머지는 0x20250226).
4. 모든 보드 확정 후 다리 조립 진행 → 체인 sweep + 흔들림 테스트.
