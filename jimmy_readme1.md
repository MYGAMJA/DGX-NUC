# Jimmy README #1 — BHL 모터 점검 / 캘리브레이션 세션 요약

> 2026-05-15 작업 세션 정리. `/home/laba/DGX-NUC/BHL/` 폴더의 4개 md 파일을 기준으로 현장 점검 + id=12 모터 재캘리브 진행.

---

## 1. BHL 폴더 md 4개 요약

### 1-1. [BHL/BHL_CAN_Debug_Handoff.md](BHL/BHL_CAN_Debug_Handoff.md)
CAN 통신 디버깅 히스토리. 2026-05-15에 해결됨.

진짜 원인 3건:
1. 첫 BESC가 `device_id=1`이 아니라 **2**로 플래싱돼 있었음 → ID sweep으로 찾음
2. 새 BESC: **GND(검정) 선 헐겁게 빠짐** → 차동신호 전위 기준 없어서 응답 디코딩 실패
3. `FIRST_TIME_BOOTUP=1` 펌웨어 버그로 flash config page에 zero값(`device_id=0`)이 박힘 → `MotorController_init` 이전에 `storeConfig`가 불려서 발생. ST-Link로 config page를 0xFF로 erase해서 compile-time defaults로 부팅하게 만들어 해결.

핵심 교훈: **새 BESC 디버깅 첫 단계는 항상 ID sweep 1~14**. 가정 금지.

### 1-2. [BHL/WORKFLOW.md](BHL/WORKFLOW.md)
12개 모터 단일 캘리브레이션 종합 워크플로 확정판.

- HW 점검 (CAN 3선 GND 포함 + 자석 + BESC 마운팅 + 24V)
- `calibrate_one.py` 한 줄로 자동화 (ID sweep → before/after 백업 → cal → flash 저장 → log 추가)
- 13개 함정 회고 (GND 빠짐, FIRST_TIME_BOOTUP, NaN/0xFFFFFFFF guard 부족, 빌드 시스템 패치, cpr=−1 encoder fault, WRP 보호 등)
- **12개 모터 전부 캘리브 + flash 저장 + 동작 테스트 완료 (12/12)** 표 기록

### 1-3. [BHL/motor_calibration_2026-05-15.md](BHL/motor_calibration_2026-05-15.md)
모터별 전기적 캘리브 실행 로그. 각 실행 timestamp + before/after flux_offset 기록.

조인트 ↔ Device ID 매핑:

| CAN | ID | Joint | CAN | ID | Joint |
|---|---|---|---|---|---|
| can0 | 1 | left_hip_roll | can1 | 2 | right_hip_roll |
| can0 | 3 | left_hip_yaw | can1 | 4 | right_hip_yaw |
| can0 | 5 | left_hip_pitch | can1 | 6 | right_hip_pitch |
| can0 | 7 | left_knee_pitch | can1 | 8 | right_knee_pitch |
| can0 | 11 | left_ankle_pitch | can1 | 12 | right_ankle_pitch |
| can0 | 13 | left_ankle_roll | can1 | 14 | right_ankle_roll |

### 1-4. [BHL/calibration_2026-05-15.md](BHL/calibration_2026-05-15.md)
조인트(영점) 캘리브레이션 절차. 모터 전기적 캘리브와 별개 — **매 power cycle마다 필요**.

- `calibrate_joints.py`로 12 DOF를 limit까지 사람이 직접 밀어 offset 측정 → `calibration.yaml`에 저장
- README ↔ 실제 코드 불일치 정정 (스크립트 경로, 종료키는 gamepad **X 버튼**, sudo는 스크립트 내부에 이미 있음)
- 종료 조건: gamepad `BTN_X` 또는 좌/우 썸스틱 클릭 → `mode_switch=1` (IDLE)
- 본 세션에서는 아직 미실행 (모터 단위 캘리브 단계)

---

## 2. 현재 연결 상태 진단 (현장 점검)

### 2-1. 초기 상태
- CANable Pro USB 인식 OK (`16d0:117e`)
- `slcand -o -c -s8 /dev/ttyACM1 can0` 실행 중, can0 UP / ERROR-ACTIVE
- ST-Link/V2.1 USB 연결 (`0483:374b`) → BESC 1개 USB-C로 연결됨
- 24V 전원: 초기엔 OFF 추정 (CAN sweep 응답 0건)

### 2-2. 1차 sweep — BESC 응답 0건
```
echo (0x201~0x20E)만 잡힘. BESC 응답 (0x181~0x18E) 없음.
```
→ WORKFLOW 함정 10 진단: MCU는 USB-C로 살지만 CAN 트랜시버는 24V 필요. 24V ON 확인 요청.

### 2-3. USB 재연결 → can0 사망
- CANable이 재연결되어 `/dev/ttyACM1` → `/dev/ttyACM2`로 디바이스명 바뀜
- 옛 slcand가 사라진 디바이스 가리켜서 can0 down
- 복구 (사용자가 직접 실행):
  ```bash
  sudo killall slcand
  sudo slcand -o -c -s8 /dev/ttyACM2 can0
  sudo ip link set up can0
  ```

### 2-4. 2차 sweep — id=13 응답
```
can0  18D  [8]  CA 00 00 00 73 26 00 08
```
CAN으로 직접 읽은 config:
- `device_id = 13` (left_ankle_roll)
- `flux_offset = 0.7940` ← WORKFLOW 표 11행의 +0.79와 일치
- `phase_order = 1`
- `max_calibration_current = 3.0 A`

→ **캘리브 완료된 상태 그대로 flash에 보존** (전원 사이클 후에도 유지됨).

### 2-5. 모터 교체 후 3차 sweep — id=12 응답
```
can0  18C  [8]  CA 00 00 00 73 26 00 08
```
CAN으로 직접 읽은 config:
- `device_id = 12` (right_ankle_pitch)
- `flux_offset = 0.0` ← **캘리브 안 됨**
- `phase_order = 1`
- `max_calibration_current = 3.0 A`

WORKFLOW 표 6행은 id=12 flux_offset=−84.73으로 기록되어 있었으나, 이번 보드는 0.0. 두 가지 가능성:
1. 같은 BESC인데 자석 재부착으로 인코더 영점 바뀜
2. device_id=12로 설정만 된 다른 BESC (캘리브 안 된 새 보드)

---

## 3. id=12 재캘리브레이션 실행

명령:
```bash
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel \
  python3 /home/laba/DGX-NUC/BHL/calibrate_one.py
```

자동 흐름:
1. ID sweep → id=12 단일 감지
2. BEFORE config 백업 → `BHL/motor_id12_before_cal.json`
3. `Mode.CALIBRATION` (0x05) 진입 → 20초 자체 시퀀스
4. AFTER config 백업 → `BHL/motor_id12_after_cal.json`
5. `bus.store_settings_to_flash(12)` 호출 → flash 영구화
6. `motor_calibration_2026-05-15.md`에 로그 추가

결과:

| 항목 | before | after |
|---|---|---|
| flux_offset | 0.0 | **−3.0476** |
| phase_order | 1 | 1 |
| flash 저장 | | OK |
| 결과 | | 캘리브 + flash 저장 성공 |

**해석**: −3.05라는 작은 값은 id=13의 +0.79와 비슷한 범위. id=13처럼 정상 캘리브 사례 존재. 이전 기록(−84.73)과는 다르나, 이번에 잡힌 값이 현재 hardware의 정답이며 flash에 영구 저장됨.

---

## 4. 이번 세션에서 배운 점 / 메모

### 4-1. CANable 재연결 → ACM 디바이스명 드리프트
- USB가 잠깐 끊겼다 다시 붙으면 `/dev/ttyACMx`의 x가 바뀜 (1 → 2 등)
- slcand는 옛 path를 계속 가리켜서 can0이 죽음
- 복구는 `killall slcand` → 새 path로 재시작
- 향후: udev rule로 CANable에 고정 symlink (예: `/dev/canable`) 만들면 방지 가능

### 4-2. ST-Link RAM dump 해석은 펌웨어 빌드 종속
- 본 세션에서 `0x20000228` 오프셋이 controller 구조체와 안 맞음 (device_id=0으로 보임)
- 확실한 검증은 **CAN으로 직접 `_read_parameter_*`** 호출
- WORKFLOW.md §5의 RAM dump 스니펫은 특정 빌드 기준이므로, 빌드 바뀌면 오프셋 재확인 필요

### 4-3. flux_offset 값의 크기는 절대 기준이 아님
- 캘리브된 12개 모터 분포: +0.79, −7.46, −23.76, −25.84, −32.38, −34.72, −41.05, −44.15, −53.86, −82.55, −84.73 (등)
- 모터 + 자석 부착 상태에 따라 ±100 사이 어디든 가능
- **0.0 / NaN만 아니면 일단 통과**, 동작 테스트(`move_actuator.py`)로 최종 검증

### 4-4. CAN 응답 없으면 첫 의심은 24V OFF
- MCU는 USB-C 5V로 살아서 ST-Link 통신은 됨
- CAN 트랜시버는 24V→5V/3.3V 변환 레일로 동작 → 24V OFF면 죽음
- ping sweep 0건이면 → 즉시 24V ON 확인

---

## 5. 다음 단계 후보

1. **id=12 동작 테스트** — `move_actuator.py -c can0 -i 12`로 sin파 추종 확인
2. **나머지 모터들 일괄 검증** — 12개 모두 데이지체인 + ping sweep으로 flux_offset 현황 일괄 확인
3. **본체 조립** — 종단저항 펌웨어 변형 (양 끝 ON, 중간 OFF)
4. **조인트 영점 캘리브** — `calibrate_joints.py` (gamepad X 버튼 종료)
5. **CANable udev rule** — `/dev/canable` symlink로 ACM 드리프트 방지 (선택)

---

## 6. 참고 파일 / 경로

| 경로 | 용도 |
|---|---|
| `/home/laba/Berkeley-Humanoid-Lite-Lowlevel/` | BHL lowlevel-only fork (벤치 테스트용) |
| `/home/laba/Berkeley-Humanoid-Lite/` | BHL 메인 저장소 |
| `/home/laba/Recoil-Motor-Controller-BESC/` | BESC 펌웨어 소스 (패치 적용 상태) |
| `/home/laba/DGX-NUC/BHL/calibrate_one.py` | 1모터 캘리브 자동화 헬퍼 |
| `/home/laba/DGX-NUC/BHL/motor_id{N}_before_cal.json` | 캘리브 전 config 백업 |
| `/home/laba/DGX-NUC/BHL/motor_id{N}_after_cal.json` | 캘리브 후 config 백업 |
| `/home/laba/DGX-NUC/BHL/motor_calibration_2026-05-15.md` | 모터별 실행 로그 (계속 추가됨) |
