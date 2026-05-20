# BHL 모터 단일 캘리브레이션 — 종합 워크플로 (2026-05-15 확정판)

> 본 문서는 **2026-05-15 12개 모터 전수 캘리브 작업의 실수와 검증된 절차**를 종합한 것임.
> 모든 모터 캘리브 + flash 영구화 + 동작 검증 완료 (12/12).
> 다음 모터 작업 또는 재캘리브 시 이 문서만 따르면 됨.
>
> 관련 파일:
> - `calibrate_one.py` — 한 모터 캘리브 자동화 스크립트
> - `BHL_CAN_Debug_Handoff.md` — 디버깅 히스토리 (참고용)
> - `motor_calibration_2026-05-15.md` — 모터별 실행 로그
> - `calibration_2026-05-15.md` — 조인트(영점) 캘리브레이션 (별도, 본 문서 이후 단계)

---

## TL;DR (정상 모터 1개 처리 5분 이내)

```bash
# 1. 하드웨어 확인 (사람이): GND선 포함 CAN 3선, 자석(글루건), BESC 케이스 고정, 24V/USB-C 연결
# 2. NUC에서:
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel \
  python3 /home/laba/DGX-NUC/BHL/calibrate_one.py
```

펌웨어 빌드/플래시는 **이미 끝남** — 패치된 `.bin`이 보존됨 (`/home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Debug/Recoil-Motor-Controller-B-G431B-ESC1.bin`). 공장 새 BESC를 받으면 1회 reflash만 추가:

```bash
st-flash --reset write \
  /home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Debug/Recoil-Motor-Controller-B-G431B-ESC1.bin \
  0x8000000
```

---

## 1. 하드웨어 사전 점검 (사람이)

### 1-1. CAN 3선 — **3개 다** 단단히 체결

| 색 | 신호 | NUC측 (CANable Pro) | BESC측 (J1 패드) |
|---|---|---|---|
| 🟡 노랑 | CAN-H | H 단자 | H 패드 |
| 🟢 녹색 | CAN-L | L 단자 | L 패드 |
| ⚫ 검정 | **GND** | G 단자 | G 패드 |

⚠️ **GND(검정) 빠지면 차동 신호 기준점 없어서 BESC가 송신해도 NUC가 디코딩 못함.** (2026-05-15에 이걸로 1시간 날렸음)

### 1-2. 자석 (AS5600 인코더용)

- 종류: **diametrically magnetized** (직경방향 자화). 축방향 자화는 **안 됨** — AS5600이 자기장 방향 못 읽음
- 일반 사양: 직경 6mm × 두께 2.5mm 네오디뮴
- 부착 위치: **모터 출력축 반대쪽 로터 샤프트 끝** (rear stub) — 출력축 아님
- 부착 방법: **핫멜트 글루건** (BHL 공식 권장) — 자석 측면 둘레에만 살짝, 자석 윗면(AS5600 마주보는 면) 덮지 말 것
- 거리: 자석 윗면 ↔ AS5600 IC = **0.5~3mm** (이상적 1~2mm)
- 중심 정렬: offset 0.25mm 이내

### 1-3. BESC 마운팅

- BESC는 모터 케이스/하우징에 **단단히 고정** — 전선만으로 매달려 있으면 안 됨
- AS5600 IC가 자석 정중앙 향하도록 정렬
- 모터 회전 시 BESC는 절대 흔들리지 말아야 함

### 1-4. 전원 + 통신 연결

- 24V 출력 OFF 상태에서 모든 케이블 연결
  - 24V 전원 → BESC
  - CAN 3선 → BESC J1
  - USB-C → BESC (펌웨어 flash용, 이미 flash된 BESC면 생략 가능)
- 24V 출력 ON
- BESC LED 확인:
  - USB-C 단자 옆 적색 깜빡임 = ST-Link LD2 (펌웨어 무관, 항상 깜빡임)
  - **보드 우측/중앙 적색 깜빡임** = firmware LD3 (heartbeat — 정상 동작 시그널)
  - 녹색 2개 solid = 5V/3.3V 전원 양호

---

## 2. 펌웨어 빌드 환경 (이미 세팅됨, 새 NUC라면 1회)

### 설치
```bash
sudo apt-get install -y gcc-arm-none-eabi stlink-tools
pip install loop_rate_limiters    # move_actuator.py 의존성 (requirements.txt 누락)
```

### 빌드 시스템 패치 (한 번만 필요, 이미 적용됨)
1. **`Debug/makefile`**: Windows 경로 → Linux 상대경로
   ```
   C:\Users\TK\Desktop\...\STM32G431CBUX_FLASH.ld → ../STM32G431CBUX_FLASH.ld
   ```
2. **`Debug/*/subdir.mk`**: `-fcyclomatic-complexity` 플래그 제거 (gcc 10.3 미지원)
3. **`Debug/objects.list`**: `find . -name "*.o" | sed 's|^\./||' > objects.list`
   (Eclipse가 자동 생성하는 파일, CLI 빌드 시 수동 필요)

### 펌웨어 소스 패치 (이미 적용됨, 새 BESC펌웨어 받으면 다시 적용 필요)

`Core/Src/main.c` line ~898 (이미 적용, uncommitted):
```c
HAL_GPIO_WritePin(GPIO_CAN_TERM_GPIO_Port, GPIO_CAN_TERM_Pin, GPIO_PIN_SET);  // 종단 ON
```

`Core/Inc/motor_controller_conf.h`:
```c
#define FIRST_TIME_BOOTUP               0    // 절대 1로 두지 말 것 (펌웨어 버그 트리거)
#define LOAD_ID_FROM_FLASH              0    // 0이어도 무방. flash가 깨끗할 때 default(1) 사용
#define LOAD_CONFIG_FROM_FLASH          1
#define LOAD_CALIBRATION_FROM_FLASH     1
#define DEVICE_CAN_ID                   1    // 컴파일 기본값
```

`Core/Src/motor_controller.c` `MotorController_loadConfig` 함수:
- **모든 float 필드 NaN-guard**: `if (!isnan(controller_config->X)) controller->X = controller_config->X;`
- **모든 uint32_t 필드 0xFFFFFFFF-guard**: `if (controller_config->X != 0xFFFFFFFF) controller->X = ...;`
- 보호 대상: `flux_offset`, `gear_ratio`, `position_kp/ki`, `velocity_kp/ki`, `torque_limit`, `velocity_limit`, `position_limit_lower/upper`, `position_offset`, `torque_filter_alpha`, `i_limit/kp/ki`, `undervoltage/overvoltage_threshold`, `bus_voltage_filter_alpha`, `torque_constant`, `max_calibration_current`, `velocity_filter_alpha` (float들) + `watchdog_timeout`, `fast_frame_frequency`, `pole_pairs`, `phase_order`, `cpr` (int들)
- 이유: 공장 상태 / `0xFF`로 erased된 flash에서 부팅해도 init 실패 안 함

### 빌드 명령

```bash
cd /home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Debug
make clean
make all
arm-none-eabi-objcopy -O binary Recoil-Motor-Controller-B-G431B-ESC1.elf Recoil-Motor-Controller-B-G431B-ESC1.bin
```

산출물: `Debug/Recoil-Motor-Controller-B-G431B-ESC1.bin` (~58KB)

---

## 3. 새 BESC당 펌웨어 flash (공장 출하 / Windows에서 flash된 BESC만)

```bash
# USB-C 연결 후
lsusb | grep 0483                              # ST-Link 인식 확인
st-info --probe                                # 칩 정보 (STM32G4 Cat-2 = G431CB)

st-flash --reset write \
  /home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Debug/Recoil-Motor-Controller-B-G431B-ESC1.bin \
  0x8000000
```

기대 출력: `Flash written and verified! jolly good!`

---

## 4. 캘리브레이션 실행

```bash
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel \
  python3 /home/laba/DGX-NUC/BHL/calibrate_one.py
```

스크립트가 자동으로:
1. ID sweep 1~14 → 응답하는 모터 1개 검출
2. BEFORE config 백업 → `BHL/motor_id{N}_before_cal.json`
3. CALIBRATION 모드 진입 후 20초 대기 (로터 자동 회전)
4. AFTER config 백업 → `BHL/motor_id{N}_after_cal.json`
5. flux_offset 변화 검증 (0 → 값 학습되면 성공)
6. `store_settings_to_flash` 자동 호출 → 영구화
7. `motor_calibration_2026-05-15.md` 자동 로그

성공 기준: `flux_offset : 0.0 → <30~40 사이 값>   (changed)` 출력.

---

## 5. 빠른 진단 — controller RAM 상태 보기

작동 안 할 때 BESC 상태 즉시 확인:

```bash
st-flash read /tmp/c.bin 0x20000228 32 2>/dev/null
python3 -c "
import struct
raw = open('/tmp/c.bin','rb').read()
device_id, fw, watchdog, ff_freq, mode, error = struct.unpack('<6I', raw[:24])
print(f'device_id = {device_id}')
print(f'mode      = {mode}  (1=IDLE 2=DAMPING 5=CALIBRATION)')
print(f'error     = 0x{error:04x}')
errs = {0x0004:'INIT', 0x0008:'CAL', 0x0010:'PWR', 0x0020:'INVALID_MODE',
        0x0040:'WATCHDOG', 0x1000:'I2C_FAULT', 0x2000:'ENCODER_FAULT'}
print('errors:', [n for v,n in errs.items() if error & v])
"
```

정상 상태: `device_id=1, mode=1, error=0x0000`

비정상 → 다음 표 참조.

---

## 6. 흔한 실수 / 함정 (지난 작업 회고)

### 함정 1: 첫 BESC가 `device_id=1` 일 거라 가정 → 실제 2였음
- **증상**: `ping -i 1` 무응답
- **검증**: ID sweep 1~14 (`for N in $(seq 1 14); do cansend can0 $(printf '%03X' $((0x200|N)))#CA; done` + `candump can0`)
- **교훈**: 첫 단계는 항상 ID sweep. 가정 금지.

### 함정 2: GND 선이 헐겁게 빠져 있는데 못 알아챔
- **증상**: 펌웨어/I2C/flash/등 모든 가설이 안 맞음. echo는 잡히는데 BESC 응답 0건.
- **검증**: CAN 3선 시각적으로 단단히 박혀있는지 손으로 확인
- **교훈**: SW 디버깅 들어가기 전 항상 **물리 연결부터** 재확인

### 함정 3: `FIRST_TIME_BOOTUP=1` 흔적
- **증상**: 펌웨어 reflash해도 `device_id` 등이 0으로 로드됨. RAM dump 시 `device_id=0`.
- **원인**: 이전 누군가 `FIRST_TIME_BOOTUP=1` 로 빌드/flash 했었음. [app.c:108-112](file:///home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Core/Src/app.c#L108-L112)는 `MotorController_init` **이전에** storeConfig를 호출 → zero-init된 controller가 flash에 그대로 저장됨 → flash에 device_id=0, kp=0 등 쓰레기값 박힘.
- **해결**: loadConfig를 NaN/0xFFFFFFFF-guard로 패치 (위 §2의 펌웨어 소스 패치) → 빈 flash든 쓰레기 flash든 init 실패 안 함.
- **교훈**: `FIRST_TIME_BOOTUP=1`은 절대 빌드/flash하지 말 것. 설계 자체가 버그.

### 함정 4: Flash erase 후 INITIALIZATION_ERROR
- **증상**: flash config page를 0xFF로 채웠더니 `error=0x0004 (INITIALIZATION_ERROR)` 발생, main loop hang. CAN ping은 응답함 (ISR 동작).
- **원인**: [motor_controller.c:89-105](file:///home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Core/Src/motor_controller.c#L89-L105): loadConfig가 NaN 만나면 `return HAL_ERROR` → init status != HAL_OK → ERROR_INITIALIZATION_ERROR set + `while(1)` 무한 루프 (UART로만 송신).
- **해결**: loadConfig 패치 (NaN/0xFFFFFFFF-guard, return HAL_ERROR 제거)
- **교훈**: ISR 기반 CAN 응답은 main loop hang과 무관. ping 되더라도 main loop가 살아있다는 보장 X. **RAM dump로 mode/error 확인 필수.**

### 함정 5: cpr 필드 = 0xFFFFFFFF → encoder fault
- **증상**: AS5600은 정상 (raw 값 0~4095 범위), but `error=0x2000 (ENCODER_FAULT)` 계속 set.
- **원인**: [encoder.c:55-57](file:///home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Core/Src/encoder.c#L55-L57): `if (raw_reading >= abs(cpr)) return 0x04;`. cpr이 flash에서 0xFFFFFFFF (=−1 as int32) 로드 → `abs(−1)=1` → 어떤 raw 값이든 1 이상 → fault.
- **해결**: loadConfig에서 cpr도 0xFFFFFFFF-guard.
- **교훈**: float만 NaN-guard로는 부족. **int 필드도 모두 guard 필요**.

### 함정 6: 인코더 magnet 자체 문제
- **증상**: error=0x2000 (ENCODER_FAULT)지만 cpr은 정상. RAM dump의 i2c_buffer가 0xFF 또는 4096 초과 값.
- **원인 후보**:
  - 자석이 axial magnetized (직경방향 아님)
  - 자석 중심 정렬 어긋남
  - 자석-IC 거리 >3mm 또는 <0.5mm
  - 자석 헐겁게 떨어져 있음
  - 자석이 모터 회전축에 단단히 고정 안 됨
- **검증**: RAM dump에서 encoder.i2c_buffer (offset 자기 모델 따라 다름; 본 빌드에서는 controller 시작 + 0x118) 가 0-4095 범위인지 확인. 손으로 모터 살짝 돌리면서 값이 0-4095 사이에서 변하면 OK.

### 함정 7: 빌드 시행착오로 시간 낭비
- **증상**: 펌웨어 패치를 점진적으로 추가하느라 4번 reflash.
- **교훈**: 한 번에 **모든** guard 적용 (float NaN + int 0xFFFFFFFF). 함수 전체 다시 살피기.

### 함정 8: `make`가 `clean`만 돌고 `all` 안 함
- **증상**: `make -j$(nproc)`이 clean만 출력하고 끝남.
- **원인**: `make`의 첫 타겟이 명시 안 됨. Eclipse 생성 makefile은 `all`이 첫 명시 타겟이지만 일부 환경에서 안 잡힘.
- **해결**: `make all` 명시.

### 함정 9: objects.list 없음 → link 실패
- **증상**: 컴파일은 모두 성공, 마지막 link에서 `No rule to make target 'objects.list'`.
- **원인**: Eclipse가 자동 생성하는 파일을 CLI 빌드는 만들지 않음.
- **해결**: 빌드 후 link 직전에 `find . -name "*.o" | sed 's|^\./||' > objects.list`

### 함정 10: USB-C로 flash 가능하지만 CAN transceiver는 24V 필요
- **증상**: ST-Link로 flash + reset 했지만 CAN 응답 안 옴 (24V OFF 상태였음)
- **원인**: BESC의 CAN 트랜시버 칩은 24V→5V/3.3V 변환 레일로 전원받음. 24V OFF면 트랜시버 죽음. MCU만 USB로 동작.
- **교훈**: CAN 테스트 전 24V ON 필수.

### 함정 11: 공장 출하 BESC는 flash 쓰기 보호(WRP/RDP) 걸려 있음
- **증상**: `st-flash write` → `Flash memory is write protected`. `st-flash erase` 도 동일 에러.
- **원인**: ST 공장 펌웨어가 option byte에 WRP/RDP 설정. stlink-tools만으론 해제 불가.
- **해결** (NUC에서):
  ```bash
  sudo apt install openocd
  openocd -f interface/stlink.cfg -f target/stm32g4x.cfg \
    -c "init; reset halt; stm32l4x unlock 0; exit"
  # 그래도 WRP 안 풀리면 직접 FLASH_OPTR + WRP1AR/BR 레지스터 manipulation
  ```
- 또는 Windows STM32CubeProgrammer GUI의 "Option Bytes" 탭 사용
- **교훈**: 공장 BESC는 첫 reflash 전 WRP 해제 필요. 한 번 풀면 그 후엔 st-flash 정상 동작.

### 함정 12: `MOTOR_CALIBRATION_CURRENT=5A` (공식 default) ↔ 공급기 한계 충돌
- **공식 default** ([motor_profiles.h:22](file:///home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Core/Inc/motor_profiles.h#L22)): M6C12 150KV → 5A
- **사용자 환경**: UTP3315TFL 24V/1A 제한 → 5A 도달 불가
- **증상**: cal 도중 phase_current가 한계 초과 시 supply가 CC 모드 진입 → 전압 강하 → BESC undervoltage 셧다운 → 회전 멈춤 → flux_offset 부정확/실패
- **검출**: `bus_V`를 cal 중 모니터링 (23~24V 유지 = OK / 5~10V로 떨어짐 = supply 한계 초과)
- **해결**:
  - 공식 환경 (충분한 supply): cal_current=5A 그대로
  - 1A 제한 환경: `bus._write_parameter_f32(dev, recoil.Parameter.MOTOR_MAX_CALIBRATION_CURRENT, 0.5)` 로 낮춤 후 cal
- **교훈**: cal 전 항상 `bus_V` 모니터링 또는 cal_current를 공급 한계의 50% 이하로 설정.

### 함정 13: 모터 위치가 명령 범위와 멀면 `move_actuator.py` sin파 추종 실패
- **증상**: 캘리브 정상 완료, 통신 OK인데 sin 명령에 반응 안 함.
- **원인**: 시작 위치가 명령 ±0.5 rad 범위와 멀면 (예: -2.1 rad) torque_limit 작아서 stiction 못 넘김.
- **해결**: target = `start_pos + sin(t)*0.5` 로 시작 위치 기준 진동. (`move_actuator.py` 기본은 ±1 rad around 0)

### 함정 14: STM32 BOOT0 핀 hardware로 high 고정 → ROM bootloader에서 못 벗어남
- **증상**: ST-Link로 flash 성공 + verified OK인데 reset 후 PC가 `0x1FFF...` (System Memory). CAN/펌웨어 무동작.
- **원인**: B-G431B-ESC1의 BOOT0 핀이 high 상태. Option byte의 nSWBOOT0=1 (factory default)이면 BOOT0 핀 따라감.
- **해결**: option byte로 BOOT0 핀 무시 + flash 부팅 강제:
  ```bash
  openocd -f interface/stlink.cfg -f target/stm32g4x.cfg -c "
    init; reset halt
    mww 0x40022008 0x45670123; mww 0x40022008 0xCDEF89AB
    mww 0x4002200C 0x08192A3B; mww 0x4002200C 0x4C5D6E7F
    sleep 100
    mww 0x40022020 0xFAEFF8AA   # nSWBOOT0=0 (bit24), nBOOT0=0 (bit26)
    mww 0x40022014 0x00020000   # OPTSTRT
    sleep 500
    mww 0x40022014 0x08000000   # OBL_LAUNCH
    sleep 1500
    exit"
  ```
- **함정 안의 함정**: bit 위치 헷갈리기 쉬움. STM32G4에서 **nBOOT0은 bit 26** (bit 25 아님). 0xFEEFF8AA로 쓰면 bit 26 그대로 1 = ROM 부팅 유지됨. 0xFAEFF8AA가 정답.

### 함정 15: 공장 BESC가 USB mass storage (DAPLink/MSC) 드래그-앤-드롭 flash 지원
- ST-Link가 USB-C로 NUC에 연결되면 `/media/laba/DIS_G431CB` 등으로 자동 마운트됨
- `.bin` 파일을 그 폴더에 복사 → 자동 flash. st-flash 안 되는 경우(RDP/WRP 등) 대안:
  ```bash
  cp firmware.bin /media/laba/DIS_G431CB/
  ```
- DETAILS.TXT, MBED.HTM 표시되면 정상 DAPLink. FAIL.TXT 생기면 flash 실패.

### 함정 16: 공급기 한계가 cal_current 보다 작으면 cal sequence 도중 supply 죽음
- **증상**: cal 도중 `bus_V`가 23V → 5V로 급락 → BESC undervoltage 셧다운 → 회전 정지 → flux_offset 안 변함 (또는 ERROR_CALIBRATION_ERROR set)
- **원인**: M6C12 default cal_current=5A. UTP3315TFL 1A 제한 supply로는 cal sequence peak 못 받쳐줌
- **검출 방법**: cal 중 `bus_V` 실시간 모니터 — 23V 유지면 OK, 5-10V로 떨어지면 supply 부족
- **해결**:
  - 강한 supply (≥2-3A) 사용 + cal_current 5A (공식 default)
  - 또는 1A supply + cal_current 0.5A로 낮춤 (`bus._write_parameter_f32(dev, recoil.Parameter.MOTOR_MAX_CALIBRATION_CURRENT, 0.5)`)
- **경험 비교** (id=12 right_ankle_pitch):
  - 1A supply + cal_I=3A → flux_offset=−84.73 (부정확)
  - 5A supply + cal_I=5A → flux_offset=−3.05 (정확, 동작 진폭 30° → 94° 향상)

### 함정 17: UTP3315TFL 전류 한계 설정 — COARSE 노브로 메인 조정 (FINE은 ±0.5A만)
- **증상**: I 노브 돌려도 0.5A까지밖에 안 올라감
- **원인**: FINE 노브만 돌렸음. FINE 범위는 ~0.5A뿐
- **해결**: **Current COARSE** 노브로 먼저 큰 범위 조정 → FINE으로 미세 조정
- 단락 트릭: `+`/`−` 단자 굵은 전선으로 단락 → CC 모드 진입 → I 표시 = set 한계 → 노브 돌리면 한계 실시간 변경

---

## 7. 12개 모터 전체 진행 흐름 (2026-05-15 완료)

| # | ID | 조인트 | flux_offset | cal_I | 동작 | 펌웨어 |
|---|---|---|---|---|---|---|
| 1 | 2 | right_hip_roll | +34.72 | 5.0A | ✓ | Windows 빌드 (id=2) |
| 2 | 1 | left_hip_roll | +31.33 | 5.0A | ✓ | 우리 패치 (LOAD_ID=0) |
| 3 | 4 | right_hip_yaw | −32.38 | 5.0A | 47° | Windows (id=4) |
| 4 | 6 | right_hip_pitch | −25.84 | 5.0A | 37.5° | Windows (id=6) |
| 5 | 14 | right_ankle_roll | −82.55 | 3.0A | — | Windows (id=14, ankle 변종) |
| **6** | **12** | **right_ankle_pitch** | **−3.05** | **5.0A** | **94.2°** | **5A 재캘리브 (최정확)** |
| 7 | 8 | right_knee_pitch | −41.05 | 5.0A | 37.8° | 우리 패치 (LOAD_ID=1) |
| 8 | 5 | left_hip_pitch | −23.76 | 5.0A | 30.5° | Windows (id=5) |
| 9 | 3 | left_hip_yaw | −7.46 | 5.0A | 66.8° | Windows (id=3) |
| 10 | 7 | left_knee_pitch | −53.86 | 5.0A | 36.9° | Windows (id=7) |
| 11 | 13 | left_ankle_roll | +0.79 | 3.0A | 40° | Windows (id=13, ankle 변종) |
| 12 | 11 | left_ankle_pitch | −44.15 | 0.5A | 30° | 우리 패치, 저전류 cal |

**전부 캘리브 + flash 저장 + 동작 테스트 완료 (12/12).**

> 💡 **id=12 동작 진폭 94.2°가 가장 큰** 이유 — 5A target에서 정확하게 캘리브돼 flux_offset이 가장 정확. 다른 모터들도 강한 supply로 재캘리브하면 더 큰 진폭 가능할 것.

각 BESC는 자기 flash에 flux_offset + device_id를 영구 저장. 전원 사이클해도 유지.

각 모터당 ID를 다르게 할당하고 싶으면, calibrate_one.py 실행 후 다음 1줄:

```python
# 예: 새 모터(공장 default device_id=1)를 id=3으로 변경
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel python3 -c "
import berkeley_humanoid_lite_lowlevel.recoil as recoil
bus = recoil.Bus('can0', 1000000)
bus._write_parameter_u32(1, recoil.Parameter.DEVICE_ID, 3)   # 1 → 3
bus.store_settings_to_flash(3)                                # 새 id로 flash 저장
bus.stop()
"
```

⚠️ 단, **현재 빌드는 `LOAD_ID_FROM_FLASH=0`** — 부팅 시 device_id가 compile default(=1)로 초기화됨. ID를 모터별로 다르게 유지하려면:
- 옵션 A: `LOAD_ID_FROM_FLASH=1`로 conf.h 수정 후 rebuild
- 옵션 B: 매 모터마다 다른 DEVICE_CAN_ID로 compile해서 별도 .bin 12개 생성

조립 시점에 결정하면 됨 (현재 벤치 단계에서는 무관).

---

## 8. 다음 단계

본 모터 단위 캘리브 완료 후:

1. **12개 모터 모두 완료** (본 문서 절차 반복)
2. **본체 조립** + 종단저항 펌웨어 변형 (중간 모터들은 PC14 RESET으로 빌드, 양 끝 ON)
3. **조인트(영점) 캘리브레이션** → `calibration_2026-05-15.md` 참조
4. **로봇 보행** → DGX-NUC 메인 README 참조
