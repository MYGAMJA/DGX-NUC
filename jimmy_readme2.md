# BHL 모터 캘리브레이션 + 펌웨어 작업 종합 정리

> 기간: 2026-05-15 ~ 2026-05-19 (3일)
> 대상: Berkeley Humanoid Lite (BHL) 다리 12개 모터 + BESC + 펌웨어
> 작업자 인수인계용

---

## 🎯 프로젝트 개요

Berkeley Humanoid Lite (BHL) 다리 12개 모터를 작동시키기 위해:
- 각 모터의 **전기적 캘리브레이션** (flux_offset 학습)
- BESC **펌웨어 빌드/패치/플래시**
- **CAN 데이지체인 결선** 준비

### 사용 하드웨어
| 항목 | 모델/사양 |
|---|---|
| NUC | Mini PC, Ubuntu 22.04, 커널 6.18.19-rt-x64v3-xanmod1 |
| USB-CAN 어댑터 | MKS CANable V2.0 Pro (16d0:117e) × 1 (또는 2개 사용 예정) |
| 모터 컨트롤러 | ST B-G431B-ESC1 × 12 (BESC) |
| 모터 | MAD M6C12 150KV BLDC × 12 |
| 엔코더 | AS5600 자기 엔코더 (각 BESC 내장) |
| 전원공급기 | UNI-T UTP3315TFL (24V, 처음 1A 한계 → 후에 5A로 업그레이드) |

### 펌웨어
- **Recoil Motor Controller** (T-K-233 작성, GitHub) — BESC 위에서 도는 코드
- 위치: `/home/laba/Recoil-Motor-Controller-BESC/`
- 우리가 수정한 부분 (3곳):
  1. `motor_controller.c` `loadConfig` — NaN/0xFFFFFFFF guard 추가
  2. `motor_controller_conf.h` `LOAD_ID_FROM_FLASH=1` 변경 + 다른 LOAD 플래그
  3. `main.c` line 901 — PC14 (CAN 종단저항) ON/OFF 설정

### Python 라이브러리
- **BHL Low-level** (Berkeley)
- 위치: `/home/laba/Berkeley-Humanoid-Lite-Lowlevel/`
- `berkeley_humanoid_lite_lowlevel.recoil` 모듈 — CAN 통신

---

## 📅 일자별 진행

### 2026-05-15 (Day 1)
**전기적 캘리브 1차 완료 (모든 12개 모터)**

겪은 함정 (시간 순):
1. **id=2 BESC가 device_id=1로 가정** → 실제로는 2였음. ID sweep으로 발견.
2. **GND 검정선 빠짐** — CAN 차동신호는 보내지지만 NUC가 디코딩 못 함.
3. **FIRST_TIME_BOOTUP=1 펌웨어 버그** — flash에 zero-init 값이 박혀 device_id=0이 됨. [app.c:108-112](file:///home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Core/Src/app.c#L108-L112). 해결: flash config page (0x0801F800) 2KB를 0xFF로 채워 NaN으로 만듦.
4. **INITIALIZATION_ERROR + while(1)** — loadConfig가 NaN flash 만나 HAL_ERROR 반환 → init 실패 → 무한 루프. 해결: loadConfig 패치 (NaN-guard).
5. **encoder.cpr이 0xFFFFFFFF 로드** — int 필드도 guard 필요. 패치 확장.
6. **AS5600 자석 미부착** → encoder fault.
7. **`MOTOR_CALIBRATION_CURRENT=5A` ↔ supply 1A 한계 충돌** — cal 도중 supply CC 모드 진입 → bus_V 23V→5V 급락 → BESC undervoltage 셧다운.
8. **Motor 12 cal_current를 0.5A로 낮춤** → 1A supply에서 성공.

결과: 12개 모터 모두 전기적 캘리브 + flash 저장 + 동작 테스트 통과.

### 2026-05-16 (Day 2)
**전수 5A 재캘리브** (1차는 1A 한계라 부정확 의심 → supply 5A로 업그레이드 후 재실행)

진행:
- Supply를 5A capacity로 변경 (current COARSE 노브 사용, 단락 트릭)
- 12개 모터 차례로 재캘리브
- 일부 모터: 5A cal 결과가 1차보다 좋음 (motion 진폭 증가)
- 일부 모터: 5A cal 결과가 1차보다 나쁨 → 1차 값 복원 (user 결정으로 motion 큰 값 keep)

새 함정:
9. **id=2 BESC 24V 입력 회로 손상** — supply 0A, 일시적 cold solder 가능성. 재납땜으로 일부 복구.
10. **id=3 BESC 24V 인식 안 됨** — 같은 cold solder 증상. 재납땜 후 복구.
11. **id=1 BESC 납땜 다시 떨어짐** — 보행 시 진동 등으로 재발 가능 위험.
12. **BOOT0 핀 stuck → ROM bootloader** (별도 BESC에서). 옵션 바이트 nSWBOOT0=0 + nBOOT0=0 (`0xFAEFF8AA`)로 해결. bit 26이 nBOOT0 (bit 25 아님).
13. **WRP/RDP write protection** (공장 BESC) — openocd로 해제.

### 2026-05-19 (Day 3, 오늘)
**종단저항 OFF 펌웨어로 11개 BESC 재플래시 + 본체 조립 + 결선 준비**

- 펌웨어 2종 준비:
  - `firmware_termON.bin` (PC14=SET) — 데이지체인 끝점 BESC용
  - `firmware_termOFF.bin` (PC14=RESET) — 중간 BESC 11개용
- 11개 BESC에 termOFF flash 완료 (id=1,2,3,4,5,6,7,8,11,12,13)
- id=14만 termON 유지

토폴로지 결정:
- **2-bus** (왼발/오른발 각각 CAN bus 1개씩 + CANable Pro 2개) — BHL 공식 방식
- 왼발 bus 끝: **id=13 (left_ankle_roll)** → **termON으로 추가 reflash 필요**
- 오른발 bus 끝: id=14 ✓ 이미 termON

---

## ⚙️ 모터 별 최종 상태

| ID | 조인트 | flux_offset (5A cal) | 펌웨어 종단 | 특이사항 |
|---|---|---|---|---|
| 1 | left_hip_roll | −63.62 | termOFF | ⚠️ 납땜 떨어짐 다시 발생, 재납땜 필요 |
| 2 | right_hip_roll | −15.26 | termOFF | 24V 입력 회로 일시 손상 → 재납땜 후 복구 |
| 3 | left_hip_yaw | −7.46 (어제 값 유지) | termOFF | 24V 입력 회로 재납땜 후 복구 |
| 4 | right_hip_yaw | −32.38 (1차 유지) | termOFF | 5A cal 더 나쁜 결과 → 1차 keep |
| 5 | left_hip_pitch | −17.48 | termOFF | OK |
| 6 | right_hip_pitch | −50.97 | termOFF | OK |
| 7 | left_knee_pitch | −47.58 | termOFF | OK |
| 8 | right_knee_pitch | −41.05 (1차 유지) | termOFF | 5A cal 더 나쁜 결과 → 1차 keep |
| 11 | left_ankle_pitch | −44.16 | termOFF | 액추에이터 조립 상태에서 cal됨 |
| 12 | right_ankle_pitch | −3.05 | termOFF | 가장 정확한 cal (94° motion) |
| 13 | left_ankle_roll | −68.32 | **termOFF → termON 변경 필요** (왼발 bus 끝) | |
| 14 | right_ankle_roll | +14.22 | **termON** ✓ (오른발 bus 끝) | cal 중 supply 잠시 다운했지만 동작 OK |

---

## 🛠️ 자주 쓰는 명령

### NUC 초기 셋업 (재부팅 시 또는 USB 재연결 시)
```bash
# CANable이 어느 ttyACM에 있나
for dev in /dev/ttyACM*; do
  udevadm info -q property -n $dev | grep ID_MODEL=
done

# 1-bus 셋업
sudo killall slcand 2>/dev/null; sleep 1
sudo slcand -o -c -s8 /dev/ttyACMN can0   # N = CANable의 번호
sudo ip link set up can0

# 2-bus 셋업 (CANable 2개)
sudo killall slcand 2>/dev/null; sleep 1
sudo slcand -o -c -s8 /dev/ttyACMA can0   # 첫번째 CANable (왼발)
sudo slcand -o -c -s8 /dev/ttyACMB can1   # 두번째 CANable (오른발)
sudo ip link set up can0
sudo ip link set up can1
```

### 모터 응답 확인 (sweep)
```bash
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel python3 -c "
import berkeley_humanoid_lite_lowlevel.recoil as recoil
bus = recoil.Bus('can0', 1000000)
for n in range(1, 15):
    if bus.ping(n):
        flux = bus.read_encoder_flux_offset(n)
        cal_I = bus.read_motor_calibration_current(n)
        print(f'id={n}: online  flux={flux:+.2f}  cal_I={cal_I}A')
bus.stop()
"
```

### 단일 모터 자동 캘리브 (자동 헬퍼)
```bash
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel \
  python3 /home/laba/DGX-NUC/BHL/calibrate_one.py
```

### 5A 정밀 캘리브 (수동, bus_V 모니터링 포함)
인라인 절차 — [BHL/motor_calibration_2026-05-15.md](BHL/motor_calibration_2026-05-15.md) 끝 부분 참조.

### 펌웨어 빌드 + flash
```bash
# 1. 펌웨어 소스 수정 (예: PC14 SET ↔ RESET)
# /home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Core/Src/main.c
#   line 901 의 GPIO_PIN_SET (종단 ON) ↔ GPIO_PIN_RESET (종단 OFF)

# 2. 빌드
cd /home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Debug
make clean && make all
arm-none-eabi-objcopy -O binary Recoil-Motor-Controller-B-G431B-ESC1.elf Recoil-Motor-Controller-B-G431B-ESC1.bin

# 3. flash (BESC USB-C 연결 후)
st-flash --reset write Recoil-Motor-Controller-B-G431B-ESC1.bin 0x8000000

# 미리 빌드된 2종 .bin
ls /home/laba/DGX-NUC/BHL/firmware_term*.bin
# firmware_termON.bin   (PC14=SET, 끝점 BESC용)
# firmware_termOFF.bin  (PC14=RESET, 중간 BESC용)
```

### BESC 상태 진단 (CAN 응답 안 할 때)
```bash
# ST-Link 인식
lsusb | grep 0483

# RAM에서 controller 구조체 읽기 (device_id, error, mode 등)
st-flash read /tmp/c.bin 0x20000228 32 2>&1 | tail -1
python3 -c "
import struct
raw = open('/tmp/c.bin','rb').read()
d, f, w, ff, m, e = struct.unpack('<6I', raw[:24])
print(f'device_id={d}, fw=0x{f:08x}, mode={m}, error=0x{e:04x}')
"

# FDCAN 페리프럴 상태 (CCCR INIT bit가 0이어야 정상)
st-flash read /tmp/fd.bin 0x40006400 0x30 2>&1 | tail -1
python3 -c "
import struct
d = open('/tmp/fd.bin','rb').read()
cccr = struct.unpack('<I', d[0x1C:0x20])[0]
nbtp = struct.unpack('<I', d[0x20:0x24])[0]
print(f'CCCR=0x{cccr:08x} INIT={cccr&1}')
print(f'NBTP=0x{nbtp:08x}')
"

# RCC 상태 (HSE 동작 여부)
st-flash read /tmp/rcc.bin 0x40021000 4 2>&1 | tail -1
python3 -c "
import struct
cr = struct.unpack('<I', open('/tmp/rcc.bin','rb').read()[:4])[0]
print(f'HSE_RDY={(cr>>17)&1}, PLL_RDY={(cr>>25)&1}')
"
```

### BOOT0 stuck → ROM bootloader 해결 (옵션 바이트)
```bash
openocd -f interface/stlink.cfg -f target/stm32g4x.cfg -c "
init; reset halt
mww 0x40022008 0x45670123; mww 0x40022008 0xCDEF89AB
mww 0x4002200C 0x08192A3B; mww 0x4002200C 0x4C5D6E7F
sleep 100
mww 0x40022020 0xFAEFF8AA  # nSWBOOT0=0 (bit24), nBOOT0=0 (bit26)
mww 0x40022014 0x00020000  # OPTSTRT
sleep 500
mww 0x40022014 0x08000000  # OBL_LAUNCH
sleep 1500
exit"
```

### Flash WRP/RDP 해제 (공장 BESC)
```bash
openocd -f interface/stlink.cfg -f target/stm32g4x.cfg \
  -c "init; reset halt; stm32l4x unlock 0; exit"
# 이후 st-flash erase + write 가능
```

### USB MSC 드래그-앤-드롭 flash (대안)
```bash
# BESC USB-C 꽂으면 자동 마운트됨 (/media/laba/DIS_G431CB)
cp /home/laba/DGX-NUC/BHL/firmware_termOFF.bin /media/laba/DIS_G431CB/
# 자동 flash됨. DETAILS.TXT만 남음 = 성공. FAIL.TXT 생기면 실패.
```

---

## 📂 파일 위치 정리

```
/home/laba/DGX-NUC/                          ← 메인 작업 디렉토리
├── jimmy_readme2.md                         ← 이 파일
├── README.md                                ← DGX-NUC 전체 프로젝트 설명
└── BHL/
    ├── SUMMARY_for_next_user.md             ← 어제 작성한 인수인계 (간소판)
    ├── WORKFLOW.md                          ← 종합 워크플로 + 17개 함정 모음
    ├── motor_calibration_2026-05-15.md      ← 모터 캘리브 실행 로그
    ├── calibration_2026-05-15.md            ← 조인트 캘리브 절차
    ├── BHL_CAN_Debug_Handoff.md             ← CAN 디버깅 기록 + 해결 보고
    ├── calibrate_one.py                     ← 모터 1개 자동 캘리브 헬퍼
    ├── firmware_termON.bin                  ← PC14=SET 펌웨어 (체인 끝)
    ├── firmware_termOFF.bin                 ← PC14=RESET 펌웨어 (중간 BESC)
    └── motor_id*_before/after_cal.json      ← 모터별 cal 전/후 config 백업

/home/laba/Recoil-Motor-Controller-BESC/    ← BESC 펌웨어 소스 (수정됨)
└── Recoil-Motor-Controller-B-G431B-ESC1/
    ├── Core/Src/main.c                      ← PC14 종단저항 설정 (line 901)
    ├── Core/Src/motor_controller.c          ← loadConfig 패치 (NaN/0xFFFFFFFF guard)
    ├── Core/Inc/motor_controller_conf.h    ← LOAD_*_FROM_FLASH 플래그
    └── Debug/Recoil-Motor-Controller-B-G431B-ESC1.bin  ← 빌드 결과물

/home/laba/Berkeley-Humanoid-Lite-Lowlevel/ ← BHL Python 라이브러리 (변경 없음)
└── scripts/motor/
    ├── calibrate_electrical_offset.py      ← 공식 캘리브 (참조)
    ├── ping.py
    └── ...
```

---

## 🔥 17개+ 함정 모음 (시간 순)

전체 내용은 [BHL/WORKFLOW.md](BHL/WORKFLOW.md) 의 함정 1~17번 참조. 요약:

| # | 함정 | 해결 |
|---|---|---|
| 1 | device_id 가정 오류 (1로 알았는데 2였음) | 첫 단계는 항상 ID sweep |
| 2 | CAN GND(검정) 선 빠짐 | 결선 시 GND 항상 확인 |
| 3 | FIRST_TIME_BOOTUP=1 펌웨어 버그 | flash erase + loadConfig 패치 |
| 4 | flash NaN → INITIALIZATION_ERROR | loadConfig NaN/0xFFFFFFFF guard |
| 5 | encoder cpr=−1 → ENCODER_FAULT | int 필드도 0xFFFFFFFF guard |
| 6 | AS5600 자석 미부착 | 자석 부착, 직경 자화 확인 |
| 7 | 패치 점진 추가로 4번 reflash | 한 번에 종합 점검 |
| 8 | `make` 가 `clean`만 돔 | `make all` 명시 |
| 9 | objects.list 없음 | `find . -name "*.o" > objects.list` |
| 10 | USB-C만으론 CAN transceiver 안 켜짐 | 24V ON 필수 |
| 11 | 공장 BESC WRP/RDP write protection | openocd unlock |
| 12 | cal_current ↔ supply 한계 충돌 | bus_V 모니터, 강한 supply or 낮은 cal_I |
| 13 | move_actuator.py 사인파 추종 실패 | target = start_pos + sin·0.5 |
| 14 | BOOT0 핀 stuck → ROM bootloader | 옵션 바이트 nBOOT0=0 (bit 26!) |
| 15 | 공장 BESC USB-MSC 드래그-앤-드롭 가능 | `cp *.bin /media/laba/DIS_G431CB/` |
| 16 | 1A supply로 5A cal target 불가 | UTP3315TFL COARSE 노브로 한계 올림 |
| 17 | UTP3315TFL FINE 노브 ±0.5A만 | COARSE로 main 조정 |

---

## 🔄 남은 작업 (다음 사용자가 할 것)

### 1단계: id=13 termON으로 추가 reflash (2-bus 토폴로지면 필수)
```bash
# id=13 BESC USB-C 연결 후
st-flash --reset write /home/laba/DGX-NUC/BHL/firmware_termON.bin 0x8000000
```

### 2단계: 본체 조립 + 데이지체인 결선

**토폴로지 결정**: 사용자가 **2-bus (왼발/오른발 각각 CAN)** 결정.

#### CAN 결선
```
[CANable Pro #1] ─── id=1 ─── id=3 ─── id=5 ─── id=7 ─── id=11 ─── id=13 (termON)
   (can0 = 왼발)                                                       ↑ 끝점

[CANable Pro #2] ─── id=2 ─── id=4 ─── id=6 ─── id=8 ─── id=12 ─── id=14 (termON)
   (can1 = 오른발)                                                     ↑ 끝점
```

각 BESC의 H/L/G 패드는 분기점 — **두 wire가 같은 패드에 납땜** (이전 BESC에서 들어오는 + 다음 BESC로 나가는). 끝점 BESC만 wire 1개.

#### 24V 결선
- 모든 BESC + 단자 / − 단자에 병렬 공급
- 메인 wire 굵게 (18 AWG 이상, peak 60A 가능)
- 사용자 supply UTP3315TFL 3.3A → 보행 시 전류 부족 가능 → 강한 supply 권장

#### 외관 + 접근 구멍
- 사용자 결정: 외관에 접근 구멍 만들어 조인트 캘리브 시 손으로 limit 잡을 수 있게
- 모든 결선 + 검증 통과 후 외관 닫기

### 3단계: CAN sweep 검증
```bash
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel python3 << 'EOF'
import berkeley_humanoid_lite_lowlevel.recoil as recoil
print("=== can0 (왼발) ===")
bus = recoil.Bus('can0', 1000000)
for n in [1, 3, 5, 7, 11, 13]:
    if bus.ping(n): print(f"  id={n}: online")
bus.stop()
print("=== can1 (오른발) ===")
bus = recoil.Bus('can1', 1000000)
for n in [2, 4, 6, 8, 12, 14]:
    if bus.ping(n): print(f"  id={n}: online")
bus.stop()
EOF
```
각 bus에서 6개 응답하면 OK.

### 4단계: 조인트 캘리브 (`calibrate_joints.py`)
```bash
cd /home/laba/Berkeley-Humanoid-Lite-Lowlevel
python3 source/berkeley_humanoid_lite_lowlevel/scripts/calibrate_joints.py
```
- 실행 중 12개 관절 모두 limit 자세로 누르고 있음 (접근 구멍 활용)
- 조이스틱 X 버튼 또는 좌/우 썸스틱 클릭으로 종료
- `calibration.yaml` 파일 생성됨 (NUC 디스크 영구 저장)
- 매 power cycle 시 재실행 필요 (관절 0점 위치 재학습)
- 절차 상세: [BHL/calibration_2026-05-15.md](BHL/calibration_2026-05-15.md)

### 5단계: 보행 테스트
- DGX 측 학습된 정책 사용
- 또는 simple oscillation으로 다리 움직임 검증

---

## ⚠️ 주의 사항 (다음 사용자가 알아야 할 것)

### 1. id=1 BESC 납땜 문제 재발 가능
- 어제 한 번 떨어짐 → 재납땜
- 보행 진동으로 다시 떨어질 수 있음
- 증상: 24V 연결해도 supply에 0.00A
- 처치: BESC + / − 패드 재납땜

### 2. id=2 BESC 24V 입력 회로 손상 이력
- 5월 16일 일시적 0A → 사용자가 수리 의도
- 만약 사용자가 못 고치면 id=3을 id=2로 변경하는 방법 (SDO write):
  ```python
  bus._write_parameter_u32(3, recoil.Parameter.DEVICE_ID, 2)
  bus.store_settings_to_flash(2)
  ```

### 3. cal_current 보존
- 어제 작업 중 일부 BESC의 cal_current=0.5A로 변경됐을 수 있음
- 다음 cal 시 5.0A로 다시 설정해야 정확
- ankle 모터들은 원래 3.0A default

### 4. 조인트 캘리브는 매번 필요
- AS5600은 motor 축에만 있고 출력축에는 없음
- power cycle 시 관절 0점 재학습 필요
- 외관 접근 구멍으로 사용자 직접 limit 잡기 가능

### 5. 펌웨어 변경 시 device_id 보존
- 우리 패치 펌웨어는 `LOAD_ID_FROM_FLASH=1` + guard
- BESC flash에 device_id 저장돼 있으면 reflash 후에도 보존됨
- 단 flash config page (`0x0801F800`)를 erase하면 default(1)로 reset됨

---

## ⭐ 한 줄 요약

**12개 모터 다 전기적 캘리브 + flash 영구화 + 11개 termOFF / 1개 termON 펌웨어 분배 완료. 남은 일: id=13 termON 추가 reflash → 데이지체인 결선 → 조인트 캘리브 → 보행 테스트.**
