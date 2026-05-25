# Berkeley Humanoid Lite — CAN 통신 디버깅 핸드오프

## ✅ 2026-05-15 해결됨 — 진짜 원인 (총 3건)

### (1) 첫 BESC: device_id 가정 오류
- BESC가 `device_id=1`이 아니라 **`2`**로 플래싱돼 있었음.
- 검증: `cansend can0 202#CA` → `0x182` 에서 `CA 00 00 00 6B 26 00 08` 응답 (latency ~300µs).
- `ping.py -c can0 -i 2` → **`Motor is online`**

### (2) 새 BESC (15:00 이후): GND 선 (검정) 빠져있었음
- CAN 3-wire 중 검정선이 BESC 측에서 헐겁게 빠져 있었음.
- 차동신호는 보내져도 NUC↔BESC 전위 기준이 맞지 않아 응답 디코딩 실패.

### (3) 새 BESC: flash config 페이지 오염 (펌웨어 버그)
- RAM dump 결과 `device_id=0` (1 아님), `mode=DAMPING`, `error=0x2000` 등 비정상 상태.
- 원인: 이전 누군가 `FIRST_TIME_BOOTUP=1`로 빌드/flash → [app.c:108-112](file:///home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Core/Src/app.c#L108-L112)의 **펌웨어 버그**가 트리거:
  ```c
  #if FIRST_TIME_BOOTUP
      APP_initFlashOption();
      MotorController_storeConfig(&controller);  // ← zero-init controller가 그대로 flash에 저장됨!
      while (1) {}
  #endif
  ```
  `MotorController_init` 호출 **전에** storeConfig가 불려서 `device_id=0` 등 zero값들이 flash(`0x0801F800`)에 박힘.
- `LOAD_ID_FROM_FLASH=1`이라 새 펌웨어 부팅 시 flash의 0값이 로드됨 → 모든 CAN ID 매칭 실패.
- **해결**: ST-Link로 flash config page (`0x0801F800`, 2KB)를 0xFF로 덮어씀 → flux_offset이 NaN 으로 읽힘 → [motor_controller.c:222](file:///home/laba/Recoil-Motor-Controller-BESC/Recoil-Motor-Controller-B-G431B-ESC1/Core/Src/motor_controller.c#L222)의 `if(isnan(...)) return HAL_ERROR` 로 loadConfig 조기 종료 → controller가 compile-time defaults(device_id=1) 유지.

  ```bash
  python3 -c "open('/tmp/erase.bin','wb').write(b'\xff'*2048)"
  st-flash write /tmp/erase.bin 0x0801F800
  st-flash reset
  ```

- 핸드오프의 "Nothing received" (line 94 부근)은 (1) 때문에 응답이 ID=1 응답 기대와 안 맞아서 발생.
- 핸드오프의 "Nothing received" (line 94 부근)은 두 가지가 겹친 결과:
  1. python-can 기본값 `receive_own_messages=False` → echo 자체가 안 들어옴
  2. recoil ping은 함수ID=3 (TRANSMIT_PDO_1) 필터를 거는데, echo는 함수ID=4 (RECEIVE_PDO_1)라 어차피 걸러짐
  3. 그리고 실제 motor 응답은 ID=1 가정 때문에 영영 나오지 않음
- 결국 가설 1/2/3/4 어느 것도 진짜 원인이 아니었음 (펌웨어 OK, slcan OK, python-can OK, 종단저항 OK).
- `DEVICE_CAN_ID = 1` (line 64) 항목은 펌웨어 소스 기대값이고, 실제 플래싱된 칩의 ID와 다를 수 있음. 새 BESC 받을 때마다 `ping` sweep으로 확인 권장.

> ⚠️ **새 BESC 디버깅 시 첫 단계**: 단일 ID(`-i 1`)로 ping해서 offline 나오면 곧바로
> ```bash
> for N in $(seq 1 14); do
>   hexid=$(printf "%03X" $((0x200 | N)))
>   cansend can0 ${hexid}#CA
> done
> ```
> 와 `candump can0` 동시 가동 → 응답하는 ID 찾기. 5초 안에 결판남.

(아래는 디버깅 당시 핸드오프 원본 — 기록 보존용)

---

## 🎯 현재 막힌 문제 (원래 기술 — 위 해결 보고로 대체됨)

**BESC(B-G431B-ESC1) 모터 컨트롤러 1개와 CAN 통신이 안 됨.**
- ping 결과: `Motor is offline`
- 모든 일반적인 원인을 다 점검했는데도 해결 안 됨
- 진짜 원인을 찾고 효율적으로 디버깅 재개 필요

---

## 💻 환경

### 하드웨어
- **NUC**: Mini PC, Ubuntu 22.04.5 LTS, 커널 `6.18.19-rt-x64v3-xanmod1` (XanMod RT)
- **USB-CAN 어댑터**: MKS CANable V2.0 Pro
  - USB ID: `16d0:117e`
  - 펌웨어: `CANable2 b158aa7 github.com/normaldotcom/canable2.git` (slcan 기반 fork)
  - `/dev/ttyACM1`로 인식
  - 120R 점퍼: **ON** (확인됨)
- **모터 컨트롤러**: ST 정품 **B-G431B-ESC1** (라벨로 확인)
- **모터**: MAD M6C12 (150KV) BLDC, AS5600 자기 엔코더
- **전원공급기**: UNI-T UTP3315TFL, 24V / 1A 제한 (단락 트릭으로 설정)

### 소프트웨어
- BHL 저장소: `~/Berkeley-Humanoid-Lite-Lowlevel` (HybridRobotics)
- 펌웨어 저장소: `~/Recoil-Motor-Controller-BESC` (T-K-233)
- Windows PC에 STM32CubeIDE 설치 (Recoil 프로젝트 import되어 있음)
- Python 의존성 설치 완료 (`pip install -r requirements.txt`)
- `PYTHONPATH=~/Berkeley-Humanoid-Lite-Lowlevel` 설정 사용

### 사용 사례
- BHL 본체에서 **하체 12개 모터만** 사용 (CAN2 버스의 ID 1~8 + 11~14)
- 팔은 SO-ARM 대체
- 결국 CAN 버스 1개만 운영 (12개 모터 데이지 체인)
- 현재는 **단일 모터 벤치 테스트** 단계

---

## ✅ 지금까지 확인된 것 (정상)

1. **USB / CAN 인터페이스**
   - CANable Pro USB 인식됨 (`lsusb`)
   - `gs_usb` 모듈은 안 맞아서 → `slcan` 방식으로 우회
   - `slcand -o -c -s8 /dev/ttyACM1 can0` + `ip link set up can0`
   - `ip -details link show can0` → `state UP`, `ERROR-ACTIVE`

2. **송신 (NUC → CAN 버스)**
   - `cansend can0 001#FF` 정상 동작
   - candump에 echo 프레임 보임

3. **배선 (BHL 공식 컨벤션과 일치)**
   - 🟡 노랑 = CAN-H
   - 🟢 녹색 = CAN-L
   - ⚫ 검정 = GND
   
   - BESC의 "CAN" 라벨 옆 J1 패드에 정확히 납땜됨
   - CANable Pro 터미널 블록: L=녹색, H=노랑, G=검정

4. **전원**
   - 24V 인가, 전류 0.01A (정상 idle)
   - BESC LED 깜빡임 정상

5. **펌웨어 설정**
   - `DEVICE_CAN_ID = 1`
   - `FIRST_TIME_BOOTUP = 0` (정상 동작 모드)
   - `HAL_FDCAN_Start()` 호출됨
   - 종단저항(PC14)을 `GPIO_PIN_SET`으로 수정 후 재플래싱 성공
     - STM32CubeProgrammer 로그: "Download verified successfully"

6. **테스트**
   - 펌웨어 정상 플래싱 후에도 ping은 여전히 offline
   - candump도 자발적 프레임 없음

---

## ❌ 막힌 지점

### 핵심 증상
- 펌웨어 종단저항 ON으로 수정 + 재플래싱 성공
- 그래도 `python3 ./scripts/motor/ping.py -c can0 -i 1` → `Motor is offline`
- candump 켜놓고 30초 봐도 BESC 자발적 프레임 0건

### 이전에 본 적 있는 단서
- 어느 순간 `candump can0` 화면에 `can0  201   [1]  CA`가 보였음
- **나중에 검증 결과 echo였음** (BESC 전원 OFF 상태에서 `cansend can0 201#CA` 해도 동일 프레임 보임)
- 즉, BESC는 한 번도 진짜 응답을 보낸 적 없음

### Python-can 직접 수신 테스트
```python
import can
bus = can.interface.Bus(channel='can0', bustype='socketcan', bitrate=1000000)
for _ in range(50): msg = bus.recv(0.1); ...
```
→ 5초 동안 **Nothing received** (echo도 안 잡힘)

---

## 🔍 ping.py 동작 분석

```python
# scripts/motor/ping.py
import berkeley_humanoid_lite_lowlevel.recoil as recoil
bus = recoil.Bus(channel=args.channel, bitrate=1000000)
status = bus.ping(args.id)
```

```python
# berkeley_humanoid_lite_lowlevel/recoil/core.py:271
def ping(self, device_id: int, timeout=0.1) -> bool:
    self.transmit(CANFrame(device_id, Function.RECEIVE_PDO_1, size=1, data=b"\xCA"))
    rx_frame = self.receive(filter_device_id=device_id, 
                            filter_function=Function.TRANSMIT_PDO_1, 
                            timeout=timeout)
    if not rx_frame: return False
    rx_data = self.unpack("<BBBBBBBB", rx_frame.data)[0]
    return rx_data == 0xCA
```

### CAN ID 매핑
- `DEVICE_ID_MSK = 0x7F`
- `FUNC_ID_POS = 7`
- `RECEIVE_PDO_1 = 0b0100` (4) — ping 요청 시 사용
- `TRANSMIT_PDO_1 = 0b0011` (3) — 응답 기대

### 송신 ID 계산
- ping 요청: `(4 << 7) | 1 = 0x201` ✓ (candump 송신 echo로 확인)
- 응답 기대 ID: `(3 << 7) | 1 = 0x181`

### Bus 초기화
```python
self.__bus = can.interface.Bus(
    interface="socketcan", channel=self.channel, bitrate=self.bitrate
)
```
- `bitrate=1000000` 명시 — slcan에서는 무시될 수 있음
- timeout 1.0초로 늘려도 동일 결과 (offline)

---

## 🤔 의심되는 진짜 원인 (다음 디버깅 방향)

### 가설 1: BESC가 응답 자체를 안 보냄
- 펌웨어가 진짜로 새 코드로 빌드/플래싱됐는지 의심
- Clean Build 후 재플래싱 필요할 수도
- 또는 펌웨어 로직 자체에 ping 응답 부재

### 가설 2: 응답을 보내지만 다른 ID라 ping.py가 못 받음
- 펌웨어 응답이 `TRANSMIT_PDO_1` (func_id=3, ID=0x181)이 아닌 다른 ID일 수도
- 결정적 검증: candump 켜고 `cansend can0 201#CA` → BESC 응답으로 `0x201` echo 외 다른 ID 프레임이 나타나는지 확인

### 가설 3: slcan + python-can 호환성 문제
- python-can의 socketcan 백엔드가 slcan 위에서 수신 누락
- CANable 펌웨어를 candleLight으로 재플래싱하면 `gs_usb`로 자동 잡혀서 해결 가능
- → CANable Pro 자체 펌웨어를 https://canable.io/updater/canable2.html 에서 candleLight으로 재플래싱

### 가설 4: BESC 측 종단저항 펌웨어 수정이 빌드/플래싱 시 누락
- 코드 수정은 확인됨 (sed로 RESET → SET)
- 빌드 시 변경사항이 .elf에 반영됐는지 불확실
- 해결: STM32CubeIDE에서 `Project → Clean` 후 다시 Run

---

## 📋 새 대화에서 우선 시도할 것

### 1️⃣ 결정적 검증: BESC 진짜 응답 확인
```bash
# 터미널 1
candump can0
# 터미널 2
cansend can0 201#CA
```
- candump에 `201 [1] CA`(echo) **외에 다른 ID** 프레임이 보이면 → BESC 응답 있음 (가설 2)
- echo만 보이면 → BESC가 송신 자체를 안 함 (가설 1, 4)

### 2️⃣ python-can으로 임의 프레임 수신 시도
```python
import can
bus = can.interface.Bus(channel='can0', bustype='socketcan')
print(bus.recv(5.0))  # 5초 동안 어떤 프레임이든
```
- None이 나오면 → slcan 백엔드 자체 수신 문제 (가설 3)
- 뭔가 잡히면 → recoil 라이브러리 필터링 문제

### 3️⃣ CANable Pro 펌웨어를 candleLight으로 재플래싱
- BOOT 점퍼 BOOT 위치 → USB 재연결 → Chrome으로 https://canable.io/updater/canable2.html → candleLight 펌웨어 선택
- 그러면 `gs_usb`로 자동 잡혀서 `can0` 생성, `bitrate 1000000` 같은 옵션도 제대로 동작

### 4️⃣ BESC 펌웨어 Clean Build 후 재플래싱
- STM32CubeIDE에서 `Project → Clean...` → Clean → Run
- 변경사항이 진짜 .elf에 들어갔는지 확실히

---

## 🛠️ 자주 쓰는 명령 참고

### CAN 인터페이스 시작 (slcan 방식)
```bash
sudo ip link set can0 down 2>/dev/null
sudo killall slcand 2>/dev/null
sleep 1
sudo slcand -o -c -s8 /dev/ttyACM1 can0
sudo ip link set up can0
ip -details link show can0
```

### Python 환경
```bash
cd ~/Berkeley-Humanoid-Lite-Lowlevel
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 테스트
```bash
candump can0                                  # 모니터링
cansend can0 201#CA                           # 직접 송신
python3 ./scripts/motor/ping.py -c can0 -i 1  # ping
```

---

## 🎯 최종 목표

1. 단일 모터에 대해 `Motor is online` 받기
2. 전기적 오프셋 캘리브레이션 (`calibrate_electrical_offset.py`)
3. 사인파 동작 테스트 (`move_actuator.py`)
4. 동일 절차로 나머지 11개 모터 작업 (총 12개)
5. 본체 조립 후: 마지막 1개만 종단 ON, 나머지 11개는 OFF로 펌웨어 재플래싱

---

## 📝 메모

- **종단저항 12개 분배**: 데이지 체인 양 끝(CANable + 마지막 모터)에만 ON, 중간 모터들은 OFF. 단일 모터 벤치 테스트에서는 그 모터가 끝이라 ON 필요.
- **앞으로 펌웨어 변경 효율을 위해** `motor_controller_conf.h`에 `CAN_TERMINATION_ENABLE` 매크로를 추가하고 `main.c`를 `#if` 처리하는 것이 권장됨 (12개 모터 작업 직전에 정리)
- ping.py의 인자: `-c can0 -i 1` (channel, id)
