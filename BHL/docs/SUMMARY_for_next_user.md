# BHL 모터 작업 정리 (인수인계용)

> **읽는 사람**: 이전 작업자 아닌 새 사용자
> **목적**: 지금까지 뭐 했고, 지금 어떤 상태이고, 다음에 뭐 해야 하는지

---

## 🎯 우리가 뭘 하고 있는가

Berkeley Humanoid Lite (BHL) 로봇의 다리 12개 모터를 작동시키기 위한 **모터 캘리브레이션** 작업.

각 모터는:
- BLDC 모터 (MAD M6C12)
- BESC (모터 컨트롤러 보드, ST B-G431B-ESC1)
- AS5600 자기 엔코더 (회전 위치 측정용 자석)

캘리브레이션 = 모터의 전기적 정렬값(`flux_offset`)을 학습시키는 작업. 안 하면 모터 못 돌림.

---

## ✅ 현재 상태 — **12개 모터 전부 캘리브 완료**

각 모터의 캘리브 값과 동작 진폭 (sin파 명령 시 모터가 실제로 움직인 각도):

| 모터 # | ID | 조인트 위치 | flux_offset | 동작 | 비고 |
|---|---|---|---|---|---|
| 1 | 2 | 오른쪽 hip_roll | −15.26 | 63° | OK |
| 2 | 1 | 왼쪽 hip_roll | −63.62 | 118.6° ✨ | ⚠️ 납땜 다시 떨어짐 — 재납땜 필요 |
| 3 | 4 | 오른쪽 hip_yaw | −32.38 | 58° | OK |
| 4 | 6 | 오른쪽 hip_pitch | −50.97 | 64° | OK |
| 5 | 14 | 오른쪽 ankle_roll | +14.22 | 81° | cal 중 supply 잠시 다운, 동작은 OK |
| 6 | 12 | 오른쪽 ankle_pitch | −3.05 | 94° ✨ | OK (가장 정확한 cal) |
| 7 | 8 | 오른쪽 knee_pitch | −41.05 | 64° | OK |
| 8 | 5 | 왼쪽 hip_pitch | −17.48 | 62° | OK |
| 9 | 3 | 왼쪽 hip_yaw | −7.46 | ~40° | OK |
| 10 | 7 | 왼쪽 knee_pitch | −47.58 | 59° | OK |
| 11 | 11 | 왼쪽 ankle_pitch | −44.16 | 36° | 액추에이터 안에 조립된 상태 |
| 12 | 13 | 왼쪽 ankle_roll | −68.32 | 81° | OK |

**값은 모두 각 BESC의 flash 메모리에 저장됨** → 전원 꺼도 유지.

---

## ⚠️ 주의 사항 (다음 사용자가 알아야 할 것)

### 1. id=1 BESC 납땜 문제
- **증상**: ST-Link USB 연결하면 BESC LED 켜짐 (MCU 살아있음). 24V 연결하면 LED 안 켜지고 supply에 0.00A.
- **원인**: 24V 입력 솔더 조인트가 떨어짐 (cold solder).
- **해결**: BESC 보드의 + 와 − 패드를 **납인두로 재납땜**.
  - 인두를 패드+wire 접합부에 3~5초 짧게 댐 (wire 절연 안 녹게)
  - 새 솔더 살짝 흘려넣기
  - 인두 떼고 2초 wire 고정 (cold solder 재발 방지)
- **재납땜 성공 확인**: 24V 다시 연결했을 때 supply에 0.01A 흐름

### 2. id=3 BESC도 같은 증상 → 이미 재납땜 완료
- 오늘 재납땜으로 복구됨. flush=−7.46 캘리브 저장됨.
- 만약 또 떨어지면 같은 방법으로 재납땜.

### 3. id=14 (액추에이터 조립 상태) cal 중 supply 다운
- 액추에이터 기어 부하로 인해 5A cal 도중 supply 전압이 잠시 4V로 떨어짐
- cal 값(+14.22)는 다소 의심스럽지만 동작 테스트 81°로 잘 됨
- 보행 테스트에서 ankle_roll 이상 보이면 → 분해 후 재캘리브

### 4. 캘리브 값 = "정확한 단일 값" 아님
- 같은 모터 여러 번 cal해도 값이 약간씩 다름 (cal 시퀀스 시작 위치에 따라 변동)
- "동작 테스트 진폭이 클수록 더 정확한 cal" → 진폭 큰 값을 flash에 저장하는 방식 사용

---

## 🔄 다음에 할 일

### 1단계: 본체 조립
12개 모터 + BESC를 BHL 다리 구조에 mounting.

### 2단계: CAN 데이지체인 결선
- 12개 BESC를 CAN 버스 1개에 직렬 연결
- 노란선(H), 녹색선(L), **검정선(GND)** — 3선 모두 필수
- 검정선(GND) 빠지면 통신 안 됨! (오늘 작업 중 발견된 흔한 함정)
- 양 끝(NUC + 마지막 BESC)만 종단저항 ON, 중간 11개는 OFF로 펌웨어 다시 굽기

### 3단계: 조인트 캘리브레이션 (`calibrate_joints.py`)
- 각 관절의 기계적 영점 (zero position) 학습
- 절차: [`calibration_2026-05-15.md`](calibration_2026-05-15.md) 참조
- 매 전원 사이클마다 실행 필요 (휘발성)

### 4단계: 보행 테스트
- 실제 보행 명령으로 로봇 동작 확인
- 문제 모터 있으면 → 분해 + 해당 모터만 재캘리브

---

## 🛠️ 자주 쓰는 명령 (참고)

### 환경 셋업 (NUC 재부팅 시 매번 필요)

```bash
# CANable USB가 어느 ttyACM에 있는지 확인
for dev in /dev/ttyACM*; do
  udevadm info -q property -n $dev | grep ID_MODEL=
done
# → ID_MODEL=CANable2_... 라고 적힌 게 CANable. 그 ttyACMN 사용.

# slcand로 CAN 인터페이스 띄움 (X는 CANable의 번호로 교체)
sudo killall slcand 2>/dev/null; sleep 1
sudo slcand -o -c -s8 /dev/ttyACMX can0
sudo ip link set up can0
```

### 모터 상태 확인

```bash
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel python3 -c "
import berkeley_humanoid_lite_lowlevel.recoil as recoil
bus = recoil.Bus('can0', 1000000)
for n in range(1, 15):
    if bus.ping(n):
        flux = bus.read_encoder_flux_offset(n)
        cal_I = bus.read_motor_calibration_current(n)
        bv = bus._read_parameter_f32(n, recoil.Parameter.POWERSTAGE_BUS_VOLTAGE_MEASURED)
        print(f'id={n}: online  flux={flux:+.2f}  cal_I={cal_I}A  bus_V={bv:.2f}')
bus.stop()
"
```

응답하는 BESC가 정상. 응답 없으면:
- 24V supply ON 확인 (supply에 0.01A 흐르는지)
- CAN 3선 (특히 GND 검정선) 확인
- USB-C로 ST-Link 통해 BESC 살아있는지 확인

### 단일 모터 재캘리브

이미 캘리브된 모터를 다시 cal하고 싶으면:

```bash
# 자동 헬퍼 스크립트
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel \
  python3 /home/laba/DGX-NUC/BHL/calibrate_one.py
```

또는 5A 정밀 cal (수동 절차)은 [`motor_calibration_2026-05-15.md`](motor_calibration_2026-05-15.md) 끝 부분 참조.

---

## 📂 관련 파일 위치

| 파일 | 내용 |
|---|---|
| `BHL/SUMMARY_for_next_user.md` | **이 파일** (인수인계) |
| `BHL/WORKFLOW.md` | 종합 작업 가이드 + 17개 함정/교훈 |
| `BHL/motor_calibration_2026-05-15.md` | 모터별 cal 실행 기록 (어제+오늘) |
| `BHL/calibration_2026-05-15.md` | 조인트(영점) 캘리브 절차 |
| `BHL/BHL_CAN_Debug_Handoff.md` | CAN 통신 디버깅 기록 |
| `BHL/calibrate_one.py` | 모터 1개 자동 캘리브 스크립트 |

펌웨어:
- `/home/laba/Recoil-Motor-Controller-BESC/` — Recoil 펌웨어 소스
- `Debug/Recoil-Motor-Controller-B-G431B-ESC1.bin` — 빌드된 펌웨어 binary

BHL 코드:
- `/home/laba/Berkeley-Humanoid-Lite-Lowlevel/` — 메인 BHL 저장소
- `/home/laba/Berkeley-Humanoid-Lite/` — 풀 BHL 저장소

---

## 🆘 곤란할 때

1. **모터가 응답 안 함** → CAN 결선 (특히 GND), 24V supply, 24V→BESC 결선 순서로 확인
2. **CAN 인터페이스 사라짐** (`No such device`) → CANable USB 다시 꽂고 slcand 재시작
3. **flash 쓰기 안 됨** (`write protected`) → openocd로 RDP/WRP 해제 (`WORKFLOW.md` 함정 11 참조)
4. **펌웨어 새로 굽기** → STM32CubeProgrammer 또는 `cp *.bin /media/laba/DIS_G431CB/` (USB MSC 드래그-앤-드롭)

---

## ⭐ 핵심 한 줄

**12개 모터 다 캘리브됨. id=1만 납땜 다시 필요. 다음은 본체 조립 + 영점 캘리브.**
