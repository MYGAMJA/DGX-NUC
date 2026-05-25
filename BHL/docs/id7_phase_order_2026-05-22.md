# id=7 (left_knee_pitch) phase_order 진단 (2026-05-22)

> BESC+엔코더 보드 교체 후 id=7이 폐루프 제어에서 buzz(한계진동). 원인 추적 → phase_order 문제로 확정.
> 순서: **현재 상황(결론) → 해결책 → 진행한 테스트들**

---

## 0. 2026-05-23 업데이트 — 펌웨어 수정 완료

- id=7 보드에 `phase_order=-1` flash 로드 버그 수정 펌웨어 적용 완료.
- 새 펌웨어: `BHL/firmware_id7_phasefix_termOFF_20260523.bin`
- CAN termination: termOFF.
- CAN 확인:
  - `device_id = 7`
  - `firmware_version = 0x20260523`
  - `ERROR = 0x0000`
  - `VBUS ≈ 23.65V`
  - `flux_offset = 86.06014251708984`
  - `phase_order = -1`
- ST-LINK reset 후에도 `phase_order=-1` 유지 확인.
- USB/ST-LINK 분리 + 전원공급기 OFF/ON 이후에도 `phase_order=-1` 유지 확인.
- 즉, 기존의 "전원 사이클마다 phase_order가 1로 복귀" 문제는 펌웨어 수정으로 해결된 상태.

주의:
- 작업 중 실수로 id=1 보드에 phasefix 펌웨어가 잠깐 플래시됐으나, 즉시 기존 `BHL/firmware_termOFF.bin`으로 복구 플래시 완료.
- id=1 config page는 건드리지 않았으므로 `device_id=1`, `flux_offset=-11.7579984664917`은 유지될 것으로 예상. 추후 id=1 단독 CAN 확인 권장.

남은 검증:
- [x] id=7 저위험 이동추종 테스트 — **9/9 통과** (2026-05-23, 자유회전, ilim=2.5A, 측정범위 0.988 rad, 최대 추종오차 0.081 rad)
- [x] id=1 단독 CAN 확인 — **config page 무사** (2026-05-23): `device_id=1`, `firmware=0x20250226` (원래 펌웨어, phasefix 아님), `flux_offset=-11.7580`, `phase_order=1`, `ERROR=0x0`, `VBUS=23.72V`
- [ ] 왼다리 체인 sweep (`id=1/7/13` 동시 검출) — 다리 조립 시점에 재개
- [ ] 왼다리 체인 조립 후 반복 sweep + 흔들림 테스트

---

## 1. 현재 상황 (결론)

### TL;DR
- id=7은 새 BESC에 **3상 모터선이 비표준 순서로 연결**돼서 `phase_order = -1`이 필요함.
- `phase_order = -1`을 적용하면 **모터는 완벽히 정상 동작** — 이동추종 테스트 9/9 통과 검증됨.
- 그런데 **이 펌웨어 빌드는 `phase_order = -1`을 flash에 저장하지 못함** → 전원 사이클마다 1로 복귀 → 다시 buzz.
- 즉 모터·엔코더·캘리브 다 정상이고, 막힌 건 오직 "phase_order=-1의 영구 저장"이다.

### 근본 원인 (펌웨어 확정)
`motor_controller.c:252` — `MotorController_loadConfig`:
```c
if (controller_config->motor.phase_order != 0xFFFFFFFF)
    controller->motor.phase_order = (int8_t)controller_config->motor.phase_order;
```
- `phase_order = -1`을 flash에 저장하면 32비트로 **`0xFFFFFFFF`**가 됨.
- 위 가드는 빈/erased flash 방어용(WORKFLOW 함정 5 패치) — 값이 `0xFFFFFFFF`면 "비어있음"으로 보고 무시.
- 그래서 `phase_order = -1`이 가드에 걸려 무시됨 → 부팅마다 컴파일 기본값 `1`로 복귀.
- `flux_offset`은 float(NaN 가드)라 영향 없음 → 86.06은 정상적으로 유지됨.

### 왜 id=7만 문제인가
- 표준 BHL 조립: AS5600이 BESC 위에 있고 BESC는 모터 케이스에 고정 → 센서·권선이 한 강체. 다른 11개 모터는 3상 결선이 표준 순서 → `phase_order = 1` → 정상 저장/동작.
- id=7은 BESC 교체 시 3상 선을 새 보드에 다른 순서로 연결 → `phase_order = -1` 필요 → 저장 불가 버그에 걸림.

### 현재 id=7 상태 (2026-05-22 기준)
- `device_id = 7`, `ERROR = 0x0`, `VBUS ≈ 23.7V` — 통신·전원 정상.
- `flux_offset = 86.06` (flash 저장됨, 유지됨).
- `phase_order = 1` (부팅 기본값 — −1 저장 불가로 복귀).
- → flux_offset(−1 기준 캘리브값)과 phase_order(1) 불일치 → **폐루프에서 buzz, 미작동.**
- 모드: IDLE (안전).

---

## 2. 해결책

`phase_order = -1`은 "3상 중 2개를 맞바꾼 것"의 소프트웨어 등가물. 세 가지 방법:

### A. 3상 선 2개 맞바꾸기 — **권장**
1. id=7 BESC에서 모터 3상 선 중 **아무 2개를 물리적으로 맞바꿔** 연결.
2. 그러면 모터가 `phase_order = +1`(기본값, 정상 저장됨)로 올바르게 동작.
3. `calibrate_one.py --id 7`로 재캘리브 (phase_order는 기본값 1 그대로 둠) → 새 flux_offset 산출.
4. flush_offset은 float이라 영구 저장 OK. 전원 사이클 후에도 유지됨.
- 장점: 영구적, 펌웨어 안 건드림, id=7이 나머지 11개와 동일 구성(phase_order=1).

### B. 펌웨어 수정
- `motor_controller.c:252` 가드를 phase_order=-1과 충돌 안 하게 수정(다른 sentinel 사용 등) 후 재빌드·재플래시.
- 단점: 빌드 환경 필요(WORKFLOW §2), id=7만 다른 펌웨어가 됨.

### C. 매 부팅 소프트웨어 재적용 (임시)
- 로봇 기동 스크립트에서 매번 id=7에 `phase_order = -1` 써넣기.
- 단점: id=7 전용 예외 → 누락 위험. 임시 검증용으로만.

> 진행 결정: **A 권장.** 선택 시 절차 3~4 수행 후 이동추종 테스트로 재검증 + 전원 사이클 검증.

---

## 3. 진행한 테스트들

### 3.1 전기 캘리브 시도 (3회)

| # | 시각 | flux_offset (전→후) | phase_order | 조건 | 결과 |
|---|---|---|---|---|---|
| 1 | 19:52 | −53.86 → **−79.40** | 1 | 다리 구조물에 조립된 채 (회전 막힘) | ❌ buzz — 풀회전 불가로 캘리브 무효 |
| 2 | 22:04 | −79.40 → **−35.42** | 1 | 분리 후 자유회전 | ❌ buzz — 자유회전인데도 실패 → phase_order 의심 |
| 3 | 22:16 | −35.42 → **86.06** | **−1** | 자유회전 | ✅ 이동추종 9/9 통과 — 단 전원사이클 후 phase_order=1 복귀 |

- 펌웨어 캘리브는 로터를 **한 바퀴 기계 회전**시킴(`motor_controller.c`). #1은 다리에 막혀 풀회전 못 해 무효.
- `calibrate_one.py`의 "flux 바뀜=성공" 판정은 약함 — 실제 폐루프 동작으로 검증해야 함.

### 3.2 POSITION 동작 검증

| 대상 | 테스트 | 결과 |
|---|---|---|
| flux −79.40, po=1 | POSITION 유지 | ❌ 폭주판정 — velocity +13.77 rad/s, 사용자: 제자리 진동 |
| flux −53.86, po=1 | POSITION 유지 + 3s 트레이스 | ❌ 한계진동 — drift ±0.2, velocity ±12, torque 포화 지속 |
| flux −35.42, po=1 | 이동추종 (±0.5 rad 9단계) | ❌ 4/9, 측정이동범위 0.456 rad, 사용자: 진동 |
| **flux 86.06, po=−1** | **이동추종 (±0.5 rad 9단계)** | **✅ 9/9, 측정이동범위 1.001 rad, 추종오차 ≤0.043 rad** |
| flux 86.06, po=1 (전원사이클 후) | 이동추종 | ❌ 1/9, 측정이동범위 0.222 rad — phase_order 복귀로 재실패 |

- 검증 핵심: `phase_order=-1`일 때만 9/9 깔끔히 추종. → phase_order가 원인임을 확정.

### 3.3 엔코더 진단 (정상 확인 — 오진 배제용)

| 테스트 | id=7 결과 | id=1(정상 모터) 결과 | 판정 |
|---|---|---|---|
| 손돌림 position 추종 | position 거의 안 변함 | — | (모터 본체 미고정 시 측정 함정, 무효 데이터) |
| raw 엔코더 (정지) | i2c_cnt=0, raw 950~957 (지터 ±3~4), velocity ±10~22 노이즈 | i2c_cnt=0, raw 3413~3418 (지터 ±2~3), velocity ±10~22 | **동일 — id=7 엔코더 정상** |

- `ENCODER_I2C_UPDATE_COUNTER = 0`은 정상 모터도 동일 → 이 펌웨어가 안 쓰는 값. **고장 지표 아님.**
- velocity ±20 rad/s 노이즈는 정지 상태에서도 정상 모터(id=1)와 동일 → 이 엔코더/펌웨어의 정상 노이즈. **판정 근거로 쓰면 안 됨.**

### 3.4 배제된 가설들

| 가설 | 배제 근거 |
|---|---|
| 엔코더 하드웨어 고장 | id=1과 raw 지터·노이즈 동일 |
| flux_offset 값 문제 | −53.86(옛 검증값), −79.40, −35.42 전부 buzz — 값을 바꿔도 안 됨 |
| 캘리브 시 모터 본체 미고정 | 엔코더+BESC가 모터 케이스에 한 강체 → 본체 미고정은 캘리브 수학에 영향 없음 (다른 11개도 동일 셋업) |
| 중력 부하 | 출력을 손 놓으면 가만히 있음 — 부하 없음 |
| 회전 막힘 (다리 조립) | 캘리브 #1만 해당. 분리 후 #2/#3은 자유회전 |

### 3.5 결정적 단서

1. 캘리브값 3개 전부 phase_order=1에서 buzz.
2. phase_order=−1로 바꾸자 이동추종 9/9 즉시 통과.
3. 전원 사이클 후 phase_order만 1로 복귀, flux는 유지 → 펌웨어 저장 가드 의심.
4. `motor_controller.c:252` 소스 확인 → `0xFFFFFFFF` 가드가 −1을 거부함을 확정.

---

## 다음 작업 (해결책 A 기준)

- [ ] id=7 BESC 3상 선 2개 맞바꿔 재연결
- [ ] `calibrate_one.py --id 7` 재캘리브 (phase_order는 기본값 1)
- [ ] 이동추종 테스트 통과 확인 (`/tmp/move_test.py` 또는 동등 스크립트)
- [ ] 전원 사이클 후 flux_offset·phase_order 유지 + 이동추종 재확인
- [ ] `motor_calibration_2026-05-15.md`의 2026-05-22 id=7 항목(−79.40/−35.42/86.06 "✅") 정정 — 실제로는 실패였음
