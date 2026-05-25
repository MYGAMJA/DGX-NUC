# CAN 결선 & 신호 테스트 — 하드웨어 브링업 (2026-05-20)

> 작성: 2026-05-20
> **오늘 목표: CAN 버스에 12개 모터 신호 잡히는지 확인**
> 선행: 모터 캘리브레이션 (`jimmy_readme2.md`, 2026-05-15~19)
> 다음: 조인트 영점 캘리브 → 정책 배포 ([35_deployment_design](35_deployment_design_2026-05-20.md))

---

## 현황 (2026-05-20)

- ✅ 다리(모터 12개 장착) 조립 완료
- ✅ 모터 12개 전기 캘리브 + flash 영구화 (jimmy_readme2)
- ✅ 각 BESC에서 CAN 신호선 **3가닥(H/L/G) × 2리드(in/out)** 인출 완료
- 토폴로지: **2-bus 확정**

```
can0 왼발 : CANable#1[120Ω] ─ id1 ─ id3 ─ id5 ─ id7 ─ id11 ─ id13[termON 펌웨어]
can1 오른발: CANable#2[120Ω] ─ id2 ─ id4 ─ id6 ─ id8 ─ id12 ─ id14[termON 펌웨어]
```

## 오늘 작업

1. 신호선 데이지체인 결선 (납땜)
2. id=13 termON 펌웨어 reflash (왼발 버스 종단) — ✅ 완료 (2026-05-20)
3. **CAN 신호 테스트** ← 오늘 목표

---

## 1. 배선 — 30 AWG 괜찮은가?

**오늘 신호 테스트 목적이면 30 AWG 납땜으로 OK.**

CAN 신호선(H/L/G)은 데이터선이라 흐르는 전류가 아주 작음(수십 mA). 30 AWG도 전기적으로 충분.

단, 2가지 권장:

| 항목 | 권장 | 이유 |
|---|---|---|
| **H/L 두 가닥 꼬기** | 데이지체인 구간마다 H+L 을 **트위스트 페어**로 꼬기 | CAN은 차동신호 — H/L이 꼬여야 노이즈 상쇄. 안 꼬면 1Mbit에서 에러 가능 |
| **최종 보행 빌드** | 26~28 AWG + strain relief | 30 AWG는 보행 진동에 피로 파단 위험. 이미 id=1·id=2 납땜 파손 이력 있음 |
| 선 종류 | 단선보다 **연선(stranded)** | 굽힘·진동에 강함 |

→ 오늘 테스트는 30 AWG로 진행 OK. 보행까지 갈 거면 신호선도 결국 굵게/연선으로 교체 권장.

## 2. 데이지체인 결선 (납땜)

각 보드: 3가닥(H/L/G) × 2리드. **보드 N의 "out" 리드 → 보드 N+1의 "in" 리드.**

- 접합 매핑: **H↔H, L↔L, G↔G** — H/L 바뀌면 통신 전부 실패
- 접합점당 3 납땜 — 각각 수축튜브. 특히 **GND 확실히** (jimmy 함정 2: GND 빠지면 1시간 날림)
- 별(star) 배선 금지, 가지(stub) 짧게 — 선형 버스 유지
- 접합부 strain relief — 보행 진동 받는 부위

## 3. 종단 120Ω — 각 버스 양 끝 2곳

- **CANable Pro 쪽 끝**: 내부 120Ω 점퍼 ON — **CANable 2개 다** 확인
- **반대쪽 끝 BESC**:
  - can1 오른발: id=14 `termON` 펌웨어 ✓ (이미 됨)
  - can0 왼발: id=13 → `termON` 펌웨어 ✓ (2026-05-20 reflash 완료)
- ⚠️ 끝당 종단 1개만 — id=13은 termON 했으므로 외부 저항 **추가 금지** (둘 다 하면 60Ω). 혹시 저항 넣었으면 제거.

## 4. CAN 신호 테스트 (오늘 목표)

### 4-1. slcand 셋업 (2-bus)

```bash
# CANable이 어느 ttyACM인지 확인
for dev in /dev/ttyACM*; do udevadm info -q property -n $dev | grep ID_MODEL=; done

sudo killall slcand 2>/dev/null; sleep 1
sudo slcand -o -c -s8 /dev/ttyACMx can0   # 왼발 CANable (x = 실제 번호)
sudo slcand -o -c -s8 /dev/ttyACMy can1   # 오른발 CANable
sudo ip link set up can0
sudo ip link set up can1
ip -details link show can0 | grep -E "state|bitrate"
```

### 4-2. 버스별 sweep — 한 버스씩

```bash
PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel python3 -c "
import berkeley_humanoid_lite_lowlevel.recoil as recoil
bus = recoil.Bus('can0', 1000000)
for n in [1,3,5,7,11,13]:
    print(f'id={n}:', 'online' if bus.ping(n) else 'OFFLINE')
bus.stop()
"
```
can1 도 `[2,4,6,8,12,14]` 로 동일 실행. 각 버스 6개 응답하면 성공.

### 4-3. 안 잡힐 때 (jimmy 디버깅 이력 기준)

- `candump can0` 켜고 `cansend can0 201#CA` → echo(`201`) 외에 다른 ID 프레임 오나
- 24V ON 인지 — CAN 트랜시버는 24V 레일로 켜짐 (함정 10)
- GND 결선 / 종단저항 / H-L 안 바뀌었나
- 6개 단위로 추적 — 12개 한 번에 묶고 디버깅 금지

---

## 체크리스트

```
[ ] can0 (왼발) 6보드 데이지체인 결선 (납땜)
[ ] can1 (오른발) 6보드 데이지체인 결선 (납땜)
[x] id=13 끝 종단 — termON 펌웨어 reflash 완료
[ ] CANable 2개 120Ω 점퍼 ON 확인
[ ] 24V ON
[ ] slcand can0 / can1 기동
[ ] sweep can0 — id 1,3,5,7,11,13 응답
[ ] sweep can1 — id 2,4,6,8,12,14 응답
```

## 진행 로그

- 2026-05-20 — 시작

| 시각 | 작업 | 결과 |
|------|------|------|
|  |  |  |

### sweep 결과 (붙여넣기)

```
can0 :
can1 :
```

### 이상 모터 / 이슈

| CAN | ID | 조인트 | 증상 | 조치 |
|---|---|---|---|---|
|  |  |  |  |  |
