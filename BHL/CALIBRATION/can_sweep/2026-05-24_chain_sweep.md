# 2026-05-24 — 12/12 CAN 체인 sweep 결과

조립된 다리에 배터리 전원 + CANable 2개 동시 연결 상태에서 12개 BESC 전수 ping 응답 검증.
**12/12 전부 응답** ✓

## 환경

- **전원**: 6S 리포 배터리 (~22V) → 전원차단스위치 → PDB 2개 (좌·우다리 각 6개 분기)
- **CAN 어댑터**: MKS CANable V2.0 Pro × 2 (slcan 펌웨어, 1Mbps)
  - 왼다리: `/dev/canable-left` (시리얼 `209B31A05842`) → `can0`
  - 오른다리: `/dev/canable-right` (시리얼 `20A031A65842`) → `can1`
- **호스트**: laba-desktop (Linux 6.18.19-rt-x64v3-xanmod1)
- **bitrate 설정**: slcand `-s8` (= 1Mbps)
- **체인 종단**: 좌 id=13 termON, 우 id=14 termON, 나머지 termOFF (`0524_motor_index.md` 와 일치)

## 명령

```bash
sudo bash /home/laba/DGX-NUC/BHL/scripts/can_up.sh   # slcand 기동 + ip link up

for id in 1 3 5 7 11 13; do
  PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel \
    python3 BHL/reference/Berkeley-Humanoid-Lite-Lowlevel-main/scripts/motor/ping.py \
    -c can0 -i $id
done

for id in 2 4 6 8 12 14; do
  PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel \
    python3 BHL/reference/Berkeley-Humanoid-Lite-Lowlevel-main/scripts/motor/ping.py \
    -c can1 -i $id
done
```

## 결과

|       | 왼다리 (can0) | 오른다리 (can1) |
|------:|---|---|
| hip_roll    | id=1 ✅ online | id=2 ✅ online |
| hip_yaw     | id=3 ✅ online | id=4 ✅ online |
| hip_pitch   | id=5 ✅ online | id=6 ✅ online |
| knee_pitch  | id=7 ✅ online | id=8 ✅ online |
| ankle_pitch | id=11 ✅ online | id=12 ✅ online |
| ankle_roll  | id=13 ✅ online | id=14 ✅ online |

→ **12 / 12** 응답 확인. 보드 교체된 id=7 (left_knee_pitch, phasefix 0x20260523) 포함 전부 OK.

## 의미

- `project_bhl_id7_phase_order` 메모리의 "체인 sweep만 남음" 항목이 이걸로 해소됨.
- 5-21 CAN debug 당시 미해결이던 id=7 무응답 이슈도 보드 교체 + 배선 정리로 완전 해결.
- 12개 BESC 모두 flash 캘리브 값 + 펌웨어 + CAN 종단 설정 정합 상태에서 통신까지 확인.

## 작업 중 짚어둘 점

- `slcand` 가 `/dev/canable-left` 를 잡고 있을 때 CANable USB 를 뽑았다 다시 꽂으면 stale 핸들 상태가 됨. 재꽂은 후 반드시 `BHL/scripts/can_up.sh` 재실행 필요 (`killall slcand` 후 재기동).
- 새로 들어온 CANable 은 `ttyACM` 번호가 바뀔 수 있지만 (이번엔 `ttyACM0` 으로 잡힘 — ST-LINK가 빠진 슬롯), udev 룰 `99-canable.rules` 가 시리얼 기준 매핑이라 `canable-left/right` 심볼릭은 항상 일관.

## 다음 단계 후보

- 개별 모터 동작 테스트 (`move_actuator.py` 로 sin 궤적, 한 모터씩)
- id=7 회전 방향 최종 검증 (phase_order=−1 결과 시각 확인)
- 조인트 영점 캘리브 (`calibrate_joints.py`)
