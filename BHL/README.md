# BHL — Berkeley Humanoid Lite (하체 12 모터)

## 디렉터리

| 경로 | 내용 |
|---|---|
| `CALIBRATION/electronic_calib/` | `calibrate_one.py` — 모터 1개 단위 전기적 캘리브. 결과 JSON은 여기 직속에 저장. |
| `CALIBRATION/electronic_calib/backup/` | 과거 캘리브 JSON 스냅샷. **AI는 명시 요청 없이 읽지 말 것 ([../CLAUDE.md](../CLAUDE.md))** |
| `CALIBRATION/firmware/` | BESC 펌웨어 `.bin` 3개 (termON / termOFF / id7-phasefix) + id7 flash 백업 1개 |
| `CALIBRATION/can_sweep/` | CAN ping/sweep 도구 (작성 예정) |
| `CALIBRATION/apply_id7_phase_order.py` | id=7 phase_order 워크어라운드 (전원사이클마다 실행, phasefix 펌웨어 적용 후엔 불필요) |
| `scripts/can_up.sh` | CAN 인터페이스 셋업 (`sudo bash scripts/can_up.sh`) |
| `udev/99-canable.rules` | CANable V2 시리얼 기반 고정 심볼릭 링크 룰 |
| `docs/` | 작업 기록 MD (날짜별, 디버그 핸드오프, WORKFLOW 등) |
| `reference/` | 외부 소스/매뉴얼 (편집 금지) |

## ID ↔ 조인트 매핑 (하체 12 모터)

| ID | 조인트 | CAN | term | ID | 조인트 | CAN | term |
|---|---|---|---|---|---|---|---|
| 1 | left_hip_roll | can0 | OFF | 2 | right_hip_roll | can1 | OFF |
| 3 | left_hip_yaw | can0 | OFF | 4 | right_hip_yaw | can1 | OFF |
| 5 | left_hip_pitch | can0 | OFF | 6 | right_hip_pitch | can1 | OFF |
| 7 | left_knee_pitch | can0 | OFF | 8 | right_knee_pitch | can1 | OFF |
| 11 | left_ankle_pitch | can0 | OFF | 12 | right_ankle_pitch | can1 | OFF |
| **13** | left_ankle_roll | can0 | **ON** (체인 끝) | **14** | right_ankle_roll | can1 | **ON** (체인 끝) |

## 자주 보는 문서 (`docs/`)

| 파일 | 내용 |
|---|---|
| `WORKFLOW.md` | 펌웨어 빌드/플래시/캘리브 전체 워크플로우 |
| `motor_calibration_2026-05-15.md` | 12개 모터 캘리브 로그 (calibrate_one.py 자동 append) |
| `id7_phase_order_2026-05-22.md` | id=7 phase_order 버그 + phasefix 펌웨어 적용 |
| `wiring_guide_2026-05-20.md` | 결선 / termON·OFF 배치 |
| `can_debug_2026-05-21.md` | CAN 디버그 세션 |
| `BHL_CAN_Debug_Handoff.md` | 초기 CAN 디버그 핸드오프 |
| `SUMMARY_for_next_user.md` | 인수인계 요약 |
| `jimmy_readme1.md`, `jimmy_readme2.md` | 개인 작업 노트 |

## 펌웨어 `.bin` 가이드

| 파일 | 빌드 | 용도 |
|---|---|---|
| `firmware_termOFF.bin` | ~2026-05-19 | CAN 체인 중간 (id 1~8, 11, 12) |
| `firmware_termON.bin` | 2026-05-16 | CAN 체인 끝점 (id 13, 14) |
| `firmware_id7_phasefix_termOFF_20260523.bin` | 2026-05-23 | id=7 전용 (phase_order=-1 flash 저장 가드 수정) |

플래시: `st-flash --reset write <file> 0x8000000`

## 외부 의존성 (`reference/`)

- `Berkeley-Humanoid-Lite-Lowlevel-main/` — `berkeley_humanoid_lite_lowlevel.recoil` 모듈 (calibrate_one.py가 import)
- `Recoil-Motor-Controller-BESC-id7-phasefix/` — phasefix 펌웨어 소스
- `official_manuals/BESC/` — B-G431B-ESC1 데이터시트
- `official_manuals/canable/` — CANable V2 submodule
- `MOTOR/` — 펌웨어 플래싱 가이드 (PDF + StepByStep MD)
