<!--
공식 BHL 문서 vs 이 프로젝트의 실제 상태 차이점 모음
created: 2026-05-24
maintainer: Jimmy
-->

# 공식 문서 ↔ 본 프로젝트 상태 차이점

> 공식 BHL GitBook 의 내용과 사용자(Jimmy)의 이 프로젝트(`/home/laba/DGX-NUC/BHL/`) 의 실제 작업 상태가 **다르거나 보강된** 항목 모음.
> 새 AI 세션이 공식 문서를 보고 "X 라고 적혀있던데?" 라고 단정 짓기 전에 먼저 여기를 확인해야 함.

---

## 1. 캘리브레이션 전류: 1A vs 5A

- **공식**: [`15_hw_flashing.md`](15_hw_flashing.md) 에 `~1 A from power supply` 라고만 명시
- **본 프로젝트**: Jimmy 는 **5A** 로 12개 모터 전수 재캘리브 진행 (memory: `project_bhl_recal_2026-05-16`)
- **확인 방법**: `BHL/CALIBRATION/electronic_calib/calibrate_one.py` 의 캘리브 전류 인자 / 사용자 워크플로

## 2. MOTOR_PHASE_ORDER = -1 flash 저장 버그

- **공식**: [`15_hw_flashing.md`](15_hw_flashing.md) 는 회전 방향이 반대일 때 단순히 `MOTOR_PHASE_ORDER` 를 `-1` 로 바꾸라고만 적혀있음. flash 저장 동작에 관한 언급 **없음**.
- **본 프로젝트 발견**: `phase_order = -1` 은 `0xFFFFFFFF` 가드 때문에 flash 에 저장되지 않음. 펌웨어 패치 필요.
  - memory: `reference_recoil_phase_order_bug`
  - 해결: `BHL/Recoil-Motor-Controller-BESC-id7-phasefix` (phasefix 펌웨어)
  - id=7 검증 완료 (2026-05-23, 9/9). memory: `project_bhl_id7_phase_order`

## 3. 5010 370KV 모터 데이터 부재

- **공식 Flashing 페이지**: `MOTORPROFILE_MAD_5010_370KV` 옵션 존재 → 펌웨어가 370KV 도 지원
- **공식 Motor Characterization 표**: 110KV / 140KV / 310KV 만 실측치 제공. **370KV 누락**
- **본 프로젝트**: 어느 모터가 370KV 인지, 그리고 그 파라미터(R, L, Kτ)가 어디서 오는지 확인 필요. 펌웨어 헤더의 `MOTORPROFILE_MAD_5010_370KV` 정의를 직접 봐야 함 → `BHL/Recoil-Motor-Controller-BESC-id7-phasefix/.../motor_controller_conf.h`

## 4. 파워 커넥터: XT30/XT60 vs XT60/XT90

- **공식**: [`16_hw_robot.md`](16_hw_robot.md) — 메인 버스 **XT60**, 각 액추에이터 분기 **XT30**
- **본 프로젝트**: 24V 분배는 **XT60/XT90** (memory: `project_bhl_power_connectors`, WAGO 아님)
- → 어느 쪽이 현재 본 로봇의 실제 상태인지 한 번 재확인 필요. 사용자 메모리가 공식 문서보다 우선이지만 출처가 어디였는지 확인하면 좋음.

## 5. 학습 머신 OS vs 온보드 OS

- 공식은 **학습 머신 = Ubuntu 24.04**, **온보드 NUC = Ubuntu 22.04** 로 분리
- 본 NUC 시스템: `Linux 6.18.19-rt-x64v3-xanmod1` — 공식 권장 xanmod RT 커널이 이미 깔려 있음

## 6. Joint ID Mapping 최신성

- 공식이 **2025-11-10** 에 한계각 값을 한 번 수정함 (Release Log)
- 본 프로젝트 안 `BHL/docs/` 의 wiring/캘리 노트가 그보다 이전이면 한계각 값에서 차이가 있을 수 있음. 직접 비교 필요.
- 공식 한계각 표는 [`46_indepth_joint_id.md`](46_indepth_joint_id.md) 에 보존됨.

## 7. 모터 인덕턴스 공식 오타 (공식 문서 자체 오류)

- [`43_indepth_motor_char.md`](43_indepth_motor_char.md) 내 phase inductance 계산:
  ```
  L_q = 1/2 R_ll = ...
  ```
  공식은 `L_q = 1/2 * L_ll` 이어야 하지만 본문에 `R_ll` 로 오타. **숫자 값은 맞음** (분모/단위 보면 인덕턴스 계산임이 명확).
- 새 AI 가 이걸 보고 혼동하지 않도록 표만 믿으면 됨.

## 8. BOM에 `limit stopper` 누락 이력

- 공식 Release Log 2025-08-23: "add the missing limit stopper part"
- 본 프로젝트가 그 이전 버전을 다운받았다면 limit stopper 가 빠져있을 수 있음 → MakerWorld 에서 최신 .3mf 재다운로드 검토

## 9. "ankle 변종" = 5010 110KV 확정 (5-24)

- **본 프로젝트 5-15 ~ 5-20 노트** (`docs/motor_calibration_2026-05-15.md`, `docs/WORKFLOW.md`, `CALIBRATION/electronic_calib/0524_motor_index.md`) 에서 ankle 조인트들을 **"ankle 변종"** 으로 표기 — 모델명 미확정 hedge.
- **5-24 교차검증으로 5010 110KV 확정**:
  - 공식 [`43_indepth_motor_char.md`](43_indepth_motor_char.md) Summary 표: 5010 110KV → Kt = 0.1176
  - 공식 lowlevel `BHL/reference/Berkeley-Humanoid-Lite-Lowlevel-main/robot_configuration.backup.json`: ankle 4개 (l/r × pitch/roll) 전부 `torque_constant=0.1176, max_calibration_current=3.0`
  - 펌웨어 `motor_profiles.h` `MOTORPROFILE_MAD_5010_110KV` 블록: `MOTOR_TORQUE_CONSTANT 0.1176f, MOTOR_CALIBRATION_CURRENT 3` — 동일
- **조인트 ↔ 모터 매핑 (다리)**:
  - hip(roll/yaw/pitch) + knee_pitch = **M6C12** (= "6512 액추에이터", Kt 0.08958 펌웨어값 / 0.0919 공식값, 5A cal). 한쪽당 4개.
  - ankle(pitch/roll) = **5010 110KV** (Kt 0.1176, 3A cal). 한쪽당 2개.
- **본 프로젝트 cal_current 차이**: 공식 5010 110KV 권장 3A 인데 SWD 검증 결과 id=12·13·14 는 `max_cal_I = 5A` 로 박혀있음 (id=11만 3A). Cal 시 5A 사용한 흔적 — 5010 110KV 의 한계전류 초과 가능성 있어 별도 확인 필요.

---

## 이 파일의 운영 원칙

- 공식문서가 업데이트되면 → 이 파일 항목도 같이 정정/추가
- 사용자(Jimmy)가 새로운 차이점을 발견하면 즉시 여기에 기록
- 항목 형식: **공식 / 본 프로젝트 / 확인방법** 3단으로
