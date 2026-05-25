<!--
Berkeley Humanoid Lite 공식 문서 로컬 미러
source: https://berkeley-humanoid-lite.gitbook.io/docs
last_sync: 2026-05-24
last_official_update: 2025-11-10
total_pages: 27
-->

# BHL 공식 문서 로컬 미러 — 라우팅 가이드

GitBook(https://berkeley-humanoid-lite.gitbook.io/docs) 의 모든 27개 페이지를 로컬에 보존한 폴더.
각 파일 상단의 HTML 주석에 **원본 URL**과 **마지막 sync 날짜**가 박혀 있다.

## 어떤 질문이 들어왔을 때 어디를 봐야 하나

### 🔋 모터 / 액추에이터 / 캘리브레이션
- **FOC 이론** (Clarke/Park, SVPWM, PI 게인 계산) → [`41_indepth_foc.md`](41_indepth_foc.md)
- **M6C12/5010 모터 R/L/Kτ/관성 측정값** (표) → [`43_indepth_motor_char.md`](43_indepth_motor_char.md)
- **electrical offset 캘리브 명령/절차/회전방향** → [`14_hw_actuator.md`](14_hw_actuator.md) + [`15_hw_flashing.md`](15_hw_flashing.md)
- **BESC 펌웨어 4단계 플래시** (`FIRST_TIME_BOOTUP`, `LOAD_*_FROM_FLASH`) → [`15_hw_flashing.md`](15_hw_flashing.md)
- **MOTOR_PHASE_ORDER = -1 의미** (회전 방향 반전) → [`15_hw_flashing.md`](15_hw_flashing.md) 후반부
- **AS5600 인코더 저항 swap** → [`14_hw_actuator.md`](14_hw_actuator.md)
- **펌웨어 실시간성/I2C 78μs/FOC 12kHz 한계** → [`42_indepth_firmware_timing.md`](42_indepth_firmware_timing.md)

### 🚌 CAN / 통신
- **CAN 프로토콜 전체 스펙** (NMT/SDO/PDO1~4/FLASH/HEARTBEAT) → [`45_indepth_can.md`](45_indepth_can.md)
- **Joint ↔ CAN bus/ID 매핑 + 한계각** (22 DOF 전부) → [`46_indepth_joint_id.md`](46_indepth_joint_id.md)
- **CAN bitrate/timing 설정** (1Mbps, sample-point 0.75) → [`45_indepth_can.md`](45_indepth_can.md)

### 🤖 하드웨어 빌드
- **BOM 구글시트 링크** → [`11_hw_bom.md`](11_hw_bom.md)
- **3D 프린팅 프로파일** (Housing vs Shaft) → [`13_hw_3d_printing.md`](13_hw_3d_printing.md)
- **배선/커넥터** (XT30/XT60, WAGO, CAN 케이블) → [`16_hw_robot.md`](16_hw_robot.md)
- **IMU 비교** (BNO085 vs IM10A) → [`44_indepth_imu.md`](44_indepth_imu.md)
- **필요 도구** → [`12_hw_tools.md`](12_hw_tools.md)

### 💾 소프트웨어 / 학습
- **개발환경 셋업** (uv/conda, Isaac Sim 4.5/Lab 2.1, Ubuntu 24.04) → [`22_sw_training.md`](22_sw_training.md)
- **MuJoCo sim2sim** → [`23_sw_sim2sim.md`](23_sw_sim2sim.md)
- **온보드 NUC** (Ubuntu 22.04, xanmod RT 커널, C vs Python codebase) → [`24_sw_onboard.md`](24_sw_onboard.md)
- **SteamVR 텔레옵** (Windows PC 필요, UDP 172.28.0.x) → [`25_sw_mocap.md`](25_sw_mocap.md)
- **Onshape → URDF/MJCF/USD export** → [`47_indepth_onshape.md`](47_indepth_onshape.md)
- **학습 코딩 컨벤션** (Isaac Lab configclass 순서) → [`48_indepth_coding_conv.md`](48_indepth_coding_conv.md)
- **학습서버 sync (vscode-sftp)** → [`49_indepth_server_sync.md`](49_indepth_server_sync.md)

### 📥 다운로드 / 외부 자원
- **Onshape CAD, MakerWorld 3D프린팅 .3mf, GitHub repos, arXiv 논문** → [`01_releases.md`](01_releases.md)
  - 메인 repo: `HybridRobotics/Berkeley-Humanoid-Lite`
  - 펌웨어 repo: `T-K-233/Recoil-Motor-Controller-BESC`
  - 로우레벨: `HybridRobotics/Berkeley-Humanoid-Lite-Lowlevel`
  - assets(URDF): `HybridRobotics/Berkeley-Humanoid-Lite-Assets`

### ⚠ 사용자 작업상태 ↔ 공식문서 차이점
- **반드시 한번은 읽을 것** → [`_CROSS_REFERENCE.md`](_CROSS_REFERENCE.md)
  공식 문서에 적힌 값과 이 프로젝트(`BHL/CALIBRATION/`, `BHL/scripts/` 등)의 실제 상태가 다른 항목을 모아둠. 사용자가 "공식 문서에서는 X라는데?" 라고 물을 때 먼저 봐야 함.

## 전체 파일 목록 (번호 순)

| 번호 | 주제 | 원본 경로 |
|---|---|---|
| 00 | Home | /docs |
| 01 | **Releases (다운로드 허브)** | /docs/releases |
| 10~16 | Hardware (overview, BOM, tools, 3D printing, actuator, flashing, robot) | /docs/getting-started-with-hardware/* |
| 20~25 | Software (overview, env, training, sim2sim, onboard, mocap) | /docs/getting-started-with-software/* |
| 30 | lerobot (빈 페이지: "Coming soon") | /docs/lerobot-integration |
| 40~49 | In-depth (overview, FOC, fw timing, motor char, IMU, CAN, joint ID, Onshape, coding conv, server sync) | /docs/in-depth-contents/* |
| 90 | Contribute | /docs/contribute |

## 유지보수 메모

- 마지막 공식 업데이트: **2025-11-10** (Joint ID Mapping 한계각 수정)
- 마지막 로컬 sync: **2026-05-24**
- 재동기화하려면: 각 파일 상단의 `raw_md:` URL에 `curl -sL` 하면 GitBook이 깔끔한 markdown 반환
- 페이지마다 끝에 붙어있던 30줄짜리 "Agent Instructions" 보일러플레이트는 제거 후 저장됨 (sync 시 다시 제거 필요)
- 이미지는 `/files/<ID>` 형태로만 남아 있고 실제 바이너리는 미포함 (필요 시 원본 GitBook 방문)
