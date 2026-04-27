# BHL 하드웨어 스펙 (δ3 참조)

출처: Berkeley Humanoid Lite GitHub 리포지토리 분석 (2026-03-27)

---

## 모터

### MAD M6C12 150KV — 고관절/무릎 (6개)

| 파라미터 | 값 |
|---------|-----|
| KV | 150 KV |
| 토크 상수 (Kt) | 0.0919 Nm/A |
| 극쌍 수 | 14 |
| 연속 전류 한계 | 20.0 A |
| IsaacLab effort_limit | 6.0 Nm |
| IsaacLab velocity_limit | 10.0 rad/s |

탑재 위치: hip_pitch, hip_roll, hip_yaw, knee_pitch (좌우 각 3+1 = 6개)

### MAD 5010 110KV — 발목 (4개)

| 파라미터 | 값 |
|---------|-----|
| KV | 110 KV |
| 토크 상수 (Kt) | 0.1176 Nm/A |
| 극쌍 수 | 14 |
| 연속 전류 한계 | 20.0 A |
| IsaacLab effort_limit | 6.0 Nm |
| IsaacLab velocity_limit | 10.0 rad/s |

탑재 위치: ankle_pitch, ankle_roll (좌우 각 2개 = 4개)

---

## 사이클로이드 기어박스

| 파라미터 | 값 |
|---------|-----|
| 기어비 | -15.0 (음수 = 방향 반전) |
| 적용 범위 | 모든 관절 동일 |

출처: `motor_configuration.json`, `robot_configuration.backup.json`

---

## ESC (B-G431B-ESC1) 제어 파라미터

### CAN 버스

| 파라미터 | 값 |
|---------|-----|
| 프로토콜 | CANopen 유사 (PDO/SDO) |
| 비트레이트 | 1 Mbps |
| 디바이스 ID 범위 | 0~127 (7비트) |
| 물리 인터페이스 | socketcan (Linux) |
| 워치독 타임아웃 | 1000 ms |
| 펌웨어 버전 | 0x20250226 (2025-02-26) |

### 제어 모드

| 코드 | 모드 |
|------|------|
| 0x00 | DISABLED |
| 0x01 | IDLE |
| 0x02 | DAMPING |
| 0x05 | CALIBRATION |
| 0x10 | CURRENT |
| 0x11 | TORQUE |
| 0x12 | VELOCITY |
| 0x13 | POSITION |

### 제어 주기

| 레벨 | 주기 | 주파수 |
|------|------|--------|
| IsaacLab 시뮬 (physics) | 0.005 s | 200 Hz |
| IsaacLab 제어 (decimation=8) | 0.04 s | 25 Hz |
| 실로봇 저수준 제어 | — | ~250 Hz |

---

## IsaacLab 액추에이터 파라미터

### 고관절/무릎 (MAD M6C12)

```python
effort_limit   = 6.0    # Nm
velocity_limit = 10.0   # rad/s
stiffness      = 20.0   # Nm/rad
damping        = 2.0    # Nm·s/rad
armature       = 0.007  # kg·m² (hip), 0.002 (knee)
```

### 발목 (MAD 5010)

```python
effort_limit   = 6.0    # Nm
velocity_limit = 10.0   # rad/s
stiffness      = 20.0   # Nm/rad
damping        = 2.0    # Nm·s/rad
armature       = 0.002  # kg·m²
```

---

## ROS2 토픽 초안

기획서 5.3절 기반. Week 1 합의 미팅에서 확정 예정.

### Orin → NUC (상위 → 하위)

| 토픽 | 메시지 타입 | 내용 | 주기 |
|------|------------|------|------|
| `/gait/cmd` | `geometry_msgs/Twist` | 보행 속도 명령 (vx, vy, wz) | 25 Hz |
| `/gait/mode` | `std_msgs/String` | 보행 모드 (WALK/STAND/STOP) | on-change |
| `/arm/cmd` | `sensor_msgs/JointState` | 팔 관절 목표 위치 | 30 Hz |

### NUC → Orin (하위 → 상위)

| 토픽 | 메시지 타입 | 내용 | 주기 |
|------|------------|------|------|
| `/gait/status` | `std_msgs/String` | 보행 상태 피드백 | 25 Hz |
| `/joint/state` | `sensor_msgs/JointState` | 관절 위치/속도/토크 | 250 Hz |
| `/imu/data` | `sensor_msgs/Imu` | IMU (가속도, 자이로, 쿼터니언) | 200 Hz |
| `/contact/state` | `std_msgs/Bool[4]` | 발바닥 접촉 여부 | 200 Hz |

### 진단/설정

| 토픽 | 메시지 타입 | 내용 |
|------|------------|------|
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 모터 온도, 전류, 에러 |
| `/can/raw` | `can_msgs/Frame` | CAN 원시 프레임 (디버그) |

> **미확정** (Week 1 합의 미팅에서 결정): SO-ARM 관절 수 및 매핑, 카메라 토픽 구조, 비상정지 `/estop`

---

## 참고 파일 경로

```
/home/laba/Berkeley-Humanoid-Lite/
├── motor_configuration.json
├── source/berkeley_humanoid_lite_lowlevel/
│   ├── robot_configuration.backup.json
│   └── berkeley_humanoid_lite_lowlevel/recoil/core.py  # CAN 프로토콜
└── source/berkeley_humanoid_lite_assets/
    ├── berkeley_humanoid_lite_assets/robots/
    │   └── berkeley_humanoid_lite.py  # IsaacLab 액추에이터 설정
    └── data/robots/berkeley_humanoid/berkeley_humanoid_lite/urdf/
        └── berkeley_humanoid_lite.urdf
```
