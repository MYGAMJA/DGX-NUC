# 하이리온 소프트웨어 개발 계획

> 기준일: 2026.04.20 
> 대상: Orin Nano Super + NUC + SO-ARM101 ×2 + BHL 다리 + ESP32

---

## 개발 철학

> **"작은 것부터 살아있음을 증명한다"**  
> 전체를 한 번에 붙이고 테스트하지 않는다.  
> 모터 1개 → 팔 전체 → SmolVLA 제어 → 상태 머신 트리거 → 풀 시나리오 순으로 올라간다.
> **"환경을 분리해서 작업한다."**
> docker, config, factory 구조를 이해하여 코드에서 발생할 수 있는 이슈를 최소화한다.

---

## 핵심 고려사항
> bhl로 isaac으로 학습하여 걷는다.
> so-arm으로 smolVLA를 통해 짚는다.
> 입의 서보모터, 마이크, 스피커로 대화와 감정표현을 한다.
> 네트워크 연결 해제, 낙상의 문제에 대비한다.

---

## 전체 흐름 요약

```
1단계  전체 구조 설계        인터페이스 확정, 파일 구조 확정
2단계  하드웨어 연결 검증     물리 연결이 살아있는지 터미널 수준에서 확인
3단계  연결부 통신 확인       소프트웨어 인터페이스 end-to-end 확인
4단계  개별 단위 프로그래밍   각 모듈 완성도 높이기
5단계  통합 테스트            풀 시나리오 동작 확인
```

---

## 1단계 — 전체 구조 설계

### 목표
코드를 한 줄도 짜기 전에 전체 인터페이스를 확정한다.  
이후 변경은 전체 공지 필수.

### 프로젝트 파일 구조

```
Hylion/
│
├── jetson/                               # Jetson에서 실행되는 코드
│   ├── arm/                              # so-arm (팔)
│   │   ├── policy/                       # SmolVLA runner (팔 제어 정책)
│   │   │   ├── smolvla_runner.py         # 추론 실행
│   │   │   └── async_inference.py        # 비동기 추론
│   │   ├── so_arm.py                     # 실제 하드웨어 인터페이스
│   │   ├── mock_arm.py                   # 개발용 Mock
│   │   └── factory.py                    # Real/Mock 선택
│   │
│   ├── cloud/                            # Groq 클라우드 연동
│   │   └── groq_client.py               # STT / LLM / TTS API
│   │
│   ├── perception/                       # 인식 모듈
│   │   ├── camera.py                     # RGB 카메라 (Real)
│   │   ├── mock_camera.py                # 개발용 Mock
│   │   ├── mediapipe_tracker.py          # MediaPipe 시선 추적 (CPU)
│   │   ├── imu.py                        # IMU (Real, 낙상 감지용)
│   │   ├── mock_imu.py                   # 개발용 Mock
│   │   └── factory.py                    # Real/Mock 선택
│   │
│   ├── expression/                       # 감정표현 모듈
│   │   ├── mouth_servo.py                # 입 서보모터 제어 (Real)
│   │   ├── mock_mouth_servo.py           # 개발용 Mock
│   │   ├── speaker.py                    # 스피커 출력
│   │   ├── microphone.py                 # 마이크 입력
│   │   └── factory.py                    # Real/Mock 선택
│   │
│   ├── safety/                           # 안전 모듈
│   │   ├── emergency_stop.py             # 비상정지 통합 관리
│   │   ├── watchdog.py                   # 통신 상태 감시
│   │   └── fault_detector.py             # 이상 상태 감지
│   │
│   ├── state_machine/                    # 상태 머신
│   │   └── fsm.py                        # IDLE / TALKING / MANIPULATING / EMERGENCY
│   │
│   ├── core/                             # 통합 조율 로직
│   │   └── coordinator.py               # 상태 보고 각 모듈 실행 판단
│   │
│   └── scenarios/                        # 시연 시나리오
│       ├── base_scenario.py              # 시나리오 베이스 클래스
│       └── serve_cup.py                  # "컵 가져다줘" 시연 예시
│
├── nuc/                                  # NUC에서 실행되는 코드
│   └── bhl/                             # BHL git submodule (보행)
│       ├── csrc/                         # C/C++ 실시간 보행 제어 + ONNX 실행
│       │                                 # IMU (보행 균형용) 내장
│       │                                 # C++ 수준 비상정지 내장
│       ├── robot/                        # 보행 하드웨어 인터페이스
│       ├── mock_biped.py                 # 개발용 Mock
│       └── factory.py                    # Real/Mock 선택
│
├── comm/                                 # Orin ↔ NUC 실시간 통신 (ZeroMQ TCP, 양쪽 모두 설치)
│   ├── protocol.py                       # 공유 메시지 스키마 (msgpack 직렬화)
│   ├── orin/
│   │   ├── sender.py                     # Orin → NUC 명령 전송 (ZMQ PUB)
│   │   └── receiver.py                   # NUC → Orin 상태 수신 (ZMQ SUB)
│   ├── nuc/
│   │   ├── sender.py                     # NUC → Orin 상태 발신 (ZMQ PUB)
│   │   └── receiver.py                   # Orin → NUC 명령 수신 (ZMQ SUB)
│   └── mock_bridge.py                    # 개발용 Mock (단일 프로세스 루프백)
│
├── sim/                                  # 개발 전용 (배포 제외)
│   ├── isaaclab/                         # BHL 보행 RL 학습 환경
│   │   └── tasks/
│   │       ├── velocity_biped.py         # Velocity-BHL-Biped-v0
│   │       └── velocity_humanoid.py      # Velocity-BHL-v0 (팔 포함)
│   ├── mujoco/                           # BHL sim2sim 검증
│   │   └── play_mujoco.py
│   └── isaaclab_arena/                   # SmolVLA 정책 Isaac Sim 검증
│       └── eval_smolvla.py
│
├── dgx/                                  # 개발 전용 (배포 제외)
│   ├── train_biped.sh                    # BHL RL 학습 (Isaac Sim)
│   ├── train_arm.sh                      # SmolVLA 파인튜닝 (LeRobot)
│   └── requirements_dgx.txt             # DGX 전용 패키지
│
├── tests/                                # 개발 전용 (배포 제외)
│   ├── 2_hw_connection/                  # 하드웨어 연결 검증
│   │   ├── check_can.sh                  # CAN 버스 확인
│   │   ├── check_camera.py              # 카메라 확인
│   │   ├── check_imu.py                 # IMU 확인 (Jetson + NUC)
│   │   ├── check_arm.py                 # SO-ARM 확인
│   │   └── check_mouth_servo.py         # 입 서보모터 확인
│   │
│   ├── 3_interface/                      # 소프트웨어 인터페이스 확인
│   │   ├── test_groq_api.py             # Groq STT/LLM/TTS 통신
│   │   ├── test_zmq_bridge.py           # ZeroMQ 토픽 end-to-end
│   │   ├── test_smolvla_io.py           # SmolVLA 입출력
│   │   ├── test_fsm.py                  # 상태머신 전환
│   │   └── test_emergency_stop.py       # 비상정지 트리거 확인
│   │
│   ├── 4_unit/                           # 단위 테스트
│   │   ├── test_mediapipe.py
│   │   ├── test_arm_control.py
│   │   ├── test_walking.py
│   │   ├── test_expression.py
│   │   ├── test_imu.py                  # IMU 낙상 감지
│   │   ├── test_watchdog.py             # 통신 감시
│   │   └── test_tts.py
│   │
│   └── 5_integration/                    # 풀 시나리오 통합 테스트
│       ├── test_full_scenario.py         # 음성 → 팔 동작 전체
│       ├── test_biped_arm.py            # 보행 + 팔 동시
│       └── test_emergency_scenario.py   # 비상정지 풀 시나리오
│
├── checkpoints/                          # ✅ 배포 필요
│   ├── biped/                            # ONNX (NUC에 배포)
│   └── arm/                             # SmolVLA 모델 (Jetson에 배포)
│
├── data/                                 # 개발 전용 (배포 제외)
│   └── episodes/                         # LeRobotDataset 형식
│
├── configs/                              # ✅ 배포 필요
│   ├── dev.yaml                          # Mock 하드웨어 (개발용)
│   ├── prod.yaml                         # 실제 하드웨어 (배포용)
│   └── policy_latest.yaml               # 현재 사용 정책 설정
│
├── docker/                               # 선택적 배포
│   ├── docker-compose.dev.yml
│   ├── docker-compose.prod.yml
│   └── docker-compose.dgx.yml
│
└── scripts/                              # 배포 스크립트
    ├── deploy_jetson.sh                  # Jetson 배포 자동화
    └── deploy_nuc.sh                     # NUC 배포 자동화
```



### 상태 머신 전환 규칙

> WALKING 상태는 보행 기능 반영을 위해 추가. `fsm.py` 구현 시 5개 상태로 확장.

| 현재 상태 | 이벤트 | 다음 상태 |
|----------|--------|----------|
| IDLE | 음성 감지 | TALKING |
| TALKING | LLM 결과 수신 | MANIPULATING |
| MANIPULATING | 집기 완료 | IDLE |
| IDLE / TALKING | 보행 명령 | WALKING |
| WALKING | 정지 명령 | IDLE |
| 모든 상태 | 비상정지 신호 | EMERGENCY |

### ZeroMQ 메시지 인터페이스 (Orin ↔ NUC)

> 전송 방식: ZeroMQ PUB/SUB over TCP (Ethernet 직결)  
> 직렬화: msgpack  
> 포트 규약: Orin PUB=5555, NUC PUB=5556

| 토픽 | 방향 | 타입 | 내용 |
|------|------|------|------|
| `gait/cmd` | Orin → NUC | `str` | `walk_forward`, `stop`, `turn_left` 등 |
| `gait/status` | NUC → Orin | `str` | `walking`, `stopped`, `error` 등 |

### 완료 기준
- [ ] 파일 구조 확정 및 팀 공유
- [ ] ZeroMQ 메시지 스키마 확정 (`comm/protocol.py`)
- [ ] 상태 머신 전환 규칙 확정
- [ ] `configs/dev.yaml` 항목 초안 작성

---

## 2단계 — 하드웨어 연결 검증

> 소프트웨어 없이 터미널 명령어 수준에서 물리 연결을 확인한다.  
> 이 단계를 건너뛰면 이후 디버깅 시 HW/SW 원인 구분이 불가능하다.

### 2-1. Orin → SO-ARM (Waveshare 보드)

```bash
# Waveshare 보드 포트 확인
ls /dev/ttyACM*

# LeRobot으로 모터 1개 응답 확인
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

**확인 항목**
- [ ] 포트 인식 (`/dev/ttyACM0` 등)
- [ ] 모터 1번 ID 응답
- [ ] 모터 6개 전체 응답
- [ ] 캘리브레이션 파일 저장 확인

### 2-2. Orin ↔ NUC (Ethernet)

```bash
# NUC IP 확인 후 ping
ping 192.168.1.x

# ZeroMQ 통신 확인 (NUC에서 PUB 실행 후 Orin에서 SUB로 수신 테스트)
python comm/nuc/sender.py --test
python comm/orin/receiver.py --test
```

**확인 항목**
- [ ] Ethernet 직결 ping 응답
- [ ] ZeroMQ PUB/SUB 메시지 상호 수신
- [ ] `gait/cmd` 발행 → NUC 수신 확인

### 2-3. NUC → BHL 다리 (CAN)

```bash
# CAN 버스 인터페이스 확인
ip link show can0

# 모터 1개에 CAN 패킷 전송 및 응답 확인
```

**확인 항목**
- [ ] CAN 인터페이스 4개 인식
- [ ] BLDC 모터 1개 응답
- [ ] 250Hz 제어 루프 안정성

### 2-4. ESP32 → MOSFET 전원 차단

**확인 항목**
- [ ] MPU6050 낙상 감지 임계값 트리거
- [ ] MOSFET 차단 동작 (BHL 전원 OFF)
- [ ] Orin 전원(배터리 A)은 유지되는지

### 2-5. 카메라 / 마이크 / 스피커

```bash
# 카메라 인식
ls /dev/video*
ffplay /dev/video0   # 영상 확인

# 마이크 확인
arecord -l

# 스피커 확인
aplay -l
speaker-test -t wav
```

**확인 항목**
- [ ] 카메라 인덱스 확정 및 `configs/dev.yaml`에 기록
- [ ] 마이크 입력 레벨 정상
- [ ] 스피커 출력 정상

---

## 3단계 — 연결부 통신 확인

> 하드웨어가 살아있음을 확인한 후, 소프트웨어 인터페이스를 end-to-end로 검증한다.

### 3-1. STT → LLM → TTS 파이프라인

```bash
# jetson/cloud/groq_client.py 단독 실행 테스트
# 마이크 입력 → Groq STT → LLM → TTS 출력까지 왕복 시간 측정
python jetson/cloud/groq_client.py --test
```

**확인 항목**
- [ ] Groq API 키 동작
- [ ] STT 인식률 확인 (한국어)
- [ ] LLM 물체 이름 추출 정확도 ("빨간 컵" → "starbucks_cup")
- [ ] TTS → 스피커 출력 (`jetson/expression/speaker.py` 연동)
- [ ] 전체 왕복 TTFT < 500ms (기획서 Phase 2 게이트 조건)

### 3-2. SmolVLA 추론 → SO-ARM 동작

```bash
# jetson/arm/policy/smolvla_runner.py 단독 실행 테스트
# 카메라 입력 + 언어 지시 → SmolVLA → SO-ARM 액션
python jetson/arm/policy/smolvla_runner.py --task "starbucks_cup" --test
```

**확인 항목**
- [ ] SmolVLA 모델 로드 시간 측정
- [ ] 추론 → 액션 출력 Hz 확인 (TensorRT 적용 전/후 비교)
- [ ] SO-ARM 실제 동작 확인 (`jetson/arm/so_arm.py` 연동)
- [ ] Orin GPU 온도 모니터링 (써멀 스로틀링 확인)

### 3-3. Orin → NUC 보행 명령

```bash
# ZeroMQ 기반 메시지 전송 테스트
# Orin 측 발행 (comm/orin/sender.py)
python comm/orin/sender.py --topic gait/cmd --msg walk_forward

# NUC 수신 및 상태 응답 확인
python comm/nuc/receiver.py --topic gait/cmd
python comm/orin/receiver.py --topic gait/status
```

**확인 항목**
- [ ] `gait/cmd` 발행 → NUC 수신 → 다리 동작
- [ ] `gait/status` 수신 확인
- [ ] 명령 지연시간 측정

### 3-4. 비상정지 end-to-end

```bash
# jetson/safety/emergency_stop.py 단독 트리거 테스트
python jetson/safety/emergency_stop.py --simulate-fall
```

**확인 항목**
- [ ] 낙상 시뮬레이션 → MOSFET 차단까지 시간 측정
- [ ] EMERGENCY 상태 진입 후 Orin 로그 유지 확인
- [ ] 재시작 절차 동작 확인

---

## 4단계 — 개별 단위 프로그래밍

### 4-1. SmolVLA 파인튜닝

| 단계 | 내용 |
|------|------|
| Stage 1 | LeRobot Hub SO-100 공개 데이터로 사전 파인튜닝 |
| Stage 2 | 자체 600 에피소드로 추가 파인튜닝 |

**데이터 수집 계획**

| 물체 | 에피소드 수 | 변형 조건 |
|------|------------|----------|
| 스타벅스 컵 | 200개 | 위치 ±3cm, 회전 0/90/180° |
| 텀블러 | 200개 | 위치 ±3cm, 회전 0/90/180° |
| 하이리온 인형 | 200개 | 위치 ±3cm, 회전 0/90/180° |

**확인 항목**
- [ ] Week 2: 초기 30개 수집 → 미니 파인튜닝 → 치명적 결함 조기 발견
- [ ] Week 3~: ε2 투입, 본격 수집
- [ ] SmolVLA v1 pick-place 성공률 > 70%
- [ ] SmolVLA v2 성공률 > 85% (시연 조건 기준)

### 4-2. 상태 머신 (`jetson/state_machine/fsm.py`)

```python
class State(Enum):
    IDLE         = "idle"
    TALKING      = "talking"
    MANIPULATING = "manipulating"
    WALKING      = "walking"
    EMERGENCY    = "emergency"
```

**구현 항목**
- [ ] 각 상태 진입/퇴출 시 리소스 on/off 로직
- [ ] MANIPULATING 중 음성 입력 큐잉
- [ ] WALKING 중 SmolVLA 완전 중단
- [ ] EMERGENCY 진입 후 복구 절차

### 4-3. Walking RL (NUC)

**확인 항목**
- [ ] IsaacLab 환경 로드 (`sim/isaaclab/tasks/velocity_biped.py`)
- [ ] 상체 더미 웨이트 반영
- [ ] Sim-to-real 1차 비교 (`sim/mujoco/play_mujoco.py`)
- [ ] 지면 보행 안정화

### 4-4. 입 서보 PWM 제어 (`jetson/expression/mouth_servo.py`)

```python
# TTS 재생 중 입 서보 동기화
# 음성 amplitude에 맞춰 MG90S 각도 제어
```

**확인 항목**
- [ ] PWM 핀 설정 및 동작 확인
- [ ] TTS 오디오 amplitude → 서보 각도 매핑
- [ ] 자연스러운 입 움직임 확인

---

## 5단계 — 통합 테스트

### 부팅 자동화 (systemd)

> 진입점: `jetson/core/coordinator.py`  
> 배포 시 `scripts/deploy_jetson.sh`로 경로 자동 설정.

```bash
# /etc/systemd/system/hylion.service
[Unit]
Description=Hylion Robot Main
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/hylion/robot_project/jetson/core/coordinator.py
WorkingDirectory=/home/hylion/robot_project
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable hylion   # 부팅 시 자동 실행 등록
sudo journalctl -u hylion -f   # 실시간 로그 확인
```

### 부팅 후 자동 실행 플로우

```
Orin 전원 ON
  → systemd: coordinator.py 자동 실행
  → 하드웨어 연결 확인 (약 5초)
  → 캘리브레이션 파일 로드 (약 1초)
  → SmolVLA 모델 로드 (약 30~60초)
  → ZeroMQ 소켓 초기화 (comm/orin/)
  → IDLE 상태 진입 → 시연 대기
```

### 시나리오 통합 테스트

| 테스트 항목 | 기준 |
|------------|------|
| 음성 인식 → 물체 집기 end-to-end | 3회 연속 성공 |
| 보행 중 SmolVLA 중단/재개 | 오류 없이 전환 |
| 비상정지 → 복구 | 30초 이내 |
| 풀 시나리오 4분 연속 | 낙상 없음 |
| Orin 온도 | 30분 연속 스로틀링 없음 |

### Fallback 확인

- [ ] 네트워크 불량 시 오프라인 키워드 매칭 동작
- [ ] SmolVLA 실패 시 precoded 동작으로 전환 (`jetson/scenarios/base_scenario.py`)
- [ ] 비상정지 풀 시나리오 독립 동작 확인 (`tests/5_integration/test_emergency_scenario.py`)

---

## 담당자 매핑

| 모듈 | 담당 |
|------|------|
| SmolVLA 아키텍처 결정 | δ3 |
| 데이터 수집 (Week 2) | δ1 |
| 데이터 수집 (Week 3~) | ε2 |
| DGX 학습 실행 / TensorRT / Orin 배포 | ε1 |
| 상태 머신 / 통합 | ε1 |
| Walking RL / NUC 배포 | δ2 |
| SmolVLA 평가 / sim-to-real 분석 | ε2 |
| 하드웨어 연결 검증 총괄 | δ1 |

---

## 리스크 및 대응

| 리스크 | 발견 시점 | 대응 |
|--------|----------|------|
| SmolVLA Orin Hz 부족 | 3단계 | TensorRT INT8 적용 |
| Groq 네트워크 불안정 | 3단계 | Whisper tiny + 키워드 매칭 fallback |
| 상태 머신 전환 버그 | 5단계 | 단위 테스트 먼저, 통합 나중 |
| Orin 써멀 스로틀링 | 3단계~ | 환기 구조 확인, MediaPipe CPU 모드 유지 |
| 캘리브레이션 드리프트 | 시연 전날 | 시연 조건과 동일한 환경에서 재캘리브레이션 |
