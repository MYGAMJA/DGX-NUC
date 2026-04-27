# 데모 시각화 가이드 (2026-04-15, 업데이트 2026-04-16)

## 현재 학습 상태

| 단계 | 설명 | 상태 |
|------|------|------|
| Stage-A | BHL biped (다리만) 6,000 iter | ✅ 완료 |
| Stage-B | Hylion v6, 12,000 iter, pretrained from A | ✅ 완료 (0.5 m/s 100%, 0.7 m/s 75%) |
| Stage-C | Stage-B resume, perturbation interval ±10N, vx 0.7 m/s, standing 10% | 🔄 진행 중 (`/tmp/hylion_v6_stageC_opt.log`) |

Stage-C 모니터링:
```bash
grep -E "Iteration:|Mean reward|ETA:|base_orientation|track_lin_vel" /tmp/hylion_v6_stageC_opt.log | tail -10
```

### Stage-C 주요 파라미터 (논문 기반 최적화, 2026-04-16)

| 항목 | Stage-B | Stage-C (최적화) | 근거 논문 |
|------|---------|-----------------|-----------|
| PPO clip | 0.10 | **0.15** | H2O (CMU 2024) |
| learning rate | 5e-5 | **1e-4** | H2O, BHL |
| entropy coef | 0.005 | **0.01** | BHL 표준 |
| num epochs | 2 | **5** | RSL-RL 표준 |
| LR schedule | fixed | **adaptive** | BHL |
| grad norm | 0.05 | **0.1** | RSL-RL 표준 |
| feet_air_time wt | 3.0 | **1.5** | Walk These Ways (MIT 2022) |
| ext. force | 없음 | **±10N, interval** | Humanoid Parkour (CoRL 2024) |
| force mode | reset | **interval 1.5~3s** | 버그 수정 (reset 직후 충격 제거) |
| standing 비율 | 2% | **10%** | H2O 팔 조작 시나리오 |
| base_mass 랜덤 | 없음 | **-0.3~+1.5 kg** | 팔 하중 변동 시뮬레이션 |
| max vx | 0.5 | **0.7 m/s** | - |

**초기 결과 (1 iter 후):**
- `track_lin_vel: 0.22` (이전 잘못된 Stage-C: 0.01)
- `Mean reward: +1.1` (이전: -2.3)
- ETA: ~5.4시간

---

## 실행 명령 (현재 베스트 체크포인트)

```bash
cd /home/laba/project_singularity
source /home/laba/env_isaaclab/bin/activate

DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
  python δ3/scripts/play_hylion_v6_BG.py \
  --num_envs 4 --viz kit \
  --ckpt_path δ3/checkpoints/stage_b_hylion_v6/best.pt \
  --lin_vel_x 0.3 --base_mass_add_kg 0.0 \
  --leg_gain_scale 1.2 --feet_air_threshold 0.2 \
  --stand_sec 2 --warmup_sec 3
```

- `--viz kit` : DGX 모니터에 Isaac Sim GUI 창 표시 (필수)
- `DISPLAY=:1` : DGX에 연결된 모니터 (`:1` 고정)
- 포그라운드로 실행해야 창이 뜸 (`&` 붙이면 안됨)

---

## 체크포인트 정리

```
δ3/checkpoints/
  stage_a_biped/best.pt         → .../biped/2026-04-06_15-27-27/model_5999.pt
  stage_b_hylion_v6/best.pt     → .../hylion/2026-04-15_13-36-12/model_11998.pt  ← 현재 베스트
  stage_c_hylion_v6/best.pt     → (Stage-C 완료 후 생성 예정)
```

Stage-B 성능 (11998 iter 기준):
- 성공률: **vx=0.3 → 100%**, **vx=0.5 → 100%**, **vx=0.7 → 75%**
- episode_length: **478 / 500 steps**
- track_lin_vel_xy: **0.85**

---

## Stage-C 실패 시 다음 프로세스

### Stage-C 실패 판단 기준
- 500 iter 이후에도 `base_orientation` 종료율 > 80% 지속
- `track_lin_vel_xy_exp` < 0.05 (Stage-B 정착 수준 회복 못 함)
- `Mean reward` 가 계속 음수 (-1.0 이하)

### 실패 시 선택지 (우선순위 순)

| 옵션 | 방법 | 예상 시간 |
|------|------|-----------|
| **C-1** | perturbation 끄고 vx=0.7만 먼저 학습 후 perturbation 추가 커리큘럼 | 10~12시간 |
| **C-2** | Stage-B에서 더 학습 (20k iter), perturbation 없이 속도 범위만 확장 | 5~6시간 |
| **C-3** | force range 줄임 (±5N), standing_ratio 줄임 (5%) 후 재시도 | 5~6시간 |
| **C-4** | 현재 Stage-B 그대로 하드웨어 sim-to-real 시도 | 즉시 |

**추천**: C-3 (파라미터만 줄여서 재시도가 빠름)

### Stage-C 성공 후 다음 단계
1. `stage_c_hylion_v6/best.pt` 심링크 생성
2. 성능 평가: vx = 0.3/0.5/0.7/0.8 m/s 각각 32 env
3. push 테스트 (play 스크립트에서 외력 추가 확인)
4. 하드웨어 배포 준비 (Hylion → ROS 인터페이스)

---

## 팔 작동 중 직립 균형 가능 여부

### 시나리오 (Hylion 목표 행동)
```
걷다가 멈춤 (lin_vel_x = 0)
  → 서서 SO-ARM으로 물건 집기
  → 다시 걷기
```

### 현재 아키텍처 이해

```
정책 관찰 (45차원)
  ├─ base IMU (roll/pitch/yaw rate, gravity vector)  ← 팔 움직임 시 CoM 이동 감지
  ├─ command (vx, vy, wz)
  ├─ 다리 관절 위치/속도 (12개)
  └─ 이전 action (12개)

팔(SO-ARM 10 DOF) → 관찰에 없음, 별도 ROS 제어
```

### 단계별 균형 가능 여부

| 단계 | 가능 여부 | 조건 |
|------|-----------|------|
| **현재 Stage-B** | 부분 가능 | vx=0 명령 + 팔 천천히 움직임 |
| **Stage-C 완료 후** | 가능 | base_mass 랜덤화로 팔 하중 변동 학습 |
| **근본 해결 (미래)** | 안정적 | SO-ARM 관절 상태를 관찰에 추가 → 재학습 |

### 왜 균형이 가능한가?
- `HYLION_BASE_MASS_ADD_KG = -0.3~+1.5 kg` 랜덤화 → 팔 자세 변화에 따른 상체 질량 이동 시뮬레이션
- `rel_standing_envs = 10%` → 10%는 vx=0 제자리 서기 시나리오로 학습
- `base_external_force_torque interval` → 팔 가속도가 만드는 CoM 이동 충격 모델링
- IMU가 tilt 감지 → 다리 정책이 즉각 보상

### 팔 움직임 크기별 예측

| 팔 동작 | 예상 안정성 |
|---------|------------|
| 천천히 전방 뻗기 | ✅ Stage-B에서도 가능 |
| 물건 집기 (~0.5 kg) | ✅ Stage-C 후 가능 |
| 빠른 휩쓸기 동작 | ⚠️ Stage-C 후 50~70%, 추가 학습 필요 |
| 1 kg 이상 물건 | ❌ CoM 이동 너무 큼, 재학습 필요 |

### 실제 구현 시나리오

```python
# 하드웨어에서의 순서
1. lin_vel_x = 0 명령 전송 → 로봇 멈춤
2. 2초 대기 (직립 안정화, warmup_sec=2 참고)
3. SO-ARM 제어 시작 (별도 ROS 노드)
4. SO-ARM 완료 후 lin_vel_x = 0.3 재개
```


  ├─ base IMU (roll/pitch/yaw rate, gravity vector)  ← 팔 움직임에 반응 가능
  ├─ command (vx, vy, wz)
  ├─ 다리 관절 위치/속도 (12개)
  └─ 이전 action (12개)

팔(SO-ARM 10 DOF) → 관찰에 없음, 별도 제어
```

### 서서 팔 움직일 때 균형 가능한가?

**단기 (현재 정책 그대로)**: 부분적으로 가능
- command = (0, 0, 0)으로 제자리 서기 명령
- SO-ARM이 움직여 CoM이 이동 → IMU가 tilt 감지 → 다리 정책이 반응
- 단, 팔 움직임에 노출되지 않아 큰 하중(물건 집기) 시 넘어질 수 있음

**중기 (학습 개선)**: 안정적
- `base_mass_add_kg` 랜덤화 켜기 (팔 하중 변동 시뮬레이션)  
- 코드: `env_cfg_BG.py`에서 `HYLION_BASE_MASS_ADD_KG` 를 `0.0~1.5` 범위로

**장기 (근본 해결)**: 팔+다리 통합 제어
- SO-ARM 관절 상태를 관찰에 포함 → 정책이 팔 자세를 인식하고 다리로 보상
- 구현 비용 높음 (재학습 필요)

### 실제 시나리오

```
로봇 걷는 중
  → lin_vel_x = 0 명령 (멈춤)
  → warmup 1~2초 (직립 안정화)
  → SO-ARM 작동 (물건 집기)
  → 완료 후 다시 걷기 명령
```

이 방식이면 현재 Stage-B 정책으로도 동작 가능.  
**팔이 빠르게 움직이거나 무거운 물건을 들 때**는 base_mass 랜덤화 재학습 권장.

---

## 트러블슈팅

### 창이 안 뜰 때
- `--viz kit` 빠뜨린 경우 → headless로 실행됨
- `DISPLAY=:1` 없으면 X 연결 실패
- `XAUTHORITY=/run/user/1000/gdm/Xauthority` 필요할 수 있음 (background 실행 시)
- **포그라운드로 직접 실행해야 창이 뜸** (`&` 붙이면 안됨)

### 많이 넘어질 때
- `vx=0.5 m/s` 이하 사용 권장 (100% 성공률 구간)
- `--num_envs 4` 로 줄이면 각 로봇이 더 잘 보임
- action clamp ±1.0 하면 절대 안됨 — action_scale=0.25라 raw output이 ±2 이상 나옴

### `--checkpoint` 인자 충돌 에러
- `cli_args`가 이미 `--checkpoint`를 등록함
- 스크립트에서 `--ckpt_path`로 대신 사용

---

## 커맨드 변경 방법

`play_hylion_v6_BG.py` CLI 인자 사용:

```bash
# 제자리 서기 → 작동 확인
python play_hylion_v6_BG.py --lin_vel_x 0.0 --stand_sec 5

# 직진 빠르게
python play_hylion_v6_BG.py --lin_vel_x 0.5

# 느리게 (데모용)
python play_hylion_v6_BG.py --lin_vel_x 0.3
```

학습 커맨드 범위 (Stage-B 기준, 이 범위 내에서 설정):
- lin_vel_x: -0.5 ~ 0.5 m/s
- lin_vel_y: -0.25 ~ 0.25 m/s
- ang_vel_z: -1.0 ~ 1.0 rad/s
