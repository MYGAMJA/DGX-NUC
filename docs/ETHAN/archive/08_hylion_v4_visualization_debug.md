# Hylion v4 시각화/보행 디버깅 정리 (2026-04-01)

## 1) 목표
- Isaac Sim 5.1.0 GUI에서 Hylion v4를 "보이게" 하고 "안정적으로 걷게" 만드는 것.

## 2) 오늘 작업 요약
- play_hylion_511.py에서 rsl-rl 5.x 호환 문제 수정
  - ActorCritic 기반 설정을 MLPModel(actor/critic) 형식으로 변환.
  - obs_groups에 actor/critic 명시.
- 5.1.0 재생 안정화용 보정 추가
  - 랜덤화 항목 일부 비활성.
  - action NaN/Inf 방어.
- GUI 재생 프로세스 정리
  - 중복 실행 인스턴스 정리 후 최신 인스턴스 1개만 유지.
- 시각화 원인 분석
  - 축(명령 디버그)만 보이는 현상과 로봇 미표시 현상을 분리해서 점검.
  - Robot USD 선택 경로를 바꿔가며 로딩/참조 오류 확인.
- 체크포인트 교체 테스트
  - Hylion ckpt 대신 검증된 biped ckpt(model_5999.pt)로 재생 테스트.

## 3) 확인된 핵심 원인

### A. "안 보임" 원인
- 현재 5.1 변환본 USD에서 visual/collision 참조 경고가 반복됨.
- 변환 로그에서 일부 visual reference unresolved 경고가 지속됨.
- 창이 여러 개 떠 있으면 이전 창(깨진 상태)만 보고 있을 가능성이 높음.

### B. "누워 있음/걷지 않음" 원인
- Hylion v4 학습 정책 자체 품질 이슈 가능성이 큼.
  - 기존 기록에서 feet_air_time이 낮고, 보행보다 자세 유지에 가까운 패턴이 있었음.
- 리셋/초기 조건 랜덤이 크면 데모에서 계속 넘어지는 인상으로 보임.
- Hylion 형상(팔 질량/관성 포함)은 biped보다 균형 난이도가 높음.

## 4) 현재 상태
- play 스크립트는 정상 기동됨.
  - Completed setting up the environment
  - Running. Close window to exit
- 최신 실행은 biped 체크포인트 + 저속 전진 명령으로 테스트 중.
- 중복 인스턴스 정리 후 최신 실행 1개 기준으로 확인 중.

## 5) 안정적으로 걷게 하려면 (권장 순서)

### 1단계: 데모 안정화 (오늘 바로)
- 명령 속도 더 낮춤
  - lin_vel_x를 0.06~0.10 구간으로 고정.
  - lin_vel_y, ang_vel_z는 0 고정.
- 리셋 랜덤 최소화
  - pose/velocity 랜덤 0으로 유지.
- 단일 실행 창만 유지
  - play_hylion_511.py 중복 프로세스 금지.

### 2단계: 정책 안정화 (1~2일)
- Hylion 전용 재학습(권장)
  - biped warm-start 후 Hylion v4에서 12k~20k iter 파인튜닝.
- 커리큘럼 도입
  - Stage 1: 정지/균형(저속)
  - Stage 2: 저속 직진
  - Stage 3: 속도 범위 확대 + 회전
- 보상 튜닝
  - flat_orientation, base height 안정 항목 강화
  - action_rate/dof_torque 패널티 과도 여부 점검
  - feet_air_time은 과소/과대 둘 다 피하도록 중간값 재튜닝

### 3단계: 모델/자산 일관성 정리 (필수)
- 5.1 재생용 USD를 "하나의 검증된 파이프라인"으로 고정.
- 변환 후 아래 검증 자동화
  - unresolved reference 없음
  - articulation joint 이름 매칭 100%
  - env 생성 직후 30~60초 무예외

## 6) 다음 액션 제안
- A안(즉시 데모): 저속 고정(0.06) + 단일 창 + biped ckpt로 "걷는 모습" 우선 확보.
- B안(근본 해결): Hylion v4 전용 재학습을 커리큘럼으로 다시 돌려 안정 보행 정책 확보.

---

작성 목적: "왜 안 보였는지/왜 안 걸었는지"를 분리하고, 데모 우선 경로와 근본 해결 경로를 동시에 제시하기 위함.

## 7) 재학습 진행 현황 (업데이트: 2026-04-01 22:34)
- 실행 스크립트: `δ3/scripts/train_hylion.py` (`--stable_walk` 모드 추가)
- 실행 프로세스: `train_hylion.py` 실행 중 (PID 151453)
- 현재 런 디렉토리:
  - `/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-01_22-25-55`
- 파인튜닝 시작 체크포인트:
  - `biped/2026-03-27_14-36-49/model_5999.pt`
- 최근 학습 지표:
  - iteration: `6551/11999`
  - mean value loss: `0.0494`
  - mean surrogate loss: `-0.0044`
  - mean reward: `29.99`
  - mean action std: `0.43`
- 생성된 체크포인트 확인:
  - `model_6000.pt`, `model_6100.pt`, `model_6200.pt`, `model_6300.pt`, `model_6400.pt`, `model_6500.pt`

## 8) 실시간 확인 방법

### A. 1회 스냅샷 확인
```bash
bash /home/laba/project_singularity/δ3/scripts/monitor_hylion_retrain.sh
```

### B. 실시간 모니터링(2초 간격)
```bash
bash /home/laba/project_singularity/δ3/scripts/monitor_hylion_retrain.sh /home/laba/project_singularity/δ3/hylion_v4_retrain_stable.log --watch
```

### C. 핵심 지표만 빠르게 보기
```bash
grep -n "Learning iteration\|Mean value loss:\|Mean surrogate loss:\|Mean reward:\|Mean action std:" \
  /home/laba/project_singularity/δ3/hylion_v4_retrain_stable.log | tail -20
```

### D. 학습 프로세스 확인
```bash
pgrep -af "train_hylion.py"
```

### E. 체크포인트 생성 확인
```bash
ls -1 /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-01_22-25-55/model_*.pt | tail -10
```

### F. 자동 복구 가드 실행 (권장, 자리 비울 때)
```bash
nohup bash /home/laba/project_singularity/δ3/scripts/auto_guard_hylion_train.sh \
  > /home/laba/project_singularity/δ3/hylion_guard_runner.out 2>&1 &
```

가드 로그 확인:
```bash
tail -f /home/laba/project_singularity/δ3/hylion_guard.log
```

가드 중지:
```bash
pkill -f "auto_guard_hylion_train.sh"
```
