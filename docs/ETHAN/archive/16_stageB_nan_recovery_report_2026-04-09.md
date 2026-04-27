# Stage-B(v6 PhysX) NaN 재발/복구 보고서 (2026-04-09)

## 0) 2026-04-13 최신 업데이트

- 근본 원인 확인:
  - `ankle_roll` 링크에 `PhysxRigidBodyAPI`가 없어 `get_net_contact_forces()`가 0만 반환됨
- 수정 완료:
  - `physx.usda`에서 left/right `ankle_roll`에 `PhysxRigidBodyAPI` 및 `sleepThreshold=0.0` 추가
  - `contact_sensor.py`에서 list 기반 glob view 대신 body별(per-body) contact view 사용
- 현재 실행 상태:
  - `debug10`은 `physx.usda` 패치 반영 전 조건으로 실행된 런
- 다음 단계:
  - `debug10` 종료 후 패치 반영 상태로 `debug11` 재시작
  - 확인 목표: `feet_air_time > 0` (비영 값 확인)

참고: 상세 근본원인/패치 내역은 `δ3/17_stageB_contact_sensor_root_cause_2026-04-13.md`에 정리되어 있음.

## 1) 요약

- Stage-B(2048 env) 학습 중 `Mean value loss: nan`, `Mean surrogate loss: nan`가 재발했다.
- 같은 구간에서 `Mean action std: 0.00`으로 붕괴되어, 정책 탐험이 사실상 멈춘 상태가 확인됐다.
- 기존 러닝을 중단하고 NaN 이전 체크포인트(`model_7200.pt`)로 롤백 재시작했다.
- 재시작만으로도 초기 복구는 가능했으나, 다시 NaN 재발하여 PPO 안정화 튜닝을 적용했다.
- 현재는 튜닝 반영된 새 러닝으로 재기동 완료 상태다.

## 2) 관측된 문제 패턴

- NaN 시작 지점: iteration 7282 부근부터 `value loss` NaN 발생.
- 이후 `surrogate loss`도 NaN으로 전이.
- `Mean action std`가 0.00으로 고정.
- reward가 정상처럼 보여도(`+` 값 포함), 일부 구간에서 `-inf` 혹은 비정상 대수값이 혼재.

핵심 해석:
- 화면상 iteration 증가/episode length 500 유지만으로 정상 판정 불가.
- `loss NaN + action std 0.00`이면 학습 붕괴로 간주.

## 3) 수행한 조치

### A. 1차 복구

- 기존 Stage-B 프로세스 종료.
- 체크포인트 롤백 후 재시작:
  - `/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-08_11-17-20/model_7200.pt`
- 복구 로그 분리:
  - `/tmp/hylion_v6_physx_2048_recover.log`

### B. 실시간 모니터 개선

- 모니터 스크립트에 아래 항목 추가:
  - `Mean value loss`
  - `Mean surrogate loss`
  - `Mean action std`
- 수정 파일:
  - `δ3/scripts/monitor_stageb_realtime.sh`

### C. 2차 안정화 튜닝 적용

- PPO 하이퍼파라미터 완화(업데이트 강도 감소):
  - `init_noise_std`: 1.0 -> 0.6
  - `clip_param`: 0.2 -> 0.1
  - `num_learning_epochs`: 5 -> 3
  - `learning_rate`: 3e-4 -> 1e-4
  - `schedule`: adaptive -> fixed
  - `desired_kl`: 0.01 -> 0.005
  - `max_grad_norm`: 0.5 -> 0.3
- 수정 파일:
  - `δ3/hylion/agents/rsl_rl_ppo_cfg.py`

### D. 튜닝 반영 재시작

- 새 로그:
  - `/tmp/hylion_v6_physx_2048_recover_tuned.log`
- 새 실행은 초기화/롤아웃 단계까지 정상 진입 확인.

## 4) 현재 운영 경로

- 권장 모니터 실행:

```bash
bash /home/laba/project_singularity/δ3/scripts/monitor_stageb_realtime.sh /tmp/hylion_v6_physx_2048_recover_tuned.log 2
```

- 반드시 함께 보는 지표:
  - `Mean value loss` (NaN 금지)
  - `Mean surrogate loss` (NaN 금지)
  - `Mean action std` (0.00 고정 금지)
  - `Mean episode length` (500 근처 유지)

## 5) 판정 규칙 (운영 기준)

- 정상:
  - loss 유한값 유지
  - action std > 0 유지
  - NaN/Traceback 없음
- 경계:
  - loss 급등/급락 반복 + action std 급감
- 실패(즉시 개입):
  - `Mean value loss: nan` 또는 `Mean surrogate loss: nan`
  - `Mean action std: 0.00` 고정

## 6) 다음 액션

- 튜닝 런에서 최소 200~300 iteration 관찰 후 재발 여부 판정.
- 재발 시 3차 안정화:
  - learning rate 추가 하향(예: 5e-5)
  - reward clipping 또는 critic value clipping 강화
  - 필요 시 env 수 2048 -> 1024 단기 안정화 테스트
- v6_flat 전환/단계 복원 실행안은 `δ3/18_v6_flat_training_plan_2026-04-13.md` 참고.
