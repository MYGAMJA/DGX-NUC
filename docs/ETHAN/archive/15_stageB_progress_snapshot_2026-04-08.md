# Stage-B(v6 PhysX) 진행 요약 (2026-04-08)

## 1) 목표

- Stage-A(BHL biped) 정책을 기반으로 Stage-B(Hylion v6 PhysX) 학습을 안정적으로 진행한다.
- 시각화 없이 로그 지표(Iteration, Reward, Episode Length, Error Marker)로 상태를 판정한다.
- 사용자 실시간 모니터링 루틴을 고정한다.

## 2) 지금까지 완료한 작업

- [x] Stage-B contact sensor 초기화 실패 문제 해결
- [x] contact sensor prim/body 매칭 로직 및 경로 정합 개선
- [x] reset/randomization 강도 완화로 NaN 발생 구간 제거
- [x] 초기 자세 안정화(초기 base 높이 상향 포함)
- [x] Stage-B 학습 루프 진입 및 지속 학습 확인
- [x] 4096/3072 환경에서 PhysX 용량 경고 확인 후 2048-env 안정 운용으로 전환
- [x] 실시간 모니터 스크립트 제공

## 3) 핵심 수정 파일

- `δ3/hylion/env_cfg_BG.py`
  - contact sensor 설정 정리
  - 과도한 랜덤화 일부 비활성/완화
  - reset 범위 안정화
- `δ3/hylion/robot_cfg_BG.py`
  - Linux 기준 USD 경로 정리
  - 초기 base z 상향
- `δ3/scripts/train_hylion_physx_BG.py`
  - pretrained checkpoint 로드 경로/옵션 정리
  - 디버그 옵션 추가(NaN 관측용)
- `IsaacLab/.../contact_sensor/contact_sensor.py`
  - nested hierarchy 대응 매칭 로직 개선
- `δ3/scripts/monitor_stageb_realtime.sh`
  - 로그/프로세스/에러/capacity 경고 실시간 모니터링

## 4) 현재 상태(스냅샷)

- 최근 관측:
  - Learning iteration: 6990/17999
  - Mean episode length: 500.00 (유지)
  - Mean reward: 최근 구간 음수(예: -5.33 ~ -1.74)
- 해석:
  - Episode length 500 유지 + NaN/Traceback 부재이면 학습 붕괴로 보기 어렵다.
  - 현재 음수 reward는 보상항 가중치 합(패널티 우세) 영향이 큰 상태로 판단.

## 5) 운영 기준(실무 판정선)

- 정상 진행:
  - Iteration이 계속 증가
  - Mean episode length가 500 근처 유지
  - NaN/Traceback 없음
- 경계 상태:
  - Reward 음수가 수백 iteration 이상 과도하게 악화
  - Episode length 하락 동반
- 즉시 개입:
  - `contains NaN values`, `Traceback`, `Mean value loss: nan`, `Mean surrogate loss: nan` 발생

## 6) 실시간 확인 명령

```bash
bash /home/laba/project_singularity/δ3/scripts/monitor_stageb_realtime.sh
```

옵션(로그 파일 지정):

```bash
bash /home/laba/project_singularity/δ3/scripts/monitor_stageb_realtime.sh /tmp/hylion_v6_physx_2048_stable.log 2
```

## 7) 주요 로그/산출물 경로

- Stage-B 안정 로그: `/tmp/hylion_v6_physx_2048_stable.log`
- Stage-B 기본 로그(파이프라인): `/tmp/hylion_v6_physx_stageB.log`
- Stage-A(Newton) 로그: `/tmp/bhl_biped_newton_stageA.log`
- 실시간 모니터 스크립트: `/home/laba/project_singularity/δ3/scripts/monitor_stageb_realtime.sh`

## 8) 다음 액션

- 2048-env 장기 러닝 유지 후 reward/length 추세 재평가
- 필요 시 reward 항 가중치 미세조정으로 평균 reward를 0 부근으로 이동
- PhysX 용량 튜닝 여건이 되면 3072 이상 재상향 테스트
