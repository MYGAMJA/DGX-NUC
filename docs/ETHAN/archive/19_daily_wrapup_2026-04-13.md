# 오늘 작업 정리 + 내일 실행 플로우 (2026-04-13)

## 1) 오늘 결론

1. Stage-B 학습은 중지 완료.
2. 핵심 병목은 feet_air_time이 0으로 고정되는 문제.
3. 내일은 무조건 S0 게이트부터 시작하고, 통과 시에만 M1 -> M2로 진행.

## 2) 왜 꼬였다고 느껴졌는지

1. 학습이 돌아가고 loss가 유한값이어도 feet_air_time이 0이면 보행 핵심 신호가 죽어 있을 수 있음.
2. 즉, 학습 진행 자체와 보행 신호 정상은 별개로 봐야 함.
3. 앞으로는 게이트 기반으로만 진행: S0 실패 시 장기학습 금지.

## 3) 오늘 한 핵심 작업

1. contact sensor 및 asset 근본 원인 문서화.
2. NaN 재발/복구 이력 정리.
3. v6_flat 단계 전환 전략 문서화.
4. 매트릭스 실행용 런처 반영.

## 4) 변경 파일

1. δ3/hylion/env_cfg_BG.py
- 런타임 실험 파라미터 추가
- HYLION_FEET_AIR_THRESHOLD
- HYLION_LEG_GAIN_SCALE
- HYLION_BASE_MASS_ADD_KG
2. δ3/scripts/run_v6_matrix_experiment.sh
- M1~M6 조합 실행 런처

## 5) 참조 문서

1. NaN 복구 보고서: δ3/16_stageB_nan_recovery_report_2026-04-09.md
2. contact sensor 근본원인: δ3/17_stageB_contact_sensor_root_cause_2026-04-13.md
3. v6_flat 실행 계획: δ3/18_v6_flat_training_plan_2026-04-13.md

## 6) 현재 상태 스냅샷

1. Stage-B 프로세스는 중지 상태.
2. 마지막 시도는 M1 로그 기준.
3. 관측값: loss는 유한값이었지만 feet_air_time은 0.0000 지속.

## 7) 내일 실행 순서 (반드시 이 순서)

1. S0 게이트 실행
- 100~200 iteration 스모크만 실행
- 확인 지표: value loss, surrogate loss, action std, feet_air_time
2. S0 판정
- PASS: feet_air_time 비영값 출현 + NaN 없음 + action std 붕괴 없음
- FAIL: feet_air_time 0 고정 또는 NaN/action std 붕괴
3. S0 PASS 시
- M1 200~300 iteration
- PASS면 M2로 진행
4. S0 FAIL 시
- 장기학습 금지
- contact/asset 경로 재확인 후 재시도

## 8) 실행/모니터 명령

1. 실행
bash /home/laba/project_singularity/δ3/scripts/run_v6_matrix_experiment.sh M1

2. 모니터
bash /home/laba/project_singularity/δ3/scripts/monitor_stageb_realtime.sh /tmp/hylion_v6_physx_M1.log 2

## 9) PASS/FAIL 운영 기준

1. PASS
- Mean value loss NaN 없음
- Mean surrogate loss NaN 없음
- Mean action std 0.00 고정 아님
- Episode_Reward/feet_air_time 비영값 출현
2. FAIL
- value 또는 surrogate loss NaN 발생
- action std 0.00 고정 지속
- feet_air_time 0.0000 지속

## 10) 내일 시작 전 2분 점검

1. 이전 학습 프로세스가 정말 0개인지 확인.
2. 이번 런 로그 파일명을 확정.
3. 오늘 문서 기준으로 판정할지, 18번 문서 기준으로 판정할지 하나로 고정.
4. 매트릭스 값(M3, M5, M6)이 문서와 스크립트에서 일치하는지 확인.

## 11) 메모

1. 오늘은 사용자 요청대로 학습 중단만 수행.
2. 내일은 실행보다 먼저 S0 게이트 판정부터 진행.

## 12) 2026-04-14 실제 실행 결과 (추가)

1. S0 스모크 직접 실행 완료
- 로그: /tmp/hylion_v6_physx_S0_M1_smoke.log
- 설정: num_envs=512, max_iterations=200, M1 조건(base_mass=0.0, gain=1.0, threshold=0.2)
- 결과: loss는 유한값 유지, action std도 유지
- 결과: feet_air_time 샘플 200개 전부 0.0000 (non_zero=0)

2. 결론
- S0 게이트 FAIL
- 따라서 M2 이상 매트릭스 진행 금지(워크플로우 준수)

3. 추가 진단(A/B)
- ankle 기준 단기 런: /tmp/hylion_v6_physx_S0_diag_ankle.log
- contact_sensor 디버그: ankle left/right force_max=0.0000, nonzero=0 지속
- 런타임 body glob은 ankle 2개로 정상 매핑됨

4. 자산/센서 확인 결과
- physx.usda의 left/right ankle_roll에 PhysxContactReportAPI, PhysxRigidBodyAPI, sleepThreshold=0.0 존재 확인
- contact_sensor.py의 per-body contact view 로직 존재 확인

5. 다음 액션 (실행 우선순위)
- 1순위: ankle_roll 링크의 실제 접지 충돌 상태를 시각/디버그로 재검증
- 2순위: 접촉 force가 0인 원인을 "센서 매핑"이 아닌 "실제 접촉 미발생" 가능성까지 포함해 분리
- 3순위: 원인 확정 전 M2~M6 진행 금지

## 13) 2026-04-14 원인 해결 업데이트

1. 실제 근본 원인
- 기존 δ3 자산 경로는 root_joint 때문에 사실상 fixed-base처럼 동작했고, 발이 바닥 접촉을 만들지 못함
- 그 결과 contact_sensor force=0, feet_air_time=0이 계속 유지됨

2. 해결 경로
- hylion_v6 URDF를 다시 변환해 생성한 자산 사용
- 사용 경로: /home/laba/project_singularity/δ1 & ε2/usd/hylion_v6/hylion_v6.usda
- 새 자산에서 PhysX root_joint를 disable하여 floating-base로 동작 확인

3. 검증 결과
- inspection 스크립트에서 root_z가 실제로 하강함
- step 10 이후 ankle_roll contact force가 비영값으로 확인됨
- 학습 스모크 로그에서 feet_air_time이 0.0004~0.0006으로 비영값 확인

4. 운영 결론
- 앞으로 Stage-B / M1~M6는 새 자산 경로 기준으로만 진행
- 기존 δ3 자산 경로는 더 이상 기준 자산으로 사용하지 않음

## 14) 2026-04-14 매트릭스 진행 현황 (실행 업데이트)

1. M1 실행
- 로그: /tmp/hylion_v6_physx_M1.log
- 상태: PASS(운영 기준)
- 관측: feet_air_time 비영값 유지(약 0.0019~0.0024), NaN 없음, action std 유지

2. M2 실행
- 로그: /tmp/hylion_v6_physx_M2.log
- 상태: 초기 안정 구간 확인 후 M3로 전환
- 관측: feet_air_time 비영값 유지(약 0.0012~0.0018), NaN 없음, action std 유지

3. M3 실행
- 로그: /tmp/hylion_v6_physx_M3.log
- 상태: 초기 안정 구간 확인 후 M4로 전환
- 관측: feet_air_time 비영값 유지(약 0.0013~0.0014), NaN 없음

4. M4 실행 (현재)
- 로그: /tmp/hylion_v6_physx_M4.log
- 상태: 안정 구간 확인
- 관측: feet_air_time 비영값 유지(약 0.0012~0.0014), value/surrogate 유한값, action std 유지

5. 다음 실행 순서
- M4 안정 추세 확인 후 M5 -> M6 순차 진행
- 중단 조건은 동일: NaN 발생, action std 붕괴, feet_air_time 0 고정

6. M5 실행
- 로그: /tmp/hylion_v6_physx_M5.log
- 상태: 초기 안정 구간 확인
- 관측: feet_air_time 비영값 유지(약 0.0016~0.0018), NaN 없음, action std 유지

7. M6 실행
- 로그: /tmp/hylion_v6_physx_M6.log
- 상태: 초기 정상 기동 확인
- 관측: feet_air_time 비영값 확인(0.0007 -> 0.0018), value/surrogate 유한값, action std 유지

8. 오늘 기준 결론
- M1~M6 전체에서 적어도 contact signal은 살아 있음
- 즉, 기존의 feet_air_time=0 고정 문제는 해소됨
- 다음 핵심 과제는 "contact 복구 이후 어떤 조합이 실제로 더 안정적인가"를 장기 지표로 비교하는 것

9. 주의
- 현재 M1~M6는 순차 기동 확인 위주라서, 조합 간 공정 비교를 끝낸 상태는 아님
- 다음 비교 단계에서는 조합별로 최소 200~300 iteration씩 별도로 관찰해야 함
- 비교 기준: feet_air_time 유지, NaN 없음, action std 유지, episode length 추세, reward 추세

## 15) 2026-04-14 공정 비교 결과 (300-iteration smoke suite)

비교 조건:
- 모든 조합을 동일 조건으로 실행
- iterations=300, num_envs=512
- summary: /tmp/hylion_v6_matrix_suite_summary.txt

결과:

1. M1
- value=0.0488, reward=-1.07, episode_length=17.95, action_std=0.38, feet_air_time=0.0009
- 판정: PASS

2. M2
- value=0.0996, reward=-1.19, episode_length=24.64, action_std=0.39, feet_air_time=0.0016
- 판정: PASS

3. M3
- value=0.0641, reward=-1.30, episode_length=20.31, action_std=0.38, feet_air_time=0.0014
- 판정: PASS

4. M4
- value=0.1616, reward=-1.35, episode_length=23.13, action_std=0.37, feet_air_time=0.0018
- 판정: PASS

5. M5
- value=0.0774, reward=-1.22, episode_length=19.59, action_std=0.38, feet_air_time=0.0012
- 판정: PASS

6. M6
- value=0.0529, reward=-1.28, episode_length=18.95, action_std=0.37, feet_air_time=0.0014
- 판정: PASS

요약 판단:

1. contact 문제 관점에서는 M1~M6 모두 통과
2. 장기 추적 우선 후보는 M2, M4
- M2: episode length가 가장 좋고 feet_air_time도 높음
- M4: feet_air_time이 가장 높고 episode length도 준수
3. 보조 후보는 M5
- reward/value 균형이 무난함

다음 권장 순서:

1. M2 장기 런(1024 env, 1000+ iteration)
2. M4 장기 런(동일 조건)
3. 두 로그를 비교해 최종 주력 조합 선정

## 16) 2026-04-15 장기 비교 단계 진입

1. 현재 실행 상태
- M2 장기 런 시작 완료
- 로그: /tmp/hylion_v6_physx_M2.log
- 조건: 1024 env, max_iterations=6250, 새 v6 자산 경로 사용

2. M2 초기 지표
- iteration 6000~6007 구간 확인
- value loss: 9.3381 -> 3.6513 하강
- surrogate loss: 유한값 유지
- action std: 0.30 유지
- feet_air_time: 0.0017~0.0020 유지
- NaN/Traceback 없음

3. 현재 해석
- contact 복구 이후 장기 런도 정상 진입 중
- 즉시 붕괴 신호는 아직 없음

4. 다음 순서
- M2를 먼저 충분히 관찰
- 이후 M4 장기 런 실행
- 두 장기 로그를 비교해 최종 주력 조합 선정

## 17) 2026-04-15 명령 가능 범위 정리

1. 별도 정리 문서 추가
- 파일: δ3/20_v6_command_capability_2026-04-15.md

2. 핵심 결론
- 현재 v6 Stage-B 정책은 "위치 목표 추종"이 아니라 "속도 명령 추종" 정책
- 따라서 몇 초 동안 앞으로/옆으로/회전 같은 명령은 구조상 가능
- 하지만 특정 좌표까지 이동시키려면 상위 waypoint/planner 계층이 추가로 필요

## 18) 2026-04-15 교수님 보고용 타임라인/워크플로우 문서

1. 별도 보고 문서 추가
- 파일: δ3/21_professor_report_timeline_workflow_2026-04-15.md

2. 현재 장기 런 상태 스냅샷
- M2 장기 런 진행 중
- 최근 확인값: Learning iteration 6698/12249, feet_air_time 0.0040
- NaN/Traceback 미검출

3. 운영 원칙 재확인
- M2 충분 관찰 후 M4 장기 런으로 전환
- M2 vs M4 장기 로그 비교 후 최종 주력 조합 1개 확정

## 19) 2026-04-15 장기 비교 실행 업데이트 (M2 -> M4 전환)

1. M2 장기 런 종료 전 스냅샷
- 최종 관찰 iteration: 6765/12249
- value loss: 0.0951~0.1068 (최근 구간)
- surrogate loss: -0.0022 ~ -0.0011
- action std: 1.13~1.14 유지
- feet_air_time: 0.0032~0.0043 유지
- NaN/Traceback 미검출

2. M4 장기 런 전환
- 현재 활성 프로세스: train_hylion_physx_BG.py (M4 설정)
- 로그: /tmp/hylion_v6_physx_M4.log
- 초기 확인값: iteration 6002/12249, feet_air_time 0.0019, action std 0.30
- 초기 NaN/Traceback 미검출

3. 다음 바로 할 일
- M4를 충분 구간까지 관찰
- M2/M4 비교표 작성(안정성/에피소드 길이/feet_air_time/오차 지표)
