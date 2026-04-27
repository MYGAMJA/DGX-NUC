# v6_flat 학습 전환 계획 (2026-04-13)

## 1) 목적

- 상체/팔 적재 이후 발생한 보행 불안정 원인을 분리한다.
- contact sensor 신뢰성 문제와 동역학(질량/관성) 문제를 분리 검증한다.
- 최종적으로 v6 원본으로 복귀 가능한 단계형 학습 경로를 만든다.

## 2) 핵심 가설

1. feet_air_time=0의 원인은 두 가지가 중첩될 수 있다.
2. 첫째는 contact force 측정 경로 문제다.
3. 둘째는 상체 적재로 인한 COM/관성 변화로 발 들기 자체가 어려워진 동역학 문제다.
4. 따라서 원본 v6 단일 실험으로는 원인 분리가 어렵다.

## 3) 전략 개요

1. 센서 검증 단계
2. v6_flat 프록시 학습 단계
3. 질량/충돌 복원 단계
4. 원본 v6 복귀 단계

## 4) 단계별 실행

### Stage S0: 센서 검증 (학습 전)

완료 기준:

1. ankle_roll left/right의 net contact force가 0 고정이 아니다.
2. feet_air_time term이 최소 일부 에피소드에서 0을 벗어난다.

검증 포인트:

1. PhysxRigidBodyAPI 존재
2. sleepThreshold=0.0 적용
3. per-body contact view 사용 상태 확인

실행 체크리스트:

1. ankle_roll left/right에 `PhysxRigidBodyAPI`와 `PhysxContactReportAPI`가 동시에 존재하는지 확인
2. ankle_roll left/right에 `physxRigidBody:sleepThreshold = 0.0` 적용 여부 확인
3. contact sensor가 list glob이 아닌 body별 view를 사용 중인지 확인
4. 100 iteration 스모크에서 `Episode_Reward/feet_air_time` 샘플 중 비영값 존재 여부 확인

### Stage S1: v6_flat 프록시 학습

목표:

1. 빠른 안정 수렴
2. loss NaN 재발 방지
3. feet_air_time > 0 재현

권장 구성:

1. 다리 12자유도는 동일 유지
2. 상체/팔은 단일 고정 질량으로 base에 합산하거나 충돌 단순화
3. 관성 텐서는 과도하게 크지 않도록 제한

USD 구성 체크리스트 (무엇을 남기고/고정할지):

1. 남길 것
2. `leg_*` 링크/조인트 체인(hip-roll, hip-yaw, hip-pitch, knee, ankle-pitch, ankle-roll)
3. 발 접촉 판단에 쓰는 링크의 rigid body/contact report 스키마
4. 학습에 필요한 루트 articulation 구조
5. 고정할 것
6. 팔 전체 조인트: locked 또는 고강성 고정
7. 머리/상체의 불필요 자유도: base 기준 고정
8. 상체 추가 질량은 base 단일 등가 질량으로 환산
9. 단순화할 것
10. 상체/팔 복잡 충돌 메시: 단순 collider(박스/캡슐)로 대체
11. 미사용 시각 요소/인스턴싱 payload: 학습 자산에서 제외
12. 유지할 것
13. 다리 조인트 이름과 순서(정책 액션 인덱스 호환 목적)
14. 접지/마찰 관련 물리 파라미터 범위

판정 기준:

1. Mean value loss, Mean surrogate loss가 유한값 유지
2. Mean action std > 0 유지
3. feet_air_time이 0 고정에서 벗어남

### Stage S2: 점진 복원 (A/B)

순서:

1. 상체 질량 30%
2. 상체 질량 60%
3. 상체 질량 100%
4. 팔 충돌/관성 복원

각 단계 공통 통과 조건:

1. NaN 없음
2. action std 붕괴 없음
3. feet_air_time 비영값 유지
4. episode length 추세 유지

### Stage S3: 원본 v6 복귀

1. S2를 모두 통과하면 원본 v6로 동일 PPO 설정 이관
2. 재발 시 S2 직전 안정 단계로 롤백

## 5) 실험 매트릭스 (최소)

축 A: 상체 질량 스케일

1. 0.0
2. 0.3
3. 0.6
4. 1.0

축 B: 액추에이터 토크/강성 스케일 (다리)

1. 1.0
2. 1.2
3. 1.4

축 C: feet_air_time threshold

1. 0.4 (기본)
2. 0.2
3. 0.1

최소 실행 세트:

1. A x B에서 6개 조합 우선
2. 각 조합 200~300 iteration 스모크
3. 통과 조합만 장기 학습 전환

우선 실행 6개 조합 (권장 순서):

1. M1: A=0.0, B=1.0, C=0.2
2. M2: A=0.0, B=1.2, C=0.2
3. M3: A=0.3, B=1.0, C=0.2
4. M4: A=0.3, B=1.2, C=0.2
5. M5: A=0.6, B=1.2, C=0.2
6. M6: A=1.0, B=1.2, C=0.2

보조 조합 (M1~M6 실패 시):

1. threshold 완화: C=0.1로 재시도
2. 토크 추가: B=1.4 단기 스모크(100 iteration)

## 6) 운영 체크리스트

1. 시작 전: 이전 프로세스 종료 확인
2. 로그 분리: 조합별 별도 /tmp 로그 파일 사용
3. 모니터: value/surrogate loss + action std + feet_air_time 동시 확인
4. 실패 즉시 중단: NaN 또는 action std 0.00 고정

추가 중단 조건:

1. reward가 `-inf` 또는 비정상 대수값 반복
2. 150 iteration 연속 feet_air_time=0.0000

## 7) 즉시 실행 권장 순서

1. 현재 debug11에서 feet_air_time 비영값 확인 시점까지 100~200 iteration 추가 관찰
2. 비영값이 끝까지 안 나오면 v6_flat 자산으로 Stage S1 즉시 전환
3. S1 성공 후 S2 질량 복원 A/B 실행

단계 전환 게이트:

1. S0 -> S1: 센서 비영값 1회 이상 확인
2. S1 -> S2: 아래 성공 판정 기준 충족
3. S2 -> S3: A=1.0에서 300 iteration 안정 유지

## 8) 성공 판정 기준 (숫자 기준)

PASS(조합 통과):

1. 최근 100 iteration 동안 `Mean value loss` NaN 0회
2. 최근 100 iteration 동안 `Mean surrogate loss` NaN 0회
3. 최근 100 iteration 평균 `Mean action std >= 0.10`
4. 최근 100 iteration 중 `Episode_Reward/feet_air_time > 0.0001` 비율 >= 20%
5. `Mean episode length`가 붕괴 추세가 아님(급락 지속 없음)

FAIL(조합 탈락):

1. NaN 1회라도 발생
2. action std 0.00 고정이 20 iteration 이상 지속
3. feet_air_time=0.0000가 150 iteration 연속
4. reward `-inf` 재발

## 9) 결과 기록 템플릿

```text
[실험 ID]
- 날짜/시간:
- 자산 버전: v6_flat / v6
- 질량 스케일:
- 토크/강성 스케일:
- feet_air_time threshold:
- iter 범위:
- value loss:
- surrogate loss:
- action std:
- feet_air_time:
- 판정: PASS / FAIL
- 비고:
```
