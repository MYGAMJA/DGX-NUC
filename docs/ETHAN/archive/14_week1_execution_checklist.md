# Week 1 실행 체크리스트 (δ3 Walking RL)

이 문서는 지금 해야 할 작업만 빠르게 체크하기 위한 실행용 리스트입니다.

## 0) 현재 상태 확인 (시작 전 1회)

- [ ] `Stage-A (biped PhysX)` 프로세스가 실행 중인지 확인
- [ ] `/tmp/bhl_biped_physx_stageA.log` 파일이 계속 갱신되는지 확인
- [ ] `/tmp/physx_pipeline.log`에 진행 로그가 15초 간격으로 찍히는지 확인

확인 명령:

```bash
pgrep -af "train_hylion_physx_BG.py --task Velocity-Berkeley-Humanoid-Lite-Biped-v0"
tail -n 20 /tmp/bhl_biped_physx_stageA.log
tail -n 20 /tmp/physx_pipeline.log
```

## 1) Stage-A (다리-only PhysX) 진행 중 체크

- [ ] `Learning iteration`이 계속 증가한다
- [ ] `Mean reward`가 급락/고정 없이 학습 추세를 보인다
- [ ] `Mean episode length`가 전반적으로 증가 추세를 보인다
- [ ] 로그에 `Traceback`, `contains NaN values`, `Mean value loss: nan`, `Mean surrogate loss: nan`가 없다

실시간 확인 명령:

```bash
watch -n 2 -- bash -lc '
echo "=== Stage-A (biped physx) ==="
pgrep -af "train_hylion_physx_BG.py --task Velocity-Berkeley-Humanoid-Lite-Biped-v0" || echo "not running"
echo
grep -nE "Learning iteration|Mean reward:|Mean episode length:|Traceback|contains NaN values|Mean value loss: nan|Mean surrogate loss: nan" /tmp/bhl_biped_physx_stageA.log | tail -n 12 || true
echo
echo "=== Pipeline ==="
tail -n 8 /tmp/physx_pipeline.log || true
'
```

## 2) Stage-A 완료 조건

- [ ] `Stage-A` 프로세스가 정상 종료됨
- [ ] run 디렉토리에 `model_*.pt` 체크포인트가 생성됨
- [ ] 마지막 체크포인트 경로를 확보함

확인 명령:

```bash
run_dir=$(grep -oE "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/biped/[0-9_-]+" /tmp/bhl_biped_physx_stageA.log | tail -1)
echo "$run_dir"
ls -1 "$run_dir"/model_*.pt | tail -n 10
```

## 3) Stage-B (v6 PhysX) 자동 연결 확인

- [ ] 파이프라인이 Stage-A 완료를 감지함
- [ ] `Stage-B` 프로세스가 시작됨 (`Velocity-Hylion-BG-v0`)
- [ ] `/tmp/hylion_v6_physx_stageB.log`가 생성되고 iteration이 증가함

확인 명령:

```bash
tail -n 30 /tmp/physx_pipeline.log
pgrep -af "train_hylion_physx_BG.py --task Velocity-Hylion-BG-v0"
grep -nE "Learning iteration|Mean reward:|Mean episode length:|Traceback|contains NaN values" /tmp/hylion_v6_physx_stageB.log | tail -n 20
```

## 4) 실패 시 즉시 대응

- [ ] `Traceback` 또는 `NaN` 발생 시 로그 마지막 100줄 저장
- [ ] 어떤 Stage에서 실패했는지 구분 (`Stage-A` / `Stage-B`)
- [ ] 실패 직전 체크포인트 경로를 기록

수집 명령:

```bash
tail -n 100 /tmp/bhl_biped_physx_stageA.log > /tmp/stageA_error_tail.log
[ -f /tmp/hylion_v6_physx_stageB.log ] && tail -n 100 /tmp/hylion_v6_physx_stageB.log > /tmp/stageB_error_tail.log
```

## 5) 오늘 종료 전 체크

- [ ] Stage-A 혹은 Stage-B의 최신 `model_*.pt` 경로를 기록
- [ ] 최신 로그 파일 2개 경로를 기록
- [ ] 다음 실행 시작점(어느 체크포인트에서 재개할지) 메모

기록 템플릿:

```text
[날짜/시간]
- Stage-A latest ckpt:
- Stage-B latest ckpt:
- Stage-A log: /tmp/bhl_biped_physx_stageA.log
- Stage-B log: /tmp/hylion_v6_physx_stageB.log
- Next action:
```

## 6) Week 1 실행계획 정리본 (중복 제거)

### 목표

- [ ] Week 1에 DGX에서 Walking RL 1차 학습을 시작한다
- [ ] SmolVLA Stage 1과 DGX GPU 시간을 사전 조율한다

### 작업 항목

- [ ] δ3 IsaacLab 환경 인계 확인
- [ ] 커스텀 상부 포함 여부 확인: torso mesh + head mass + SO-ARM 간략 모델
- [ ] Walking RL 보상 함수 초기값 확정: 직립 유지, 전진 속도, 에너지 효율
- [ ] DGX에서 학습 시작 및 프로세스/로그 확인
- [ ] 보상 커브와 episode length 추이 모니터링

### 선행 완료 항목 (표기만)

- [x] AMASS · BONES-SEED retargeting 완료

### Stage 2 준비 항목

- [ ] 30개 샘플 미니 파인튜닝 실행 (DGX 낮 슬롯)
- [ ] 미니 파인튜닝 결과로 Stage 2 테스트 적합성 확인

### DGX 스케줄 표 (Week 1)

```text
[날짜] ________
- 09:00-13:00 : SmolVLA Stage 1
- 13:00-19:00 : Walking RL (δ3)
- 19:00-24:00 : Walking RL 연속 또는 체크포인트 평가

[충돌 발생 시 우선순위]
1) 이미 실행 중인 장시간 학습 유지
2) 나머지 작업은 다음 슬롯으로 이월
3) 슬롯 변경 시 팀 채널에 즉시 공유
```
