# 학습 계획 — ±50N robustness 달성 (2026-04-30 v2) — **폐기됨 (2026-05-02)**

> ⚠ **이 plan은 폐기되었습니다.**  
> 사유: ±50N robustness가 발표 시나리오와 mismatch (보도블록 평지 + 정지 작업).  
> 학습 4번 시도 모두 실패 (catastrophic forgetting). 실 effort_limit 6Nm로 ±50N 회복 물리적 불가.  
>
> **현재 결정:** D4 정책 그대로 발표 사용. [33_d4_validation_report_2026-05-02.md](33_d4_validation_report_2026-05-02.md) 참고.  
> **다음 단계:** [34_sim2real_plan_2026-05-02.md](34_sim2real_plan_2026-05-02.md) (System ID + Sim2Real)
>
> ─── 아래는 폐기된 원본 plan (참고용) ───

---

> 마감: 2026-06-01 (발표) | 작성: 2026-04-30 | v2: 게이트 강화 + 데이터 기반 점프 폭 조정  
> 이전 plan 31 (D3→E4)은 archive로 이동.

---

## 1. 목표

다음 두 조건을 **모두** 만족하는 정책 확보:

1. **IsaacLab에서 ±50N 외란 환경에서 안정적으로 학습**
   - 마지막 200 iter 평균 fall % < 15%
   - 마지막 1000 iter fall % 기울기 ≤ 0 (감소 추세)
   - 마지막 200 iter 평균 reward > 8
2. **MuJoCo sim-to-sim에서 보행 가능**
   - 무외란 vx=0.3, ≥ 230/250 step
   - 외란 모드 ±50N, ≥ 5초 보행 유지

---

## 2. 데이터 분석 — 이전 학습 실패 정량 진단

### 2-1. 단계별 학습 추이

| Stage | 외란 | 시작 reward | 마지막 reward | 마지막 fall % | 학습 평가 |
|-------|----:|:----------:|:-------------:|:-------------:|:---------|
| D4 (PhysX) | ±5N | — | 20.20 | **13%** | ✅ 베이스라인 |
| D4.5 (Newton) | ±7N | 13.01 | 14.29 | 26% | ⚠ 미수렴 진행 |
| D5 (Newton) | ±10N | 6.32 | 11.36 | **34%** | ⚠ 미수렴 진행 (임계점) |
| E1 (Newton) | ±15N | 2.35 | 3.69 | 80% | ❌ robustness 못 배움 |
| E2 (Newton) | ±20N | 0.38 | 0.16 | 100% | ❌ 학습 신호 끊김 |
| E3, E4 | ±25, 30N | 음수 | 음수 | 100% | ❌ 무의미한 iter |

### 2-2. 진짜 원인 (수정 진단)

이전 plan의 진단(+5N 점프가 문제)은 **부분적으로만 맞음**. 데이터 보면:

- D4 → D4.5 (+2N, +40%): Newton 백엔드 충격 흡수 + 외란 적응 → 정상 학습
- D4.5 → D5 (+3N, +43%): 잘 학습됨 (reward +80%, fall 51→34%)
- D5 → E1 (+5N, +50%): 처음부터 fall 72%로 시작 → 5000 iter는 robustness 회복엔 부족
- E1 → E2 (+5N, +33%): **이미 미수렴 상태에서 외란 추가 → 즉시 100% fall → 학습 신호 끊김**

→ 진짜 원인은:
1. **각 단계가 fall < 15% 베이스라인까지 수렴 못 한 채 다음 진행** (누적 약화)
2. **iter 수 부족** (D5 5000 iter는 ±10N 회복엔 부족)
3. **fall 100% 도달 시 PPO 학습 신호 끊김** (모든 episode가 동일 결과)

### 2-3. 도출된 게이트 기준

D4의 베이스라인(fall 13%) 회복을 원칙으로:

```
PASS 조건 (4개 중 최소 3개 만족):
  (1) 마지막 200 iter 평균 fall %  <  15%
  (2) 마지막 1000 iter fall % 기울기  ≤  0  (감소 추세)
  (3) 마지막 200 iter 평균 reward  >  8
  (4) MuJoCo 무외란 보행 vx=0.3, ≥ 230/250 step

FAIL 시 처리:
  - 같은 외란으로 +2000 iter 추가 학습 (1회)
  - 그래도 FAIL → 보강 단계 삽입 (외란 +1.5N 만 증가)
  - 보강 단계도 FAIL → 이전 단계로 롤백 + 더 작은 증분
```

---

## 3. 전체 로드맵

```
[기준점] D4 (±5N, fall 13%) ← 외란 환경 검증된 안정 정책
   ↓
[Phase 1] 인프라 보강 (0.5일)
   ├─ orchestrator: PASS 조건 4개 자동 평가, FAIL 시 +2000 iter 또는 중단
   ├─ PPO config: epochs=2, entropy_coef=0.012 (E 단계 전체)
   ├─ MuJoCo 자동 게이트 (run_stage 직후)
   └─ 매 스테이지 후 README/sim2sim_progress.md 자동 갱신 hook
   ↓
[Phase 2] D4 → ±30N 안정 학습 (~5일)
   D4(±5)→D4.5(±7)→D5(±10)→E0.3(±12)→E0.6(±13.5)→E1(±15)
   →E1.4(±17)→E1.8(±19)→E2.3(±21)→E2.7(±23)→E3(±25)→E3.5(±27.5)→E4(±30)
   ↓
[Phase 3] ±30N → ±50N 확장 (~3일)
   E4(±30)→E5(±33)→E5.5(±36)→E6(±39)→E6.5(±42)→E7(±45)→E7.5(±47.5)→E8(±50)
   ↓
[Phase 4] 최종 검증 (0.5일)
   ├─ MuJoCo 무외란 vx=0.0, 0.3, 0.5
   ├─ MuJoCo 외란 모드 ±50N (5초 이상 유지)
   ├─ effort_limit 한계 확인 (필요 시 phase 3 재학습)
   └─ 발표용 영상 캡처
```

총 예상 기간: **~9일** (마감까지 30일 여유 → 충분한 마진)

---

## 4. 단계별 상세 (Phase 2)

### 4-1. 진행 표

각 스테이지: **iter 한도 8000~10000** (이전 5000은 부족했음).  
`HYLION_PERTURB_FORCE` 환경변수로 외란 조정.

| 스테이지 | 외력 | torque | mass+ | 시작점 | iter 한도 | 증분 비율 | 비고 |
|---------|----:|-------:|------:|--------|---------:|:------:|-----|
| **D4.5** | ±7 N | 2.0 | 0.75 | D4 | 8000 | +40% | 재학습 (이전 26%→15% 목표) |
| **D5** | ±10 N | 3.0 | 1.0 | D4.5 | 10000 | +43% | 재학습 (이전 34%→15% 목표) |
| **E0.3** | ±12 N | 3.5 | 1.0 | D5 | 8000 | +20% | 신규 |
| **E0.6** | ±13.5 N | 4.0 | 1.25 | E0.3 | 8000 | +13% | 신규 |
| **E1** | ±15 N | 4.5 | 1.25 | E0.6 | 8000 | +11% | 재학습 |
| **E1.4** | ±17 N | 5.0 | 1.5 | E1 | 8000 | +13% | 신규 |
| **E1.8** | ±19 N | 5.5 | 1.5 | E1.4 | 8000 | +12% | 신규 |
| **E2.3** | ±21 N | 5.5 | 1.5 | E1.8 | 8000 | +11% | 신규 |
| **E2.7** | ±23 N | 6.0 | 2.0 | E2.3 | 8000 | +10% | 신규 |
| **E3** | ±25 N | 6.5 | 2.0 | E2.7 | 8000 | +9% | 재학습 |
| **E3.5** | ±27.5 N | 7.0 | 2.0 | E3 | 8000 | +10% | 신규 |
| **E4** | ±30 N | 8.0 | 2.0 | E3.5 | 10000 | +9% | 재학습 |

각 스테이지별 ETA ~8시간 (Newton 8000 step/s 기준).  
**Phase 2 총 예상**: 12 스테이지 × 평균 8.5시간 = **102시간 ≈ 4.3일**.

### 4-2. 진입 조건과 시작 명령

**시작 명령 (D4 → D4.5 재학습):**
```bash
START_STAGE=D4_5 \
  D4_5_CKPT=/home/laba/DGX-NUC/checkpoints/biped/stage_d4_hylion_v6/best.pt \
  bash /home/laba/DGX-NUC/sim/isaaclab/scripts/newton/run_newton_training.sh
```

이전 D5 체크포인트는 fall 34%로 미수렴 → **D4.5부터 재학습 필수**. E1~E4 체크포인트는 (앞서 분석대로) 사실상 학습 못 됐으므로 재학습 시 PhysX D4 → Newton D4.5 chain만 살리고 나머지 재생성.

---

## 5. 단계별 상세 (Phase 3 — ±30N → ±50N)

### 5-1. 물리 한계 주의

**effort_limit=6 Nm로는 ±50N 외란 정적 회복 불가.**

19.89 kg 로봇, base 높이 ~0.6 m → ±50N 측방 push 시 ankle restoration 모멘트 ≈ 30 Nm (양 ankle 분담 시 15 Nm). **현재 6 Nm로는 7~8N 외란이 정적 한계**.

→ 정책이 동적 회복(squat/step) 으로 흡수해야 가능. 학습이 **이 동작을 배울 수 있는지가 ±50N의 핵심 도전**.

### 5-2. 진행 표

| 스테이지 | 외력 | torque | mass+ | 시작점 | iter 한도 | 비고 |
|---------|----:|-------:|------:|--------|---------:|-----|
| **E5** | ±33 N | 9.0 | 2.0 | E4 | 8000 | |
| **E5.5** | ±36 N | 10.0 | 2.5 | E5 | 8000 | |
| **E6** | ±39 N | 11.0 | 2.5 | E5.5 | 8000 | mass 첫 증가 |
| **E6.5** | ±42 N | 12.0 | 2.5 | E6 | 8000 | |
| **E7** | ±45 N | 13.0 | 3.0 | E6.5 | 8000 | |
| **E7.5** | ±47.5 N | 14.0 | 3.0 | E7 | 10000 | |
| **E8** | ±50 N | 14.0 | 3.0 | E7.5 | 12000 | 최종 — 더 긴 학습 |

**Phase 3 총 예상**: 7 스테이지 × 평균 9시간 = **63시간 ≈ 2.6일**.

### 5-3. 중간 평가 지점

- **E6 (±39N) 완료 시점**: PASS 조건 4개 모두 만족 못 하면 **±50N 도달 가능성 재평가**
- 그래도 외란 응답 패턴이 동적(step recovery) 이면 진행 가능
- 정적 회복 시도하다가 fall만 한다면 effort_limit 8 Nm로 학습/MuJoCo 동기 변경 검토

---

## 6. Phase 1 — 인프라 변경 (구체)

### 6-1. orchestrator 자동 평가 ([run_newton_training.sh:148](sim/isaaclab/scripts/newton/run_newton_training.sh#L148))

```bash
# 4개 게이트 평가 함수
evaluate_stage() {
    local logfile="$1"; local out_ckpt="$2"; local stage="$3"

    # (1) 마지막 200 iter 평균 fall %
    local fall_avg
    fall_avg=$(grep "base_orientation:" "$logfile" | awk '{print $NF}' \
              | tail -200 | awk '{s+=$1} END{printf "%.4f", s/NR}')

    # (2) 마지막 1000 iter slope (단순화: 첫 500 vs 끝 500 평균 비교)
    local fall_first_half fall_last_half slope
    fall_first_half=$(grep "base_orientation:" "$logfile" | awk '{print $NF}' \
                     | tail -1000 | head -500 | awk '{s+=$1} END{printf "%.4f", s/NR}')
    fall_last_half=$(grep "base_orientation:" "$logfile" | awk '{print $NF}' \
                    | tail -500 | awk '{s+=$1} END{printf "%.4f", s/NR}')

    # (3) 마지막 200 iter 평균 reward
    local reward_avg
    reward_avg=$(grep "Mean reward:" "$logfile" | awk '{print $NF}' \
                | tail -200 | awk '{s+=$1} END{printf "%.2f", s/NR}')

    # (4) MuJoCo 검증
    local mj_result mj_step
    mj_result=$(timeout 60 python3 /home/laba/DGX-NUC/sim/mujoco/play_mujoco.py \
                --ckpt "$out_ckpt" --no-viewer --vx 0.3 --duration 10.0 2>&1 | tail -1)
    mj_step=$(echo "$mj_result" | grep -oP 'Survived \K[0-9]+' || echo 0)

    log "Stage ${stage} 평가:"
    log "  (1) fall % (last 200)  : ${fall_avg}  (기준 < 0.15)"
    log "  (2) fall slope         : ${fall_first_half} → ${fall_last_half}  (감소 추세 필요)"
    log "  (3) reward (last 200)  : ${reward_avg}  (기준 > 8)"
    log "  (4) MuJoCo step        : ${mj_step}/250  (기준 ≥ 230)"

    local pass=0
    awk "BEGIN {exit !(${fall_avg} < 0.15)}"  && ((pass++))
    awk "BEGIN {exit !(${fall_last_half} <= ${fall_first_half})}" && ((pass++))
    awk "BEGIN {exit !(${reward_avg} > 8)}" && ((pass++))
    [[ "$mj_step" -ge 230 ]] && ((pass++))

    log "  통과: ${pass}/4"
    if (( pass >= 3 )); then
        log "  ✅ Stage ${stage} PASS"
        return 0
    else
        log "  ❌ Stage ${stage} FAIL — 추가 학습 또는 보강 단계 필요"
        return 1
    fi
}
```

`run_stage` 함수 끝에 `evaluate_stage` 호출, FAIL 시 `exit 2`.

### 6-2. PPO config ([rsl_rl_ppo_cfg_stageE.py](sim/isaaclab/hylion/agents/rsl_rl_ppo_cfg_stageE.py))

E1~E4 모두:
```python
num_learning_epochs = 2      # 4 → 2 (이미 E4만 적용 중, 전 단계로 확장)
entropy_coef        = 0.012  # 0.008 → 0.012
# learning_rate=5e-5, num_steps_per_env=16, max_grad_norm=0.15 유지
max_iterations = 8000        # 5000 → 8000 (충분한 수렴 시간)
# E4는 max_iterations = 10000
```

E5~E8용으로 `rsl_rl_ppo_cfg_stageF.py` 신규 생성 (동일 config, max_iterations 만 8000~12000).

### 6-3. 새 외란 단계 환경 등록

`HYLION_PERTURB_FORCE` 환경변수가 이미 외란을 받게 되어 있음 ([env_cfg_BG.py:64-66](sim/isaaclab/hylion/env_cfg_BG.py#L64-L66)). task ID는 일부 재사용 가능:

```bash
# E0.3 (±12N) — task은 E1-v0 그대로 재사용, 외란만 환경변수로
HYLION_PERTURB_FORCE=12.0 HYLION_PERTURB_TORQUE=3.5 ...
```

또는 task ID를 별도 등록(`Velocity-Hylion-BG-E0p3-v0` 등)하면 더 깔끔. 후자 권장.

---

## 7. Phase 4 — 최종 검증

### 7-1. MuJoCo 무외란 (스테이지 비교)

```bash
for ckpt in d4 d5 e1 e4 e6 e8; do
  for vx in 0.0 0.3 0.5; do
    echo "=== stage_$ckpt vx=$vx ==="
    python3 sim/mujoco/play_mujoco.py \
      --ckpt checkpoints/biped/stage_${ckpt}_hylion_v6/best.pt \
      --no-viewer --vx $vx --duration 15.0 2>&1 | tail -1
  done
done
```

기대치:
- 모든 스테이지: vx=0.3 → 250/250
- E4, E8: vx=0.5 → ≥ 230/250

### 7-2. MuJoCo 외란 모드 (신규 옵션)

`sim/mujoco/play_mujoco.py`에 `--external-push <N> --push-interval <s>` 추가:

```bash
# E8(±50N) 학습본을 ±50N 임펄스로 검증
python3 sim/mujoco/play_mujoco.py \
  --ckpt checkpoints/biped/stage_e8_hylion_v6/best.pt \
  --no-viewer --vx 0.3 --duration 15.0 \
  --external-push 50 --push-interval 2.5
```

기대치: 5초 이상 보행 유지, 매 push 후 ~2초 내 자세 복구.

### 7-3. 발표 영상

GUI 모드:
1. vx=0.3 무외란 10초 (정상 보행)
2. vx=0.5 무외란 10초 (빠른 보행)
3. vx=0.3 + ±50N push every 2.5s, 15초 (강건성)

---

## 8. 진행 시 매 스테이지 체크리스트

```
[ ] 1. 학습 명령 실행 (run_newton_training.sh, START_STAGE 지정)
[ ] 2. 학습 완료 후 evaluate_stage 자동 평가 (4개 조건)
[ ] 3. PASS → 다음 진행 / FAIL → +2000 iter 또는 보강 단계
[ ] 4. README.md "현재 학습 상태" 표 갱신 (자동화 또는 수동)
[ ] 5. docs/jimmy/sim2sim_progress.md 결과 표 갱신
[ ] 6. weight std/max|w| 기록 (인플레 모니터링)
[ ] 7. 다음 스테이지 시작
```

---

## 9. 실패 시 대응

### 9-1. 1차 대응 — 추가 학습

PASS 조건 2개 만족 시 (slope만 미달 등): `MAX_ADDITIONAL_ITER=2000` 으로 같은 외란 재학습.

### 9-2. 2차 대응 — 보강 단계 삽입

| 실패 단계 | 보강 외력 | iter | 진입점 |
|----------|---------:|-----:|--------|
| E0.3 (±12N) | ±11 N | 5000 | D5 |
| E0.6 (±13.5N) | ±12.7 N | 5000 | E0.3 |
| E1 (±15N) | ±14 N | 5000 | E0.6 |
| E1.4 (±17N) | ±16 N | 5000 | E1 |
| ... | (외란 +1.5N만 증가) | 5000 | 이전 단계 |
| E5 (±33N) | ±31.5 N | 6000 | E4 |
| E8 (±50N) | ±48.5 N | 8000 | E7.5 |

### 9-3. 3차 대응 — 알고리즘 보강

NaN 또는 weight magnitude 폭주 시:
- `num_learning_epochs` 2 → 1
- `max_grad_norm` 0.15 → 0.10
- `learning_rate` 5e-5 → 3e-5
- 이전 단계 best.pt로 롤백 후 +1N 만 증가시켜 재도전

### 9-4. 4차 대응 — 물리 파라미터 조정 (최후의 보루)

E6+ 에서 PASS 못 하면:
- effort_limit 6 → 8 Nm 로 학습/MuJoCo 동기 변경
- 이는 sim-to-real 정확도와 trade-off → 사용자 결정 필요

---

## 10. 시간표

| 일 | 작업 | 비고 |
|---:|------|------|
| 1 | Phase 1 인프라 (0.5일) + D4.5 시작 | 코드 수정, 첫 학습 시작 |
| 2 | D4.5 PASS → D5 학습 | |
| 3 | D5 PASS → E0.3, E0.6 | 작은 증분 빠른 진행 |
| 4 | E1, E1.4, E1.8 | |
| 5 | E2.3, E2.7, E3 | |
| 6 | E3.5, E4 | Phase 2 완료 |
| 7 | E5, E5.5 | |
| 8 | E6, E6.5 | 중간 평가 |
| 9 | E7, E7.5 | |
| 10 | E8 (10000~12000 iter, 더 길게) | |
| 11 | Phase 4 검증 + 영상 | |

**총 11일 (마감까지 21일 여유).**

장애 발생 시 9-1~9-3 대응으로 +3~5일 흡수 가능.

---

## 11. 결정해주실 것 (시작 전)

### 11-1. 시작점

✅ **권장: D4.5부터 재학습** (D4 PhysX 결과를 Newton으로 D4.5 첫 진입 → 베이스라인 회복부터).  
대안: D4까지 모두 재학습 (시간 +0.5일, 최대 안전).

### 11-2. effort_limit 정책

- **현행 6 Nm 유지** + 동적 회복으로 ±50N 학습 시도 (sim-to-real 정확도 우선)
- E6+ 에서 PASS 못 하면 **8 Nm로 동기 변경** (물리적으로 가능 영역 진입)

→ ✅ **권장: 6 Nm로 시작, E6 평가 시점에 재결정**.

### 11-3. MuJoCo 게이트 강도

- **권장: 230/250** (vx=0.3, 92% 통과 — 충분한 안전 마진, 학습 환경과 약간의 gap 허용)
- 엄격: 250/250 (모든 스테이지 완벽 보행 강제 — 더 자주 막힘)

→ ✅ **권장: 230/250**.

위 권장안대로면 **즉시 Phase 1 코드 수정 시작 가능**. 다른 결정 원하시면 말씀해주세요.
