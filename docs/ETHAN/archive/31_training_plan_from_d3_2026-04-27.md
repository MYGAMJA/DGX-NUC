# 학습 재개 계획 — D3 NaN 복구 후 E4까지 (2026-04-27)

---

## 1. 현재 상태 진단

### 체크포인트 감사 결과

| 스테이지 | 외력 | weight 상태 | 비고 |
|---------|------|------------|------|
| B+       | —    | ✅ 정상 | |
| C1~C4    | —    | ✅ 정상 | |
| D1       | ±1N  | ✅ 정상 | w0_mean=0.2617 |
| D1.5     | ±1.5N| ✅ 정상 | w0_mean=0.2649 |
| D2       | ±2N  | ✅ 정상 | w0_mean=0.2697 |
| **D2.5** | ±2.5N| ✅ **정상 (최종 유효)** | w0_mean=0.2709 |
| D3       | ±3N  | ❌ **NaN 폭발** | b789d8f0, 전체 nan |
| D4       | —    | ❌ D3 복사본 | 동일 NaN |
| D5       | —    | ❌ D3 복사본 | 동일 NaN |

### D3 NaN 원인 분석

- D2.5 (epochs=2) → D3 (epochs=3) **epochs 증가 + ±2.5→±3N 동시 변화**
- 넘어지는 순간 관절 가속도/토크 극대화 → value gradient 발산 → NaN 전파
- D2에서도 동일 패턴 발생했으나 당시 epochs=2에서 회복된 이력 있음

### 수정 완료 사항

- `rsl_rl_ppo_cfg_stageD_optionA.py`: D3 `num_learning_epochs` 3 → **2**
- `dgx/train_biped.sh`: D2.5 → **D3 재도전**으로 변경

---

## 2. 전체 학습 로드맵

```
[완료] D2.5  ±2.5N  epochs=2  → 재개 기점
   ↓
[지금] D3    ±3N    epochs=2   4000iter  ~6시간   ← train_biped.sh 실행
   ↓ (성공 확인 후)
[예정] D4    ±5N    epochs=3   4000iter  ~7시간
[예정] D4.5  ±7N    epochs=3   4000iter  ~7시간
[예정] D5    ±10N   epochs=4   5000iter  ~9시간
   ↓
[예정] E1    ±15N   epochs=4   5000iter  ~5시간
[예정] E2    ±20N   epochs=4   6000iter  ~6시간
[예정] E3    ±25N   epochs=4   6000iter  ~6시간
[예정] E4    ±30N   epochs=4   7000iter  ~7시간
                                         ─────────
                              총 잔여    ~53시간
```

> **마감: 2026-06-01 (발표)** → 시간 여유 충분

---

## 3. 각 스테이지 파라미터

### D3 재도전 (현재)
```
task:     Velocity-Hylion-BG-D3-v0
resume:   stage_d2_5_hylion_v6/best.pt
FORCE=3.0  TORQUE=1.0  MASS=0.5  VEL=0.5  STANDING=0.05
PPO: epochs=2, lr=5e-5, steps=16, grad_norm=0.15
```

### D4 (D3 성공 후)
```
task:     Velocity-Hylion-BG-D4-v0
resume:   stage_d3_hylion_v6/best.pt
FORCE=5.0  TORQUE=1.7  MASS=1.0  VEL=0.5  STANDING=0.05
PPO: epochs=3, lr=5e-5 (D3 정상 확인 후 epochs 복귀)
```

### D4.5 (±7N, 완충 단계)
```
task:     Velocity-Hylion-BG-D4p5-v0
resume:   stage_d4_hylion_v6/best.pt
FORCE=7.0  TORQUE=2.3  MASS=1.0  VEL=0.5  STANDING=0.07
```

### D5 (±10N)
```
task:     Velocity-Hylion-BG-D5-v0
resume:   stage_d4_5_hylion_v6/best.pt
FORCE=10.0  TORQUE=3.0  MASS=1.5  VEL=0.5  STANDING=0.10
PPO: epochs=4
```

### E1~E4 (run_e_stages_training.sh)
| 스테이지 | FORCE | TORQUE | MASS | epochs |
|---------|-------|--------|------|--------|
| E1 ±15N | 15.0 | 5.0 | 1.5 | 4 |
| E2 ±20N | 20.0 | 6.0 | 2.0 | 4 |
| E3 ±25N | 25.0 | 7.5 | 2.0 | 4 |
| E4 ±30N | 30.0 | 9.0 | 2.0 | 4 |

---

## 4. 조기 중단 기준 (500 iter 확인)

| 지표 | 정상 | 경고 | 즉시 중단 |
|------|------|------|---------|
| orientation termination | < 15% | 15~30% | **> 30%** |
| mean reward | > 20 | 10~20 | **< 10 하락 중** |
| 에피소드 길이 | > 300 | 100~300 | **< 100** |

→ **즉시 중단 기준 2개 이상 충족 시** → 아래 실패 대응 실행

---

## 5. 실패 시 대응 (단계별 분할)

| 실패 스테이지 | 즉시 삽입 | FORCE | iter |
|------------|---------|-------|------|
| D3(±3N)    | D2.8    | 2.8   | 2000 |
| D4(±5N)    | D3.5    | 3.5   | 2000 |
| D4.5(±7N)  | D4.2    | 4.2   | 2000 |
| D5(±10N)   | D4.7    | 4.7   | 3000 |
| E1(±15N)   | E0.5    | 12.0  | 3000 |
| E2(±20N)   | E1.5    | 17.0  | 3000 |
| E3(±25N)   | E2.5    | 22.0  | 3000 |
| E4(±30N)   | E3.5    | 27.0  | 3000 |

**NaN 발생 시 추가 조치:**
1. `num_learning_epochs` 1단계 감소 (3→2 또는 4→3)
2. `max_grad_norm` 0.15→0.10으로 강화
3. 이전 스테이지 best.pt로 롤백 후 재도전

---

## 6. 학습 실행 순서

```bash
# ── DGX에서 ─────────────────────────────────────────
# 1. D3 재도전 (지금 바로)
tmux new -s d3
bash /home/laba/DGX-NUC/dgx/train_biped.sh
# 다른 창에서 모니터링:
tail -f /tmp/hylion_stageD3_retry.log | grep -E "Iter|orientation|Reward"

# 2. D3 성공 확인 후 train_biped.sh를 D4용으로 수정하고 재실행
# (orientation < 15%, reward > 25 확인)

# 3. D5까지는 run_optionA_training.sh를 START_STAGE=D4로 실행 가능
START_STAGE=D4 \
  D4_CKPT=/home/laba/DGX-NUC/checkpoints/biped/stage_d3_hylion_v6/best.pt \
  bash /home/laba/DGX-NUC/sim/isaaclab/scripts/run_optionA_training.sh

# 4. D5 완료 후 E 단계
bash /home/laba/DGX-NUC/sim/isaaclab/scripts/run_e_stages_training.sh
```

---

## 7. sim-to-sim 검증 타이밍

| 시점 | 검증 체크포인트 | 목적 |
|------|--------------|------|
| 지금 (NUC) | **D2.5** | 기준선 확인 (현재 최선) |
| D3 완료 후 | D3 | 외력 3N 하에서 안정성 개선 확인 |
| D5 완료 후 | D5 | 발표용 최소 기준 확인 |
| E4 완료 후 | E4 | 최종 30N 강건성 확인 |

> **현재 sim-to-sim 실패 원인은 체크포인트 문제가 아님**  
> (D2.5는 정상, 하지만 MuJoCo에서 35스텝 후 collapse)  
> → effort_limit=6Nm 부족 가설 (ankle ~8Nm 필요) 별도 검증 필요
