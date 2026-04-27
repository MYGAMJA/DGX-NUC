# 외력 증가 시 실패 대응 계획 — 1주일 내 완료 전략 (2026-04-20)

> **마감: 2026-04-27 (7일)**  
> **현재 상태:** D1 학습 중 (iter 18700~21700, ±1N)  
> **최종 목표:** ±30N 강건 보행

---

## 1. 시간 예산 (7일)

| 기간 | 내용 | 허용 시간 |
|------|------|---------|
| 4/20~4/22 | D1~D5 완료 (±1N → ±10N) | ~20시간 학습 |
| 4/22~4/25 | E1~E4 완료 (±15N → ±30N) | ~24시간 학습 |
| 4/25~4/26 | 실패 재도전 버퍼 | 24시간 |
| 4/26~4/27 | 최종 검증 + NUC 이전 | 12시간 |

**핵심 원칙: 실패 1회당 최대 8시간 소비로 제한**

---

## 2. 실패 판단 기준

학습 시작 후 **500 iter 기준**으로 조기 판단:

| 지표 | 정상 | 경고 | 즉시 중단 |
|------|------|------|---------|
| orientation 종료율 | < 15% | 15~30% | **> 30%** |
| mean reward | > 20 | 10~20 | **< 10이고 하락 중** |
| 에피소드 길이 | > 300 | 100~300 | **< 100** |

→ 즉시 중단 기준 2개 이상 동시 충족 시 **아래 대응 절차 즉시 실행**

---

## 3. 실패 시 대응 절차 (단계별)

### STEP 1 — 이전 스테이지 best.pt로 롤백 (5분)
```bash
# 실패한 스테이지가 예: E2(±20N)라면
# E1 best.pt로 돌아가서 힘을 더 잘게 쪼갬
ls /home/laba/project_singularity/δ3/checkpoints/stage_e1_hylion_v6/best.pt
```

### STEP 2 — 힘을 절반으로 쪼개서 중간 스테이지 삽입
```
실패 예시: E1(±15N) → E2(±20N) 실패
대응:      E1(±15N) best.pt → E1.5(±17N, 3000iter) → E2(±20N) 재시도
```

**미리 계획된 분할 버퍼** (실패 시 즉시 사용):

| 실패 스테이지 | 삽입할 중간 단계 | 명령 |
|------------|--------------|------|
| D3(±3N) 실패 | ±2.5N, 2000iter | `HYLION_PERTURB_FORCE=2.5` |
| D4(±5N) 실패 | ±4N, 3000iter | `HYLION_PERTURB_FORCE=4.0` |
| D5(±10N) 실패 | ±7N, 3000iter → ±10N | `HYLION_PERTURB_FORCE=7.0` |
| E1(±15N) 실패 | ±12N, 3000iter | `HYLION_PERTURB_FORCE=12.0` |
| E2(±20N) 실패 | ±17N, 3000iter | `HYLION_PERTURB_FORCE=17.0` |
| E3(±25N) 실패 | ±22N, 3000iter | `HYLION_PERTURB_FORCE=22.0` |
| E4(±30N) 실패 | ±27N, 3000iter | `HYLION_PERTURB_FORCE=27.0` |

### STEP 3 — 중간 단계 실행 명령 템플릿
```bash
cd /home/laba/project_singularity
unset PYTHONPATH && unset PYTHONHOME

# 중간 단계 즉시 실행 (예: ±17N 삽입)
HYLION_ENABLE_PERTURBATION=1 \
HYLION_BASE_MASS_ADD_KG=1.5 \
HYLION_PERTURB_FORCE=17.0 \
HYLION_PERTURB_TORQUE=5.5 \
/home/laba/env_isaaclab/bin/python δ3/scripts/train_hylion_physx_BG.py \
  --task Velocity-Hylion-BG-E2-v0 \
  --num_envs 4096 --headless \
  --pretrained_checkpoint δ3/checkpoints/stage_e1_hylion_v6/best.pt

# torque = force × 0.32 (D 단계 비율 유지)
```

---

## 4. 마감 내 포기 기준 (시간이 부족할 때)

일주일 안에 30N이 안 되면 **달성한 최고 수준을 결과로 제출**:

| 달성 수준 | 데모 가능 내용 | 평가 |
|---------|-------------|------|
| ±10N (D5) | 가볍게 툭 치는 충격 복원 | ✅ 최소 목표 |
| ±15N (E1) | 한 손 가볍게 밀기 복원 | ✅ 양호 |
| ±20N (E2) | 팔꿈치로 밀기 복원 | ✅ 우수 |
| **±30N (E4)** | **세게 밀어도 복원** | ✅ 최종 목표 |

→ **±10N이 실증되면 이미 논문/데모 수준**. 30N 미달성이어도 진도 중단 없이 제출 가능.

---

## 5. 각 스테이지 모니터링 명령

```bash
# 현재 학습 실시간 수치 확인
/home/laba/env_isaaclab/bin/python - << 'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os, glob

log_root = "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion"
latest = sorted(glob.glob(f"{log_root}/2026-*/"), key=os.path.getmtime)[-1]
ea = EventAccumulator(latest); ea.Reload()
r = ea.Scalars("Train/mean_reward")
o = ea.Scalars("Episode_Termination/base_orientation")
print(f"로그: {latest}")
print(f"iter: {r[-1].step} | reward: {r[-1].value:.1f} | orient: {o[-1].value*100:.2f}%")
for rv, ov in zip(r[-5:], o[-5:]):
    print(f"  {rv.step}: reward={rv.value:.1f}, orient={ov.value*100:.2f}%")
EOF
```

---

## 6. NUC 이전 체크리스트

D5 또는 목표 스테이지 완료 후:

```
□ 1. best.pt 파일 확인
      ls δ3/checkpoints/stage_d5_hylion_v6/best.pt  (또는 e4)

□ 2. 체크포인트 + 설정 패키징
      tar czf hylion_v6_final_$(date +%F).tar.gz \
        δ3/checkpoints/ \
        δ3/hylion/ \
        δ3/scripts/ \
        δ3/usd/hylion_v6/

□ 3. NUC로 전송
      scp hylion_v6_final_*.tar.gz [NUC_USER]@[NUC_IP]:~/

□ 4. NUC에서 재현 테스트
      tar xzf hylion_v6_final_*.tar.gz
      # 동일 환경(isaaclab) 필요

□ 5. 시연 스크립트 실행 확인
      python δ3/scripts/play_hylion.py \
        --checkpoint δ3/checkpoints/stage_d5_hylion_v6/best.pt
```

---

## 7. 이번 주 일일 목표

| 날짜 | 목표 스테이지 | 확인 사항 |
|------|------------|---------|
| 4/20 (오늘) | D1 완료 | orientation < 15% |
| 4/21 | D2~D3 완료 | ±3N 통과 여부 (이전 C2 실패 구간) |
| 4/22 | D4~D5 완료 | ±10N 달성 — **최소 목표 확보** |
| 4/23 | E1~E2 시도 | ±15N~20N |
| 4/24 | E3~E4 시도 | ±25N~30N |
| 4/25 | 실패 재도전 버퍼 | 중간 단계 삽입 실행 |
| 4/26 | 최종 스테이지 완료 | best.pt 확정 |
| 4/27 | NUC 이전 + 시연 확인 | 패키지 전달 |

---

*작성: 2026-04-20 | 상태: 운영 중 (D1 학습 진행 중)*
