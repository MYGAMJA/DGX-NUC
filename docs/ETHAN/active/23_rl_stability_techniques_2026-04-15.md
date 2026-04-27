# RL 학습 안정화 기법 정리 (2026-04-15)

현재 겪는 문제들에 대응하는 최신 기법 모음. 필요할 때 꺼내 쓰는 용도.

---

## 현재 적용된 수정 (2026-04-15)

```
entropy_coef  : 0.05  → 0.005   (action_std 폭발 근본 원인)
init_noise_std: 1.0   → 0.5     (초기 탐험 폭 제한)
max_grad_norm : 0.1   → 0.05    (gradient 폭발 추가 방어)
체크포인트    : M2 → model_5999.pt (Stage-A 기점으로 초기화)
로그          : /tmp/hylion_v6_physx_M2_restart.log
```

---

## 문제별 기법 목록

### 문제 1. action_std 폭발

증상: action_std가 0.6 → 2.5로 폭주, episode_length 100 → 5로 붕괴

| 기법 | 논문 | 핵심 | 난이도 |
|------|------|------|--------|
| entropy_coef 낮추기 | RSL-RL 표준 | 0.05 → 0.005 | ✅ 적용됨 |
| **GPO** (Growing Policy Optimization) | arXiv:2601.20668 (2026) | 초기엔 action space 좁게, 학습되면서 점점 넓힘 (Gompertz 스케줄) | 중 |
| **CE-GPPO** | arXiv:2509.20712 (2025) | clip된 gradient를 부드럽게 복원해 entropy 제어 | 고 |

**GPO 핵심 아이디어:**
- 학습 초반에 관절 action 범위를 작게 제한
- 정책이 안정되면 자동으로 범위 확장
- action_std 폭발 자체를 구조적으로 막음

---

### 문제 2. value/surrogate loss NaN

증상: gradient 폭발 → loss nan → action_std=0.00 고정

| 기법 | 논문 | 핵심 | 난이도 |
|------|------|------|--------|
| max_grad_norm 낮추기 | RSL-RL 표준 | 0.1 → 0.05 | ✅ 적용됨 |
| obs normalization clipping | ETH Zurich 표준 | 극단 obs 값 [-10,10]으로 차단 | 하 |
| value function clipping (DPPO) | DPPO variants (2024) | value estimate 폭발 방지 | 중 |

**확인할 것:**
- IsaacLab이 obs normalization을 하지만 clipping 범위가 얼마인지 체크
- `empirical_normalization = False` 현재 꺼져 있음 → True로 켜는 것 고려

---

### 문제 3. feet_air_time = 0 (보행 신호 사망)

증상: 발이 땅에 닿아도 contact force = 0, 로봇이 서 있는 법만 학습

→ 이미 해결됨 (USD 자산 교체 + contact_sensor.py 패치)

| 기법 | 논문 | 핵심 | 비고 |
|------|------|------|------|
| feet_air_time 임계값 낮추기 | 현재 적용 | 0.4 → 0.2 | ✅ 적용됨 |
| **TumblerNet (COM/COP 보상)** | npj Robotics 2025 | feet_air_time 대신 압력중심 기반 보상 | air_time이 너무 작을 때 대안 |
| Reward Machines | 2024 | 접촉 패턴을 논리 규칙으로 인코딩 (L→R→L...) | 중 |

**TumblerNet 아이디어:**
- `feet_air_time` 대신 COM(무게중심)과 COP(압력중심) 벡터 차이를 보상으로 사용
- 접촉 자체가 아닌 "얼마나 잘 균형 잡혔나"를 직접 측정
- feet_air_time 값이 0.001~0.005로 너무 작을 때 신호가 약해지는 문제 해결

---

### 문제 4. 경량→중량 로봇 전이 불안정

증상: BHL biped(경량) → Hylion v6(팔 달린 중량) 전이 시 불안정

| 기법 | 논문 | 핵심 | 난이도 |
|------|------|------|--------|
| 질량 커리큘럼 (현재 M1~M6) | 자체 설계 | 상체 질량 0%→30%→100% 단계적 증가 | ✅ 진행 중 |
| **이중 커리큘럼** | Nature Comm. 2025 | 질량 스케일 + 태스크 난이도 동시 단계적 증가 | 중 |
| **페이로드 도메인 랜덤화** | CHRL 2024 | 학습 중 질량을 랜덤으로 흔들어 미리 적응 | 중 |
| Teacher-Student Distillation | 2024-2025 | 경량 teacher → 중량 student로 지식 전달 | 고 |

**더 잘게 쪼개는 커리큘럼 제안:**
```
현재: 0% → 30% → 100%
개선: 0% → 15% → 30% → 50% → 70% → 100%
```
각 단계에서 300 iteration 이상 안정 후 다음 단계 진행

---

### 문제 5. episode_length 붕괴

증상: 학습 중 갑자기 100+에서 5~6으로 급락

| 기법 | 논문 | 핵심 | 난이도 |
|------|------|------|--------|
| action_rate_l2 가중치 올리기 | 현재 환경 | 급격한 동작 패널티 강화 | 하 |
| GPO (action space 제한) | arXiv:2601.20668 | 초기 action 범위 제한으로 넘어짐 줄임 | 중 |
| 보상 정규화 (GRPO) | 2024 | 보상을 그룹 평균 기준으로 정규화 | 중 |

---

## 적용 우선순위 로드맵

```
즉시 완료 ✅
  - entropy_coef 0.005
  - init_noise_std 0.5
  - max_grad_norm 0.05
  - model_5999.pt 재시작

단기 (M2_restart 불안정 시) ───────────────────
  1. empirical_normalization = True 켜기
     (obs 정규화 활성화)
  2. action_rate_l2 페널티 가중치 2배
     (급격한 동작 → 넘어짐 사전 차단)
  3. feet_air_time threshold 0.2 → 0.1
     (보상 신호 감도 높이기)

중기 (장기 런 안정화 이후) ─────────────────────
  4. 질량 커리큘럼 더 잘게 쪼개기
     (0% → 15% → 30% → 50% → 70% → 100%)
  5. 페이로드 도메인 랜덤화 추가

장기 (데모 이후, 실기 전환 준비) ───────────────
  6. GPO 적용 (action space scheduling)
  7. TumblerNet 방식 보상 실험
  8. Teacher-Student distillation 검토
```

---

## 참고 논문

| 논문 | 링크 | 해결하는 문제 |
|------|------|--------------|
| GPO: Growing Policy Optimization | arXiv:2601.20668 | action_std 폭발, episode 붕괴 |
| CE-GPPO | arXiv:2509.20712 | gradient 폭발, NaN |
| TumblerNet | npj Robotics 2025 | feet_air_time 대안 |
| Bioinspired Morphology Curriculum | Nature Comm. 2025 | 경량→중량 전이 |
| RSL-RL Library | arXiv:2509.10771 | PPO 표준 구현 (ETH Zurich) |
| Real-world Humanoid Locomotion | Science Robotics 2024 | 전반적 안정화 |
