# LABA5 Bootcamp — 2026-03-27 작업 개요

## 목표

Berkeley Humanoid Lite(BHL) 로봇을 DGX Spark(NVIDIA GB10, aarch64)에서 강화학습으로 훈련하고, 학습된 policy를 GUI 창으로 시각화한다.

---

## 두 환경 전략 — 규칙 (절대 변경 금지)

> **학습은 반드시 Newton, 시각화는 반드시 5.1.0**

```
학습  → Isaac Sim 6.0.0 + Newton GPU 물리 (headless, Python 3.12)
시각화 → Isaac Sim 5.1.0 + IsaacLab 2.3.2 (GUI 창 출력, Python 3.11)
```

### 왜 이 조합이어야 하는가

| | Isaac Sim 6.0.0 (pip) | Isaac Sim 5.1.0 (standalone) |
|--|--|--|
| 위치 | `/home/laba/env_isaaclab/` | `/home/laba/IsaacSim/` |
| Python | 3.12 | 3.11 (자체 내장) |
| IsaacLab | 3.0 (develop) | 2.3.2 |
| Newton 물리 | ✅ GPU 가속 | ❌ PhysX만 (Newton 미지원) |
| GUI 창 | ❌ headless only | ✅ |
| PyTorch | CUDA (GB10) | CPU-only (aarch64 제약) |

- **학습에 Newton 필수**: DGX Spark(aarch64)에서 PhysX는 CPU로 동작 → GPU 물리 가속을 위해 Newton 필수. 6.0.0에서만 Newton 지원.
- **시각화에 5.1.0 필수**: 6.0.0 pip 버전은 headless only — GUI 창 출력 불가. 5.1.0 standalone만 DISPLAY=:1로 DGX 모니터에 창 출력 가능.
- **다른 조합은 동작하지 않음**: 6.0.0으로 시각화 시도 시 창 없음. 5.1.0으로 학습 시 Newton 미지원 + 느림.

> BHL은 원래 Isaac Sim 5.1.0 + IsaacLab 2.3.2 기준으로 개발됨 (`setup.py` 확인).
> 학습 체크포인트 포맷은 동일하므로 6.0.0에서 학습 후 5.1.0에서 `map_location="cpu"`로 로드 가능.

---

## 완료 현황 (2026-04-01 기준)

- [x] Isaac Sim 6.0.0 + IsaacLab develop + Newton 설치
- [x] BHL 환경 설치 및 IsaacLab develop API 호환 수정
- [x] RL 학습 완료 (6000 iter, 5h 46m, `model_5999.pt`)
- [x] 파라메트릭 직립 테스트 완료 (24/24 pass → δ3 무게 예산 확정)
- [x] Isaac Sim 5.1.0 + IsaacLab 2.3.2 + BHL + rsl-rl 5.0.1 설치
- [x] DGX 모니터에서 학습된 BHL 로봇 걷는 것 시각화 성공 ✅
- [x] Hylion v4 (BHL + SO-ARM, y=±0.12) USD 변환 완료 (6.0 학습용 + 5.1.0 시각화용)
- [x] Hylion v4 걷기 학습 완료 (6000 iter, 1h 30m, Newton, `model_9900.pt`)
- [ ] Hylion v4 시각화 확인 진행 중

---

## 파일 인덱스

| 파일 | 내용 |
|------|------|
| [01_env_setup.md](01_env_setup.md) | Isaac Sim 6.0.0 + IsaacLab + Newton + BHL 설치 |
| [02_training.md](02_training.md) | RL 학습 실행, 결과, Newton 전환 |
| [03_visualization.md](03_visualization.md) | Isaac Sim 5.1.0 시각화 구동 (에러 해결 포함) |
| [04_parametric_test.md](04_parametric_test.md) | 파라메트릭 직립 테스트 계획 및 결과 |
| [05_hardware_specs.md](05_hardware_specs.md) | BHL 하드웨어 스펙 (모터, ESC, ROS2 토픽) |
| [06_hylion_training.md](06_hylion_training.md) | Hylion v3 (BHL + SO-ARM) 걷기 학습 |
| [07_hylion_v4_training.md](07_hylion_v4_training.md) | Hylion v4 학습 + 시각화 (2026-04-01) |

---

## 핵심 경로

```
/home/laba/Berkeley-Humanoid-Lite/           # BHL 리포지토리
/home/laba/env_isaaclab/                     # Isaac Sim 6.0.0 가상환경
/home/laba/IsaacSim/                         # Isaac Sim 5.1.0 standalone
/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/biped/2026-03-27_14-36-49/
                                             # 학습 체크포인트 (model_5999.pt)
```
