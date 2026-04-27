# BHL RL 학습 가이드 및 결과

> **학습은 반드시 Isaac Sim 6.0.0 + Newton으로 실행해야 한다.**
> PhysX는 DGX Spark(aarch64)에서 CPU로만 동작 → GPU 물리 가속 없음 → 학습 속도 저하.
> Newton은 Isaac Sim 6.0.0 + IsaacLab develop에서만 지원된다.

## 실행 환경

- Isaac Sim 6.0.0 (pip) + IsaacLab develop
- `/home/laba/env_isaaclab` 가상환경
- 물리 백엔드: **Newton (GPU) — 필수**

---

## rsl-rl 5.0.1 설정 수정

BHL의 `rsl_rl_ppo_cfg.py`는 deprecated `policy = RslRlPpoActorCriticCfg(...)` 포맷을 사용한다.
rsl-rl 5.0.1에서는 `OnPolicyRunner`가 `cfg["actor"].pop("class_name")`을 호출하는데,
`to_dict()`가 MISSING 값을 `{}`로 변환하여 `KeyError: 'class_name'` 발생.

**해결**: `train.py`에 `handle_deprecated_rsl_rl_cfg()` 추가.

```python
import importlib.metadata as metadata
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# main() 내부, OnPolicyRunner 생성 전:
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
```

---

## Newton 백엔드 설정 (train.py에 추가)

`env = gym.make(...)` 호출 **전에** 삽입:

```python
from isaaclab_newton.physics import NewtonCfg

env_cfg.sim.physics = NewtonCfg()
env_cfg.events.physics_material = None  # PhysX 전용 API, Newton에서 오류 발생
```

> **참고**: 1차 학습은 이 코드 없이 PhysX(CPU)로 진행됨.
> 체크포인트 포맷은 동일하므로 PhysX 체크포인트에서 Newton으로 `--resume` 가능.

---

## 학습 실행 명령어

### 기본 (Newton 백엔드)

```bash
cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
source /home/laba/env_isaaclab/bin/activate
PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
  nohup python train.py \
    --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 6000 > /tmp/bhl_train_newton.log 2>&1 &
```

```bash
tail -f /tmp/bhl_train_newton.log  # 로그 확인
```

체크포인트 저장 위치: `logs/rsl_rl/biped/<timestamp>/model_<iter>.pt`

### 이어서 학습 (resume)

```bash
python train.py ... --resume
```

---

## 1차 학습 결과 (PhysX 백엔드, 2026-03-27)

| 항목 | 값 |
|------|-----|
| 학습 환경 | Isaac Sim 6.0.0, PhysX(CPU), GB10 GPU (PPO 신경망) |
| iterations | 6000 |
| 총 스텝 | 589,824,000 |
| Steps/sec | ~28,807 |
| 총 소요 시간 | **5시간 46분** |
| 체크포인트 | `logs/rsl_rl/biped/2026-03-27_14-36-49/model_5999.pt` |

### 최종 보상 (Iteration 5999)

| 항목 | 값 |
|------|-----|
| Mean reward | **33.21** |
| Mean episode length | **491.71 steps (19.6초)** |
| `track_lin_vel_xy_exp` | 1.6112 |
| `track_ang_vel_z_exp` | 0.4676 |
| `flat_orientation_l2` | -0.0147 |
| `action_rate_l2` | -0.0760 |
| `dof_torques_l2` | -0.1748 |
| `feet_air_time` | 0.0409 |
| `feet_slide` | -0.0296 |

### 안정성 지표

| 지표 | 값 | 평가 |
|------|-----|------|
| `termination/time_out` | **95.3%** | 에피소드 대부분 타임아웃(정상 완주) |
| `termination/base_orientation` | 4.7% | 낙상 극소 |
| `error_vel_xy` | 0.21 m/s | 수평 속도 오차 양호 |
| `error_vel_yaw` | 0.54 rad/s | yaw 오차 (개선 여지 있음) |

### 종합 평가

- 에피소드 평균 19.6초 유지 (파라메트릭 테스트 기준 5초의 4배)
- 낙상률 4.7% — **파라메트릭 직립 테스트 진행에 충분한 품질**

---

## PhysX vs Newton 비교 (학습에 Newton을 써야 하는 이유)

| 항목 | PhysX | Newton |
|------|-------|--------|
| 물리 연산 위치 | CPU (DGX Spark aarch64 제약) | GPU (GB10) |
| Steps/sec | ~28,700 | 향상 기대 (CPU 병목 제거) |
| 학습 6000 iter 소요 | ~5h 46m | 단축 기대 |
| 6.0.0 지원 | ✅ | ✅ |
| 5.1.0 지원 | ✅ | ❌ |

**결론: 이후 모든 학습은 Newton 백엔드로 실행한다.**

> 1차 학습(model_5999.pt)은 Newton 설정 누락으로 PhysX로 학습됨 — 이미 완료된 체크포인트이므로 재학습 불필요.
> **2차 학습부터는 train.py에 Newton 설정 코드가 포함되어 있으므로 그대로 실행하면 Newton으로 동작.**
