# Isaac Sim 5.1.0 시각화 구동 가이드

> **시각화는 반드시 Isaac Sim 5.1.0 standalone으로 실행해야 한다.**
> Isaac Sim 6.0.0 pip 버전은 headless only — DGX 모니터에 GUI 창 출력 불가.
> 5.1.0 standalone만 `DISPLAY=:1` 환경에서 RTX Viewport 창을 띄울 수 있다.

## 결과

> **DGX 모니터에서 학습된 BHL 로봇이 Isaac Sim 5.1.0 GUI 창으로 걷는 것 확인 ✅ (2026-03-30)**

- 체크포인트: `model_5999.pt` (6000 iter, PhysX 학습)
- 환경: 1 env, CPU device
- 로봇이 랜덤 velocity command를 따라 보행

---

## 환경 특성

| 연산 | GPU 여부 |
|------|---------|
| 물리 시뮬레이션 (PhysX) | ✅ GPU |
| 렌더링 (RTX Viewport 창) | ✅ GPU |
| 정책 NN 추론 (PyTorch) | ❌ CPU (torch 2.7.0+cpu) |
| Reward/Obs 계산 | ❌ CPU |

5.1.0 kit Python은 CPU-only torch만 제공 (NVIDIA PyPI에 aarch64 CUDA torch 없음).
시각화 목적(1 env)에서 NN 추론이 CPU여도 실질적 차이 없음 (~0.1ms/step).

---

## 설치 (5.1.0 환경)

### IsaacLab 2.3.2 설치 (5.1.0 kit Python 사용)

```bash
ISAACSIM=/home/laba/IsaacSim/_build/linux-aarch64/release
SITE=${ISAACSIM}/kit/python/lib/python3.11/site-packages

# IsaacLab 2.3.2 clone
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git ${SITE}/isaaclab
cd ${SITE}/isaaclab

# 5.1.0 kit Python으로 설치
${ISAACSIM}/kit/python/bin/python3 -m pip install -e source/isaaclab
${ISAACSIM}/kit/python/bin/python3 -m pip install -e source/isaaclab_assets
${ISAACSIM}/kit/python/bin/python3 -m pip install -e source/isaaclab_tasks
${ISAACSIM}/kit/python/bin/python3 -m pip install -e source/isaaclab_rl
```

### BHL 설치 (5.1.0 환경)

```bash
cd /home/laba/Berkeley-Humanoid-Lite
${ISAACSIM}/kit/python/bin/python3 -m pip install -e source/berkeley_humanoid_lite
```

### rsl-rl 5.0.1 설치 (기존 3.0.1 교체)

```bash
${ISAACSIM}/kit/python/bin/python3 -m pip install rsl-rl-lib==5.0.1
```

---

## 해결한 문제들

### 문제 1: `ModuleNotFoundError: No module named 'pxr'`

**원인**: `site-packages/isaacsim/__init__.py`의 `bootstrap_kernel()`이 kit Python 경로를 체크하지 않고 무조건 `aarch_preload_checking()` → LD_PRELOAD 검증 실패 → `sys.exit()`.

`python_packages/isaacsim` (standalone 버전)은 kit Python인 경우 체크를 skip하는 반면,
pip 설치된 `site-packages/isaacsim`은 이 skip 로직이 없어 항상 실패한다.

**해결**: `python_packages`를 PYTHONPATH 맨 앞에 추가 → standalone `isaacsim`이 pip 버전보다 먼저 로드됨.

```bash
export PYTHONPATH="${ISAACSIM}/python_packages:${PYTHONPATH}"
```

`pxr`은 SimulationApp 시작 후 kit extension 시스템이 자동 로드 (`pxr.__file__ = None`).

---

### 문제 2: `AttributeError: 'torch.device' object has no attribute 'is_cpu'`

**원인**: Warp 1.8.2의 `to_torch()`가 `a.device.is_cpu`를 체크하는데, torch 2.7.0+cpu에는 `torch.device.is_cpu` 속성이 없음.

추가로, IsaacLab 2.3.2에서는 `asset.data.*`가 이미 torch.Tensor라서 `wp.to_torch(tensor)` 호출 시 `a.device`가 `warp.Device`가 아닌 `torch.device`가 되어 동일 오류 발생.

**해결 A**: Warp torch.py 패치

파일: `/home/laba/IsaacSim/_build/linux-aarch64/release/extscache/omni.warp.core-1.8.2+la64/warp/torch.py` (line 329)

```python
# 수정 전
if a.device.is_cpu:

# 수정 후
_dev = a.device
_is_cpu = _dev.is_cpu if hasattr(_dev, "is_cpu") else (getattr(_dev, "type", None) == "cpu")
if _is_cpu:
```

**해결 B**: BHL `events.py`에서 타입 분기 처리

파일: `tasks/locomotion/velocity/mdp/events.py`

```python
# 수정 전
default_pos = wp.to_torch(asset.data.default_joint_pos)

# 수정 후
_raw = asset.data.default_joint_pos
if isinstance(_raw, wp.array):
    default_pos = wp.to_torch(_raw)
else:
    default_pos = _raw  # IsaacLab 2.3.2: 이미 torch.Tensor
```

---

### 문제 3: rsl-rl 5.0.1 config 포맷 불일치

**원인**:
- BHL config: deprecated `policy = RslRlPpoActorCriticCfg(class_name="ActorCritic", ...)` 포맷
- rsl-rl 5.0.1: `"ActorCritic"` 클래스 제거 → `"MLPModel"`로 대체
- IsaacLab 2.3.2의 `isaaclab_rl`에는 `handle_deprecated_rsl_rl_cfg()` 함수 없음

**해결**: `play_511.py`에 직접 변환 함수 구현

```python
def handle_cfg_for_rsl_rl_v5(agent_cfg):
    cfg_dict = agent_cfg.to_dict()
    policy = cfg_dict.pop("policy", None)
    if policy is not None:
        cfg_dict["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("actor_hidden_dims", [256, 256, 256]),
            "activation": policy.get("activation", "elu"),
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": policy.get("init_noise_std", 1.0),
            },
        }
        cfg_dict["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("critic_hidden_dims", [256, 256, 256]),
            "activation": policy.get("activation", "elu"),
            "obs_normalization": False,
        }
    return cfg_dict
```

---

### 문제 4: CUDA 체크포인트를 CPU에서 로드 불가

**원인**: 학습은 6.0.0 + GPU에서 진행 → 체크포인트가 CUDA 텐서.
5.1.0 kit Python은 CPU-only torch → `RuntimeError: Attempting to deserialize object on a CUDA device`.

**해결**: `map_location` 파라미터 지정

```python
ppo_runner.load(resume_path, map_location=agent_cfg.device)  # "cpu"
```

---

## 최종 실행 명령어

```bash
ISAACSIM=/home/laba/IsaacSim/_build/linux-aarch64/release
TORCH_GOMP=$(ls ${ISAACSIM}/kit/python/lib/python3.11/site-packages/torch.libs/libgomp*.so* | head -1)
SITE=${ISAACSIM}/kit/python/lib/python3.11/site-packages

export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:${TORCH_GOMP}"
export PYTHONPATH="${ISAACSIM}/python_packages:${SITE}/isaaclab/source/isaaclab:${SITE}/isaaclab/source/isaaclab_assets:${SITE}/isaaclab/source/isaaclab_tasks:${SITE}/isaaclab/source/isaaclab_rl"
source ${ISAACSIM}/setup_python_env.sh
export DISPLAY=:1
export PYTHONUNBUFFERED=1

cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
${ISAACSIM}/python.sh play_511.py \
  --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
  --num_envs 1 \
  --device cpu \
  --load_run 2026-03-27_14-36-49 \
  --checkpoint model_5999.pt
```

---

## 수정된 파일 목록

| 파일 | 수정 내용 |
|------|----------|
| `extscache/omni.warp.core-1.8.2+la64/warp/torch.py` (line 329) | `is_cpu` 속성 없는 device 처리 |
| `berkeley_humanoid_lite/tasks/.../mdp/events.py` | warp array / torch.Tensor 분기 처리 |
| `scripts/rsl_rl/play_511.py` | 5.1.0 전용 play 스크립트 (신규 생성) |

---

## play_511.py 핵심 구조

파일: `/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/play_511.py`

```python
"""Play script for Isaac Sim 5.1.0 + IsaacLab 2.3.2 (GUI visualization on DGX monitor)."""

# AppLauncher 먼저 초기화 (pxr import 전에 필수)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AppLauncher 이후 import
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
import berkeley_humanoid_lite.tasks
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    # NO Newton in 5.1.0 — PhysX 사용
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.device = args_cli.device  # "cpu"

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg_dict = handle_cfg_for_rsl_rl_v5(agent_cfg)
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, map_location=agent_cfg.device)  # CPU로 로드

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    env.close()
```
