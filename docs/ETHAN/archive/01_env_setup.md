# 환경 설치 가이드 (Isaac Sim 6.0.0 + IsaacLab + Newton + BHL)

## 설치 환경

| 항목 | 값 |
|------|-----|
| 하드웨어 | DGX Spark (NVIDIA GB10, SM 12.1) |
| OS | Ubuntu (Linux 6.17.0-1014-nvidia, aarch64) |
| Python | 3.12 |
| 가상환경 | `/home/laba/env_isaaclab` |

---

## Step 1: Isaac Sim 6.0.0 설치 (pip)

```bash
pip install isaacsim==6.0.0 --extra-index-url https://pypi.nvidia.com
echo "Yes" | isaacsim  # EULA 동의
```

---

## Step 2: IsaacLab develop 브랜치 설치

```bash
git clone https://github.com/isaac-sim/IsaacLab.git /home/laba/IsaacLab
cd /home/laba/IsaacLab
git checkout develop
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_rl
```

---

## Step 3: Newton physics 설치

```bash
pip install isaaclab-newton==0.5.9
pip install -e source/isaaclab_newton
pip install -e source/isaaclab_physx
```

> **주의**: Newton은 Isaac Sim 6.0.0 + IsaacLab develop 브랜치가 필요하다.
> IsaacLab 2.3.2 (stable)는 Newton을 지원하지 않는다.

---

## Step 4: LD_PRELOAD 설정

DGX Spark(aarch64)에서 libgomp를 먼저 로드해야 한다:

```bash
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
```

---

## Step 5: Berkeley Humanoid Lite (BHL) 설치

```bash
git clone https://github.com/berkeley-humanoid-lite/Berkeley-Humanoid-Lite.git \
  /home/laba/Berkeley-Humanoid-Lite
cd /home/laba/Berkeley-Humanoid-Lite
pip install -e source/berkeley_humanoid_lite
pip install rsl-rl-lib==5.0.1
```

---

## BHL API 호환성 수정 (IsaacLab develop 대응)

BHL은 원래 IsaacLab 2.3.2 기준이라 develop 브랜치와 API 불일치 4건을 수정해야 한다.

### 수정 1: `AdditiveUniformNoiseCfg` → `UniformNoiseCfg`

IsaacLab develop에서 클래스 이름이 변경됨.

```bash
# 적용 파일:
# tasks/locomotion/velocity/config/biped/env_cfg.py
# tasks/locomotion/velocity/config/humanoid/env_cfg.py
sed -i 's/AdditiveUniformNoiseCfg/UniformNoiseCfg/g' <파일경로>
```

### 수정 2: `sim.physx` 속성 제거됨

파일: `tasks/locomotion/velocity/velocity_env_cfg.py`

```python
# 이 줄 주석 처리 (physx 속성이 develop에서 제거됨):
# self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
```

### 수정 3: `default_joint_pos` warp array 처리

파일: `tasks/locomotion/velocity/mdp/events.py`

IsaacLab develop에서 `asset.data.default_joint_pos`는 `wp.array`를 반환한다.

```python
import warp as wp

# 수정 전 (실패):
pos = torch.tensor(asset.data.default_joint_pos, device=asset.device).clone()

# 수정 후 (성공):
default_pos = wp.to_torch(asset.data.default_joint_pos)  # shared memory
pos = default_pos.clone()
# ... 수정 후 ...
default_pos[env_ids, joint_ids] = pos  # warp array에 자동 반영
```

### 수정 4: `rewards.py` — `asset.data.*` warp array 변환

파일: `tasks/locomotion/velocity/mdp/rewards.py`

develop에서 모든 `asset.data.*` 속성이 `wp.array`를 반환하므로 `wp.to_torch()` 래핑 필요.

```python
# 수정 전:
vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), ...)

# 수정 후:
vel_yaw = quat_rotate_inverse(yaw_quat(wp.to_torch(asset.data.root_quat_w)), ...)
```

---

## 핵심 개념: IsaacLab 버전별 데이터 타입 차이

| IsaacLab 2.3.2 | IsaacLab develop |
|----------------|-----------------|
| `asset.data.*` → `torch.Tensor` | `asset.data.*` → `wp.array` |
| 직접 PyTorch 연산 가능 | `wp.to_torch()` 변환 필요 |
| `AdditiveUniformNoiseCfg` | `UniformNoiseCfg` (이름 변경) |
| `sim.physx` 속성 있음 | `sim.physx` 제거됨 |

`wp.to_torch()`는 **shared memory** 방식 — 반환된 텐서를 수정하면 원본 warp array도 수정됨.

---

## Newton 동작 확인 테스트

```bash
source /home/laba/env_isaaclab/bin/activate
LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" python -c "
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless'])
app = AppLauncher(args)
sim_app = app.app

from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym
import isaaclab_tasks  # noqa
env_cfg = parse_env_cfg('Isaac-Cartpole-Direct-v0', device='cuda:0', num_envs=4)
env = gym.make('Isaac-Cartpole-Direct-v0', cfg=env_cfg)
print('[SUCCESS] CartPole env created')
env.close()
sim_app.close()
"
```

성공 확인: `Finalizing model on device: cuda:0`

---

## BHL zero-agent 환경 테스트

```bash
source /home/laba/env_isaaclab/bin/activate
PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
  python scripts/run_bhl_zero_agent.py --num_envs 1
```

성공 출력:
```
[INFO]: Completed setting up the environment...
[INFO]: Observation space: Dict('critic': Box(-inf, inf, (1, 48), float32), 'policy': Box(-inf, inf, (1, 45), float32))
[INFO]: Action space: Box(-inf, inf, (1, 12), float32)
[INFO]: Step 0/100 done.
...
[INFO]: BHL environment test complete!
```

### Newton 백엔드로 실행 시 추가 설정

```python
from isaaclab_newton.physics import NewtonCfg

env_cfg.sim.physics = NewtonCfg()
env_cfg.events.physics_material = None  # PhysX 전용 API, Newton에서 오류 발생
```

Newton 초기화: `Time taken for simulation start: ~19 s` (이후 스텝은 GPU 병렬 실행)
