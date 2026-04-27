# Desktop PhysX v6 학습 환경 세팅 로그

> **작성일**: 2026-04-03
> **머신**: Windows 10 Home, i9-11900K, RTX 3090 24GB, RAM 32GB, CUDA 12.6

---

## 목표

Hylion v6 URDF 기반 모델을 **PhysX 백엔드**로 이 데스크탑에서 RL 학습 실행.

---

## 1차 시도: Windows venv 직접 설치 — ❌ 실패

- isaacsim 6.0.0 pip 패키지는 Windows에서 `SimulationApp`을 로드할 수 없음
- Kit 엔진이 Linux 환경을 전제로 동작
- 환경 삭제 완료

---

## 2차 시도: Docker — 보류

- `nvcr.io/nvidia/isaac-sim:6.0.0-dev2` (28.5GB) pull 후 용량 문제로 삭제

---

## 3차 시도: WSL2 Ubuntu (root 유저) — ❌ 실패

- `isaacsim` 기본 + extscache 개별 설치 방식
- `SimulationApp = None` 해결 안 됨 (extscache-kit, extscache-kit-sdk 설치해도)
- WSL 인스턴스 리셋 후 4차 시도로 전환

---

## 4차 시도: WSL2 Ubuntu (laba 유저, isaacsim[rl]) — ⏳ 진행 중

### WSL 재설치

```
wsl --install Ubuntu-22.04
```

- 유저: `laba`
- Python 3.12.13 (deadsnakes PPA, 이전 인스턴스에서 잔존)

### 설치 완료 항목

| 항목 | 경로 / 버전 | 상태 |
|------|-------------|------|
| Ubuntu | 22.04.5 LTS (WSL2) | ✅ |
| GPU | RTX 3090, CUDA 12.6 | ✅ nvidia-smi 확인 |
| Python | 3.12.13 | ✅ |
| venv | `/home/laba/env_isaaclab` | ✅ |
| isaacsim[rl] | 6.0.0 (풀 설치) | ✅ **SimulationApp OK** |
| IsaacLab | v3.0.0-beta (editable) | ✅ `/home/laba/IsaacLab` |
| BHL | editable + submodules | ✅ `/home/laba/Berkeley-Humanoid-Lite` |
| BHL assets | editable | ✅ |
| rsl-rl-lib | 5.0.1 | ✅ |
| warp-lang | 1.12.0 | ✅ |
| hydra-core | 1.3.2 | ✅ |
| gymnasium | 1.2.3 | ✅ |
| EULA | 동의 완료 | ✅ |

> **핵심**: `isaacsim[rl]` 풀 설치가 SimulationApp 문제를 해결함.
> 이전에 `isaacsim` 기본 + extscache 개별 설치로는 안 됐음.

### BHL API 호환성 패치 — 3건 적용

1. `AdditiveUniformNoiseCfg` → `UniformNoiseCfg` (biped/env_cfg.py, humanoid/env_cfg.py)
2. `self.sim.physx.gpu_max_rigid_patch_count` 주석 처리 (velocity_env_cfg.py)
3. BHL submodule URL: SSH → HTTPS 변경 (git config)

### IsaacLab 패치 — 1건 적용

**`articulation_data.py` CPU→GPU device 패치**

```
파일: IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py
```

PhysX가 `get_dof_positions()`, `get_dof_velocities()`에서 CPU warp array를 반환하는데,
IsaacLab은 GPU device에서 warp 커널을 실행하려 해서 device 불일치 에러 발생.
→ `.to(self.device)` 래핑 추가.

```python
# 수정 전:
self._joint_vel.data = self._root_view.get_dof_velocities()

# 수정 후:
_vel_data = self._root_view.get_dof_velocities()
self._joint_vel.data = _vel_data.to(self.device) if hasattr(_vel_data, "to") \
    and str(getattr(_vel_data, "device", self.device)) != str(self.device) else _vel_data
```

> **원인**: PhysX GPU pipeline이 비활성 상태 (`/physics/useGPU=None`).
> 시뮬레이션은 `sim.device=cuda:0`이지만, PhysX scene이 CPU로 실행되어 데이터가 CPU에 위치.
> 이 패치는 데이터를 GPU로 이동시키는 workaround. 다른 `get_*` 호출에서도 동일 문제 가능.

### 현재 남은 문제

#### 문제 1: PhysX GPU pipeline 비활성

```python
# carb settings 확인 결과:
/physics/cudaDevice = 0      # GPU 인식됨
/physics/useGPU = None        # ← GPU simulation 비활성!
/physics/simulationDevice = None
```

- PhysX가 CPU로 시뮬레이션 → 데이터가 CPU에 위치 → GPU 커널과 device 불일치
- `articulation_data.py` 패치로 `joint_pos`, `joint_vel`은 해결했지만, 다른 곳에서도 같은 에러 가능
- **DGX에서는 이 문제 없었음** (GPU pipeline 자동 활성화)

#### 문제 2: Hylion v6 USD 계층 구조 불일치 (hylion 전용)

```
BHL biped USD:  /robot/base, /robot/leg_left_ankle_roll   (flat, 1레벨)
Hylion v6 USD:  /robot/Geometry/base/.../ankle_roll       (nested, 6레벨)
```

- IsaacLab `ContactSensorCfg(prim_path=".../robot/.*")`가 1레벨만 검색
- BHL biped USD는 flat → 정상 동작
- Hylion v6 USD는 nested → contact sensor body 못 찾음
- **해결 필요**: URDF→USD 재변환 (IsaacLab UrdfConverter로) 또는 contact sensor 패치

### 다음 단계: VS Code WSL 연결

**현재 문제**: Windows에서 `wsl -d Ubuntu-22.04 -- bash -c '...'`로 명령 전달하는 방식의 한계:
- pipe/background 처리 불안정 (nohup, &, | 조합 안 됨)
- 실시간 로그 모니터링 불가 (stdout 버퍼링)
- interactive 입력 불가 (EULA 등)
- 긴 명령 실행 시 타임아웃

**해결**: VS Code Remote - WSL 확장으로 WSL에 직접 연결

```
1. VS Code에서 Ctrl+Shift+P → "WSL: Connect to WSL using Distro..."
2. Ubuntu-22.04 선택
3. 터미널이 WSL 안에서 네이티브 동작
4. Claude Code도 WSL 환경에서 직접 실행 가능
```

이렇게 하면:
- 터미널 = WSL bash (네이티브)
- 파일 시스템 = `/home/laba/...` 직접 접근
- 학습 실행/모니터링이 안정적
- pipe, background, nohup 정상 동작

---

## 참고: 검토한 대안

| 방법 | 장점 | 단점 | 상태 |
|------|------|------|------|
| Windows venv | 가장 간단 | SimulationApp 로드 불가 | ❌ 실패 |
| Docker | 완전한 환경 | 28.5GB 이미지 크기 | 보류 |
| WSL2 (root, 개별설치) | — | SimulationApp None | ❌ 실패 |
| WSL2 (laba, isaacsim[rl]) | SimulationApp OK | PhysX GPU pipeline 미활성 | ⏳ 진행 중 |
| Omniverse Launcher | 공식 지원 | 별도 앱 설치 필요 | 미시도 |

---

## 설치 재현 명령어 (4차 시도 기준)

```bash
# Step 1: venv
python3.12 -m venv ~/env_isaaclab
source ~/env_isaaclab/bin/activate
pip install --upgrade pip setuptools wheel

# Step 2: isaacsim 풀 설치 (핵심: [rl] extra)
pip install --extra-index-url https://pypi.nvidia.com "isaacsim[rl]==6.0.0.0"

# Step 3: IsaacLab v3.0.0-beta
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && git checkout v3.0.0-beta
pip install --no-deps -e source/isaaclab -e source/isaaclab_assets \
  -e source/isaaclab_tasks -e source/isaaclab_rl -e source/isaaclab_physx

# Step 4: 누락 패키지 (--no-deps 때문)
pip install lazy_loader gymnasium hydra-core tensorboard tqdm \
  prettytable h5py einops warp-lang==1.12.0

# Step 5: BHL
git clone --depth 1 https://github.com/HybridRobotics/Berkeley-Humanoid-Lite.git ~/Berkeley-Humanoid-Lite
cd ~/Berkeley-Humanoid-Lite
git config submodule.source/berkeley_humanoid_lite_assets.url \
  https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets.git
git config submodule.source/berkeley_humanoid_lite_lowlevel.url \
  https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Lowlevel.git
git submodule update --init --depth 1
pip install --no-deps -e source/berkeley_humanoid_lite \
  -e source/berkeley_humanoid_lite_assets
pip install rsl-rl-lib==5.0.1

# Step 6: BHL 호환성 패치
sed -i 's/AdditiveUniformNoiseCfg/UniformNoiseCfg/g' \
  ~/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg.py \
  ~/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/humanoid/env_cfg.py
sed -i 's/self.sim.physx.gpu_max_rigid_patch_count/# self.sim.physx.gpu_max_rigid_patch_count/' \
  ~/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/velocity_env_cfg.py
```
