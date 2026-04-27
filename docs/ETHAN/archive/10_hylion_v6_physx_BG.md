# Hylion v6 PhysX 학습 — BG 작업 파일 안내

> **작성자**: BG (별도 작업자)
> **작성일**: 2026-04-02
> **기존 파일 수정 없음** — 아래 파일은 모두 신규 생성본입니다.

---

## 작업 목적

기존 Newton 백엔드(`train_hylion.py`, `Velocity-Hylion-v0`) 설정을 유지하면서,
`hylion_v6.urdf` 기반 모델을 **PhysX 백엔드**로 학습하기 위한 별도 파일 세트를 추가함.

---

## 생성된 파일 목록

### 1. `hylion/robot_cfg_BG.py`

```
δ3/hylion/robot_cfg_BG.py
```

- `robot_cfg.py` 기반, USD 경로만 v6로 교체
- 기본 USD: `δ3/usd/hylion_v6/hylion_v6/hylion_v6.usda`
- 환경변수 `HYLION_USD_PATH`로 경로 override 가능
- `HYLION_CFG_BG` (ArticulationCfg) 정의
- joint 목록, actuator 설정은 v4와 동일 (다리 12 joints, SO-ARM 12 joints)

---

### 2. `hylion/env_cfg_BG.py`

```
δ3/hylion/env_cfg_BG.py
```

- `env_cfg.py`의 모든 Cfg 클래스(Commands/Observations/Actions/Rewards/Terminations/Events/Curriculums)를 그대로 재사용
- `HylionEnvCfg_BG` 정의: `scene.robot`만 `HYLION_CFG_BG`(v6)로 교체
- `train_hylion_physx_BG.py`에서 `Velocity-Hylion-BG-v0` 등록 시 사용

---

### 3. `scripts/train_hylion_physx_BG.py`

```
δ3/scripts/train_hylion_physx_BG.py
```

- `train_hylion.py` 기반, Newton 관련 코드 제거
  - `NewtonCfg()` 미적용 → Isaac Lab 기본 PhysX 백엔드로 동작
  - `physics_material = None` 라인 없음 → 도메인 랜덤화 정상 작동
- `Velocity-Hylion-BG-v0` gym 환경을 스크립트 내에서 직접 등록 (`hylion/__init__.py` 미수정)
- PPO 설정은 기존 `agents/rsl_rl_ppo_cfg.py`의 `HylionPPORunnerCfg` 재사용
- `--hylion_usd_path` 인자로 USD 경로 override 가능

> **주의**: DGX Spark(aarch64)에서 PhysX는 CPU로 동작함.
> GPU 물리 가속이 필요하면 기존 `train_hylion.py` (Newton) 사용 권장.

---

### 4. `scripts/train_hylion_physx_BG.sh`

```
δ3/scripts/train_hylion_physx_BG.sh
```

- 실행 래퍼 스크립트
- 기본 설정: `--num_envs 4096`, `--max_iterations 6000`, `--headless`
- 로그 출력: `/tmp/hylion_physx_BG_train.log`
- 체크포인트 저장: `BHL/scripts/rsl_rl/logs/rsl_rl/hylion/<timestamp>/`

---

## 실행 방법

```bash
bash /home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.sh
```

또는 직접:

```bash
cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
source /home/laba/env_isaaclab/bin/activate
PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
  python /home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py \
    --task Velocity-Hylion-BG-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 6000
```

로그 확인:

```bash
tail -f /tmp/hylion_physx_BG_train.log
```

---

## 기존 파일과의 관계

| 기존 (수정 없음) | BG 대응 파일 | 변경 내용 |
|-----------------|-------------|----------|
| `hylion/robot_cfg.py` | `hylion/robot_cfg_BG.py` | USD 경로: v4 → v6 |
| `hylion/env_cfg.py` | `hylion/env_cfg_BG.py` | robot: `HYLION_CFG` → `HYLION_CFG_BG` |
| `scripts/train_hylion.py` | `scripts/train_hylion_physx_BG.py` | Newton 제거, task ID 변경 |
| `scripts/train_hylion.sh` | `scripts/train_hylion_physx_BG.sh` | 위 스크립트 호출 |
| `hylion/__init__.py` | (미수정) | `Velocity-Hylion-BG-v0`은 train 스크립트에서 직접 등록 |

---

## 전제 조건

- `δ3/usd/hylion_v6/hylion_v6/hylion_v6.usda` 존재 확인 ✅
- `δ3/robot/hylion_v6.urdf` 존재 확인 ✅
- USD 재변환이 필요한 경우: `δ3/scripts/convert_v4_urdf.py` 참고하여 v6용 변환 스크립트 작성
