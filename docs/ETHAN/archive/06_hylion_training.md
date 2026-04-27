# Hylion v3 걷기 학습 가이드

## 개요

hylion_v3.urdf (BHL 다리 + SO-ARM 팔 × 2) 를 Newton 백엔드로 걷기 학습.

- **Policy**: 다리 12 joints만 제어 (BHL biped와 동일)
- **SO-ARM joints**: stiff actuator (stiffness=50)로 기본 자세 고정, policy 제외
- **물리 백엔드**: Newton GPU (Isaac Sim 6.0.0)
- **체크포인트 저장**: `BHL/scripts/rsl_rl/logs/rsl_rl/hylion/<timestamp>/`

---

## 파일 구조

```
δ3/
  hylion/
    __init__.py          ← gym.register("Velocity-Hylion-v0")
    robot_cfg.py         ← HYLION_CFG ArticulationCfg
    env_cfg.py           ← HylionEnvCfg (BHL biped 기반)
    agents/
      rsl_rl_ppo_cfg.py  ← HylionPPORunnerCfg
  scripts/
    convert_urdf.sh      ← URDF → USD 변환
    train_hylion.py      ← 학습 진입점
    train_hylion.sh      ← 학습 실행 셸 스크립트
```

---

## Step 1: URDF → USD 변환

```bash
bash /home/laba/project_singularity/δ3/scripts/convert_urdf.sh
```

출력: `/home/laba/project_singularity/δ1 & ε2/usd/hylion_v3.usd`

> **주의**: δ3/hylion/robot_cfg.py 의 `HYLION_V3_USD_PATH`가 이 경로를 가리킨다.

---

## Step 2: 학습 실행

```bash
bash /home/laba/project_singularity/δ3/scripts/train_hylion.sh
```

또는 직접 실행:

```bash
cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
source /home/laba/env_isaaclab/bin/activate
PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
  python /home/laba/project_singularity/δ3/scripts/train_hylion.py \
    --task Velocity-Hylion-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 6000
```

로그 확인:
```bash
tail -f /tmp/hylion_train.log
```

---

## BHL biped 대비 변경사항

| 항목 | BHL biped | Hylion v3 |
|------|-----------|-----------|
| Robot USD | berkeley_humanoid_lite_biped.usd | hylion_v3.usd |
| 총 revolute joints | 12 | 24 (다리 12 + SO-ARM 12) |
| Policy 제어 joints | 12 (다리) | 12 (다리만) |
| SO-ARM | 없음 | stiffness=50으로 고정 |
| base 질량 | 4.44 kg | 4.44 + SO-ARM ~1.6 kg |
| Gym task ID | Velocity-Berkeley-Humanoid-Lite-Biped-v0 | Velocity-Hylion-v0 |
| 실험 이름 | biped | hylion |

---

## 주요 설계 결정

### SO-ARM joints 처리
- policy에서 제외 → 관측/행동 공간 크기 BHL biped와 동일 (관측 45, 행동 12)
- stiffness=50, damping=5 → 팔이 보행 중 흔들리지 않고 기본 자세 유지
- 파라메트릭 직립 테스트(δ3/04)에서 +6 kg까지 안정 확인 → SO-ARM 무게 문제 없음

### train_hylion.py 동작
1. `sys.path`에 `δ3/` 추가
2. `import hylion` → `Velocity-Hylion-v0` gym 등록
3. 이후 BHL train.py와 동일하게 Newton 학습

---

## 트러블슈팅

### URDF 변환 시 mesh 경로 오류
hylion_v3.urdf의 mesh 경로(`../components/...`)가 상대 경로.
IsaacLab UrdfConverter는 URDF 파일 위치 기준으로 상대 경로를 해석하므로 정상 동작해야 함.
오류 시 URDF의 mesh path를 절대 경로로 수정.

### `δ1 & ε2` 경로 공백/특수문자
USD 경로에 공백과 특수문자 포함. 셸 스크립트에서는 따옴표로 처리됨.
Python 코드에서는 문자열 그대로 사용 가능.

### SO-ARM joint limits 초과
URDF의 SO-ARM joint limit 확인 후 init_state joint_pos 조정.
