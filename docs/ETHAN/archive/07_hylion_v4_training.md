# Hylion v4 걷기 학습 — 진행 기록 (2026-04-01)

## 개요

`hylion_v4.urdf` (BHL 다리 + SO-ARM101 팔 × 2, SO-ARM mount y=±0.12로 이동)를
Newton 백엔드로 걷기 학습하고, Isaac Sim 5.1.0에서 시각화.

---

## v3 → v4 변경사항

| 항목 | hylion_v3 | hylion_v4 |
|------|-----------|-----------|
| SO-ARM mount y | ±0.10 m | ±0.12 m |
| SO-ARM mount z | 0.544 m | 0.82 m |
| base mesh | 없음(임시) | original `base_visual.stl` |
| BHL collision box | `x=±0.075, y=±0.07, z=0.595~0.825` | 동일 |
| SO-ARM 충돌 | hip/knee와 항상 접촉 → 상수 패널티 -2.0 | y=±0.12 → 충돌박스 밖으로 이탈 |

**v4 전략**: torso 충돌 박스(`y=±0.07`)보다 바깥인 `y=±0.12`에 SO-ARM 마운트
→ `undesired_contacts` 패널티 재활성화 가능 (v4에서는 비활성 유지, 추후 검토)

---

## 파일 위치

```
δ3/
  robot/
    hylion_v3.urdf          ← v3 복사본 (y=±0.16 수정본, 참고용)
    hylion_v4.urdf          ← v4 학습 대상 (δ1&ε2/urdf/hylion_v4.urdf 복사)
  usd/
    hylion_v4/              ← Isaac Sim 6.0.0 변환본 (Newton 학습용)
      hylion_v4/
        hylion_v4.usda
        payloads/
          base.usda
          materials.usda
          robot.usda
          Physics/
            physics.usda
            physx.usda
            mujoco.usda
    hylion_v4.usd           ← Isaac Sim 5.1.0 변환본 (시각화용)
  hylion/
    robot_cfg.py            ← USD 경로: δ3/usd/hylion_v4/hylion_v4/hylion_v4.usda
    env_cfg.py              ← v3와 동일 (SO-ARM 관련 조정 유지)
    agents/
      rsl_rl_ppo_cfg.py
  scripts/
    convert_v4_urdf.py      ← 6.0.0용 USD 변환 스크립트
    train_hylion.py         ← 학습 진입점 (v3/v4 공용)
  hylion_v4_train.log       ← v4 학습 로그
  play_hylion_511.log       ← 시각화 실행 로그
```

---

## Step 1: v4 URDF 복사

```bash
cp "/home/laba/project_singularity/δ1 & ε2/urdf/hylion_v4.urdf" \
   /home/laba/project_singularity/δ3/robot/hylion_v4.urdf
```

---

## Step 2: USD 변환

### 2-a. 학습용 USD (Isaac Sim 6.0.0 → Newton 학습에 사용)

```bash
source /home/laba/env_isaaclab/bin/activate
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
  python /home/laba/project_singularity/δ3/scripts/convert_v4_urdf.py
```

출력: `δ3/usd/hylion_v4/hylion_v4/hylion_v4.usda`

> **API**: `isaacsim.asset.importer.urdf.URDFImporter` (6.0.0 신규 API)
> 이 USD는 `NewtonArticulationRootAPI` 포함 → Newton 학습 전용
> **시각화 불가**: mesh geometry 없이 Cylinder/Cube 기본 도형만 포함

### 2-b. 시각화용 USD (Isaac Sim 5.1.0 → IsaacLab UrdfConverter)

```bash
ISAACSIM=/home/laba/IsaacSim/_build/linux-aarch64/release
TORCH_GOMP=$(ls ${ISAACSIM}/kit/python/lib/python3.11/site-packages/torch.libs/libgomp*.so* | head -1)
SITE=${ISAACSIM}/kit/python/lib/python3.11/site-packages

export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:${TORCH_GOMP}"
export PYTHONPATH="${ISAACSIM}/python_packages:${SITE}/isaaclab/source/isaaclab:${SITE}/isaaclab/source/isaaclab_assets:${SITE}/isaaclab/source/isaaclab_tasks:${SITE}/isaaclab/source/isaaclab_rl"
source ${ISAACSIM}/setup_python_env.sh

${ISAACSIM}/python.sh \
  /home/laba/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_assets/scripts/convert_urdf_to_usd.py \
  /home/laba/project_singularity/δ3/robot/hylion_v4.urdf \
  --headless
```

출력: `δ3/usd/hylion_v4.usd`
(스크립트 규칙: `URDF 위치/../../usd/이름.usd` → `δ3/robot/../usd/hylion_v4.usd`)

---

## Step 3: robot_cfg.py 업데이트

`δ3/hylion/robot_cfg.py`의 `HYLION_V3_USD_PATH`를 학습용 USD로 설정:

```python
HYLION_V3_USD_PATH = "/home/laba/project_singularity/δ3/usd/hylion_v4/hylion_v4/hylion_v4.usda"
```

---

## Step 4: 학습 실행

```bash
cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
source /home/laba/env_isaaclab/bin/activate

PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
  python /home/laba/project_singularity/δ3/scripts/train_hylion.py \
    --task Velocity-Hylion-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 6000 \
    --pretrained_checkpoint \
      /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/biped/2026-03-27_14-36-49/model_5999.pt
```

로그 확인:
```bash
tail -f /home/laba/project_singularity/δ3/hylion_v4_train.log
```

### 학습 결과 (2026-04-01)

| 항목 | 값 |
|------|----|
| 소요 시간 | 1h 30m |
| 총 iteration | 6000 (model_9900.pt 저장) |
| 물리 백엔드 | Newton |
| num_envs | 4096 |
| iteration당 시간 | ~0.93s |
| 최종 Mean reward | -0.15 ~ -3.5 (진동) |
| feet_air_time (최종) | ~0.002 (낮음 — 발을 거의 안 들음) |
| 체크포인트 경로 | `BHL/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-01_17-03-23/` |

> **⚠️ 주의**: feet_air_time이 0.002 수준으로 낮아 실제 걷기보다는 자세 유지 수준일 가능성 있음.
> 시각화로 확인 필요.

---

## Step 5: 시각화 (Isaac Sim 5.1.0)

### 사전 준비

5.1.0 Python 환경에 `isaaclab_newton` 설치 (최초 1회):

```bash
ISAACSIM=/home/laba/IsaacSim/_build/linux-aarch64/release
${ISAACSIM}/kit/python/bin/python3 -m pip install -e /home/laba/IsaacLab/source/isaaclab_newton
```

### 실행

```bash
ISAACSIM=/home/laba/IsaacSim/_build/linux-aarch64/release
TORCH_GOMP=$(ls ${ISAACSIM}/kit/python/lib/python3.11/site-packages/torch.libs/libgomp*.so* | head -1)
SITE=${ISAACSIM}/kit/python/lib/python3.11/site-packages

export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:${TORCH_GOMP}"
export PYTHONPATH="${ISAACSIM}/python_packages:${SITE}/isaaclab/source/isaaclab:${SITE}/isaaclab/source/isaaclab_assets:${SITE}/isaaclab/source/isaaclab_tasks:${SITE}/isaaclab/source/isaaclab_rl"
source ${ISAACSIM}/setup_python_env.sh
export DISPLAY=:1

cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
${ISAACSIM}/python.sh play_hylion_511.py \
  --task Velocity-Hylion-v0 \
  --num_envs 1 \
  --device cpu
```

play 스크립트: `/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/play_hylion_511.py`

### play_hylion_511.py 핵심 동작

1. 5.1.0용 USD(`δ3/usd/hylion_v4.usd`)를 `robot_cfg` 경로에 패치
2. `hylion` 패키지 로드 → `Velocity-Hylion-v0` gym 등록
3. model_9900.pt 로드 (`map_location="cpu"`)
4. PhysX로 inference 실행 (5.1.0은 Newton 미지원)

---

## 트러블슈팅

### USD 변환 API 오류 (`UNSUPPORTED_IMPORT_FORMAT`)

6.0.0에서 `omni.kit.asset_converter`로 URDF 변환 시도 → 실패.
**해결**: `isaacsim.asset.importer.urdf.URDFImporter` 사용.

### `Available strings: []` (5.1.0에서 v3 USD 사용 시)

δ1&ε2/usd/hylion_v3은 `instances.usda` + `geometries.usd` 참조 구조로
IsaacLab 2.3.2가 articulation을 파싱하지 못함.
**해결**: 5.1.0 UrdfConverter로 별도 변환한 `hylion_v4.usd` 사용.

### `argument --checkpoint: conflicting option string`

play 스크립트에서 `--checkpoint` 추가 시 `cli_args.add_rsl_rl_args`와 충돌.
**해결**: `--hylion_ckpt`로 이름 변경.

### `Could not resolve 'ActorCritic'`

5.1.0 rsl_rl에는 `ActorCritic` 클래스가 없고 `MLPModel`만 있음.
훈련된 체크포인트 구조: `actor_state_dict` / `critic_state_dict` (MLPModel 포맷).
**해결 진행 중**: `handle_cfg_for_rsl_rl_v5` 함수 수정 필요.

---

## 현재 상태 (2026-04-01 기준)

- [x] hylion_v4.urdf 복사 (δ3/robot/)
- [x] 6.0.0용 USD 변환 완료 (학습용)
- [x] 5.1.0용 USD 변환 완료 (시각화용)
- [x] v4 걷기 학습 완료 (6000 iter, 1h 30m)
- [ ] 시각화 확인 — `ActorCritic` resolve 오류 해결 중
- [ ] feet_air_time 개선 여부 시각화로 판단
