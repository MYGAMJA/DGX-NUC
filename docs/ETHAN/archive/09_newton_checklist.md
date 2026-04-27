# Newton Execution Checklist (Isaac Sim / IsaacLab)

이 문서는 Newton 백엔드 학습/재생 전에 반드시 확인할 항목을 정리한 실무 체크리스트입니다.

## 1) Physics 엔진 경로 확인

- 학습 스크립트에서 Newton이 실제로 적용되는지 확인.
- 본 프로젝트에서는 `train_hylion.py`에서 `env_cfg.sim.physics = NewtonCfg()`가 적용되어야 함.

확인 명령:

```bash
grep -n "NewtonCfg\|sim.physics" /home/laba/project_singularity/δ3/scripts/train_hylion.py
```

## 2) 태스크 ID 정확성 확인

- 오타/옛 태스크 이름 사용 시 즉시 종료됨.
- 다리-only 권장 시작 태스크:
  - `Velocity-Berkeley-Humanoid-Lite-Biped-v0`
- Hylion 상체 포함 태스크:
  - `Velocity-Hylion-v0`

확인 명령:

```bash
grep -Rns --include='*.py' "gym.register\|Velocity-" /home/laba/Berkeley-Humanoid-Lite/source | head -n 40
```

## 3) 환경 변수 충돌 제거

- Python 경로 충돌은 NaN/ImportError/비정상 종료 원인이 됨.
- 실행 전 아래 3가지를 고정:
  - `source /home/laba/env_isaaclab/bin/activate`
  - `unset PYTHONPATH PYTHONHOME`
  - `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`

## 4) 먼저 다리-only로 안정성 검증

- 상체 포함 전에 다리-only로 Newton 안정 실행 확인.
- 최소 200~500 iteration 동안 다음이 없어야 함:
  - `Traceback`
  - `ValueError: ... contains NaN values`

권장 실행:

```bash
cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
source /home/laba/env_isaaclab/bin/activate
unset PYTHONPATH PYTHONHOME
PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
/home/laba/env_isaaclab/bin/python /home/laba/project_singularity/δ3/scripts/train_hylion.py \
  --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
  --num_envs 4096 --headless --max_iterations 6000 \
  > /home/laba/project_singularity/δ3/biped_newton.log 2>&1
```

## 5) NaN 조기 감시

- NaN은 손실보다 관측값에서 먼저 터지는 경우가 많음.
- 아래 문자열을 주기적으로 확인:
  - `contains NaN values`
  - `Traceback`
  - `Mean value loss: nan`

빠른 점검:

```bash
grep -n "Traceback\|contains NaN values\|Mean value loss: nan\|Mean surrogate loss: nan" /home/laba/project_singularity/δ3/biped_newton.log | tail -n 20
```

## 6) 체크포인트 생성 확인

- 학습이 진짜 진행 중인지 확인하는 가장 확실한 기준.
- run_dir가 찍히고 `model_*.pt`가 증가해야 정상.

확인 명령:

```bash
run_dir=$(grep -oE "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/[a-zA-Z_-]+/[0-9_-]+" /home/laba/project_singularity/δ3/biped_newton.log | tail -1)
echo "$run_dir"
ls -1 "$run_dir"/model_*.pt 2>/dev/null | tail -n 10
```

## 7) 상체 포함(Hylion)은 단계적 전환

- 권장 순서:
  1. 다리-only Newton 안정 학습 확인
  2. 안정 체크포인트로 Hylion `--stable_walk` 재학습
  3. 외란/랜덤성은 낮은 수준에서 시작
- 상체 포함에서 NaN 발생 시, 엔진 자체보다 설정 난이도 급상승을 먼저 의심.

## 8) 뉴턴 결과 시각화 전략 분리

- 안정 전략:
  - 학습은 Newton
  - 결과 확인은 headless record(mp4)
- 실시간 GUI는 버전/확장 호환성 영향이 큼.

재생 스크립트:

```bash
bash /home/laba/project_singularity/δ3/scripts/newton_record_playback.sh <checkpoint_path> 600 /home/laba/project_singularity/δ3/videos/newton_playback
```

## 9) 이슈 #555 관련 주의점 (Newton 6.0-dev2)

다음 증상이 보이면 엔진 통합 이슈 가능성도 함께 점검:

- static collider 관련 viewport freeze
- `Model.collide()` kwargs mismatch
- reentrant init guard stuck
- `shape_gap=0.1`로 인한 비접촉
- SDF parent scale 변환 이슈
- MuJoCo-Warp + EGL CUDA context 충돌

참고: https://github.com/isaac-sim/IsaacSim/issues/555

## 10) 실전 판정 기준 (Go / No-Go)

- Go:
  - 프로세스 유지 + iteration 증가
  - 최근 200라인 내 NaN/Traceback 없음
  - 체크포인트 파일 생성 증가
- No-Go:
  - `contains NaN values` 발생
  - 반복 즉시 중단/재시작 루프
  - 체크포인트 정지

No-Go 시 우선순위:

1. 태스크/경로/환경변수 재확인
2. 다리-only로 재검증
3. 상체 포함 설정(리셋 외란/명령 범위) 완화
4. 시각화와 학습 파이프라인 분리
