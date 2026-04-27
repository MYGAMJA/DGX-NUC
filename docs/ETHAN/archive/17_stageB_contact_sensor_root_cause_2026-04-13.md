# Stage-B Contact Sensor 근본 원인 분석 및 수정 보고서 (2026-04-13)

## 1) 핵심 요약

- `feet_air_time`이 항상 0.0000인 근본 원인: ankle_roll 링크에 `PhysxRigidBodyAPI`가 없어서 contact force가 PhysX에서 0으로 반환됨
- 추가 원인: `create_rigid_contact_view`에 LIST를 넘기면 내부적으로 force가 0으로 반환됨
- 두 원인 모두 수정 완료 (contact_sensor.py + physx.usda)
- 현재 debug10 런 진행 중, physx.usda 패치 적용 후 재기동 필요

---

## 2) 왜 BHL은 됐는데 v6는 안 됐나

### BHL biped 구조 (동작함)
```
/World/envs/env_0/robot/
  ├── hip_roll  [PhysicsRigidBodyAPI]  ← activate_contact_sensors 가 직접 접근
  ├── hip_yaw   [PhysicsRigidBodyAPI]
  ├── ...
  └── ankle_roll [PhysicsRigidBodyAPI]  ← 여기도 접근
```
- robot 직계 자식이 링크들 → `activate_contact_sensors`가 각 링크에 `PhysxRigidBodyAPI` + `PhysxContactReportAPI` 추가

### Hylion v6 구조 (문제)
```
/World/envs/env_0/robot/
  └── Geometry/
        └── base  [PhysicsRigidBodyAPI]  ← activate_contact_sensors 여기서 멈춤
              ├── leg_left_hip_roll/
              │     ├── .../
              │     │     └── leg_left_ankle_roll  [PhysxContactReportAPI만 있음]
```
- `activate_contact_sensors`가 `base`(첫 rigid body)에서 하강을 멈춤
- ankle_roll에는 `PhysxContactReportAPI`(physx.usda에서)만 있고 `PhysxRigidBodyAPI`가 없음
- **`PhysxRigidBodyAPI` 없으면 `get_net_contact_forces()`가 0 반환**

---

## 3) 수정된 파일 목록

### A. physx.usda — 핵심 USD 수정
**파일**: `δ3/usd/hylion_v6/hylion_v6/payloads/Physics/physx.usda`

left/right ankle_roll에 `PhysxRigidBodyAPI` 추가:
```usda
over "leg_left_ankle_roll" (
    prepend apiSchemas = ["PhysxContactReportAPI", "PhysxRigidBodyAPI"]
)
{
    float physxContactReport:threshold = 0.0
    float physxRigidBody:sleepThreshold = 0.0   ← 추가
    ...
}
```
(right ankle_roll도 동일 적용)

### B. contact_sensor.py — 리스트 글로브 문제 수정
**파일**: `/home/laba/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py`

주요 변경:
1. `create_rigid_contact_view`에 LIST 넘기면 force=0 반환 → body마다 별도 view 생성
2. per-body view에서 각각 force 읽어서 interleave
3. `_body_names_cache`: prim_paths[:num_sensors] 슬라이싱 버그 우회
4. `_per_body_contact_views`: None으로 초기화, `_invalidate_initialize_callback`에서도 None으로 리셋

핵심 로직 (`_update_buffers_impl`):
```python
if self._per_body_contact_views is not None:
    _forces_per_body = [
        wp.to_torch(cv.get_net_contact_forces(dt=self._sim_physics_dt)).reshape(self._num_envs, 3)
        for cv in self._per_body_contact_views
    ]
    _stacked = torch.stack(_forces_per_body, dim=1)   # (N, B, 3)
    _interleaved = _stacked.contiguous().view(-1, 3)   # (N*B, 3) interleaved
    net_forces_flat = wp.from_torch(_interleaved).view(wp.vec3f)
```

---

## 4) 디버그 런 이력

| 런 | 로그 | 핵심 발견 |
|----|------|-----------|
| debug6 | `/tmp/hylion_v6_physx_debug6.log` | 센서 초기화 성공, feet_air_time=0 확인 |
| debug7 | `/tmp/hylion_v6_physx_debug7.log` | model_9800.pt 로드 → 이미 NaN 상태 |
| debug8 | `/tmp/hylion_v6_physx_debug8.log` | Stage-A model_5999.pt 로드, loss 정상, air_time=0 |
| debug9 | `/tmp/hylion_v6_physx_debug9.log` | 디버그 print 추가 → force_mag_max=0.0000 확인 |
| debug10 | `/tmp/hylion_v6_physx_debug10.log` | per-body view 적용, force=0 여전히 (physx.usda 패치 전) |

---

## 5) 체크포인트 현황

| 경로 | 상태 |
|------|------|
| `biped/2026-04-06_15-27-27/model_5999.pt` | **정상 (Stage-A, 학습 미오염)** ← 재시작 기점 |
| `hylion/2026-04-08_11-17-20/model_7200.pt` | 경계 (NaN 직전) |
| `hylion/2026-04-08_17-04-31/model_9800.pt` | **오염됨 (action_std=0, loss=NaN)** |

---

## 6) 다음 액션

1. **현재 debug10 런 종료** (physx.usda 패치 이전 상태로 실행 중)
2. **재시작**: model_5999.pt (Stage-A) → physx.usda 패치 적용된 상태로
3. **확인 지표**:
   - `feet_air_time` > 0 (드디어 비零 값 확인)
   - `Mean value loss`: NaN 아님
   - `Mean action std` > 0
4. 안정 확인 후 2048-env → 3072-env 상향 검토

## 7) 재시작 명령

```bash
# 현재 런 종료
ls /proc/ | xargs -I{} sh -c 'cat /proc/{}/cmdline 2>/dev/null | tr "\0" " " | grep -q "train_hylion" && echo {}' 2>/dev/null
# 위 출력된 PID들 kill

# 재시작
cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
source /home/laba/env_isaaclab/bin/activate
nohup env PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
  python /home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py \
  --task Velocity-Hylion-BG-v0 \
  --num_envs 2048 \
  --headless \
  --pretrained_checkpoint /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/biped/2026-04-06_15-27-27/model_5999.pt \
  > /tmp/hylion_v6_physx_debug11.log 2>&1 &
```

## 8) 정상 판정 기준

- `feet_air_time` > 0.0001 (0이 아닌 값 나오면 근본 수정 성공)
- `Mean value loss` < 100 and not NaN
- `Mean action std` > 0.1
- `Mean episode length` 점진적 증가 추세
