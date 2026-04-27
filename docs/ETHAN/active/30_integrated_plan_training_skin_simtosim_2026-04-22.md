# Hylion v6 통합 계획: 학습 → 머리 탈 → Sim-to-Sim → NUC 배포 (2026-04-22)

> 참고 사이트:
> - 프로젝트 전체: https://henryhjna.github.io/hylion-3d-guide/
> - BHL 공식 문서: https://berkeley-humanoid-lite.gitbook.io/docs

---

## 1. 현재 상태 (2026-04-22 기준)

| 항목 | 상태 |
|------|------|
| 학습 스테이지 | **D3 진행 중** (±3N, ETA ~2.5시간) |
| 최종 학습 목표 | E4 ±30N |
| 마감 | **2026-04-27** |
| 프로젝트 발표 | **2026-06-01** |
| 현재 체크포인트 | `stage_d2_5_hylion_v6/best.pt` (orientation 2.35%) |

---

## 2. RL 학습 전체 로드맵

```
[완료] D1    ±1N    (3000iter)
[완료] D1.5  ±1.5N  (2000iter)
[완료] D2    ±2N    (3000iter)
[완료] D2.5  ±2.5N  (2000iter)  ← orientation 2.35% ✅
[진행] D3    ±3N    (4000iter)  ← 지금 (ETA ~2.5시간)
[예정] D4    ±5N    (4000iter)  (~7시간)
[예정] D4.5  ±7N    (4000iter)  (~7시간)
[예정] D5    ±10N   (5000iter)  (~9시간)  ← D 단계 완료 기준
       ↓
[예정] E1    ±15N   (5000iter)  (~5시간)
[예정] E2    ±20N   (6000iter)  (~6시간)
[예정] E3    ±25N   (6000iter)  (~6시간)
[예정] E4    ±30N   (7000iter)  (~7시간)  ← 최종 목표 ✨
```

**orchestrator (`run_optionA_training.sh`)가 D5까지 자동 실행.**
E1~E4는 D5 완료 후 `run_e_stages_training.sh` (미작성) 별도 실행 필요.

### 자동 저장 체계
- 각 스테이지 완료 시 → `δ3/checkpoints/stage_dX_hylion_v6/best.pt` 자동 저장
- ⚠️ 버그 수정 완료 (2026-04-22): `find_latest_ckpt()`가 잘못된 경로를 참조하던 문제 수정
  - 수정 전: `/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/...`
  - 수정 후: `/home/laba/project_singularity/logs/rsl_rl/hylion`
- D2.5 best.pt는 수동 저장 완료 (`model_28696.pt` → `best.pt`)

### 성공 기준 (스테이지별)
| 지표 | 기준 |
|------|------|
| orientation 종료율 | < 15% |
| mean reward | > 25.0 |
| episode 평균 길이 | > 300 스텝 |

---

## 3. 머리 탈(외장) 적용 계획

### 결론: **재학습 불필요**

머리 외장은 `visual mesh`만 추가하며, `collision mesh`와 `mass/inertia`는 스켈레톤 기준을 유지한다.

```
스켈레톤 USD (학습용)        외장 적용 USD (데모용)
────────────────────         ───────────────────────
visual:   스켈레톤 mesh   →  visual:   머리 탈 mesh  ← 변경
collision: 스켈레톤 col      collision: 스켈레톤 col  ← 그대로
mass/inertia: 기본값         mass/inertia: 기본값     ← 그대로
```

### 작업 순서
1. 머리 탈 STL/STEP → `δ1 & ε2/components/step_files/` 에 추가
2. Onshape에서 visual-only 파트로 조립
3. USD export: `δ1 & ε2/usd/hylion_v6_skinned.usd` (visual만 교체)
4. `play_hylion_v6_BG.py --hylion_usd_path` 로 확인

### 머리 탈 mass 허용 범위
- D5 학습 기준 `base_mass ±1.5kg` 랜덤 변동 포함
- **머리 탈 무게 < 1.5kg이면 추가 학습 없이 사용 가능**
- > 1.5kg이면 500~1000iter fine-tuning

---

## 4. Sim-to-Sim 검증 계획

### 4-1. Isaac Lab 내 시각화 (즉시 가능)

```bash
# D2.5 best.pt로 바로 확인 가능
cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
source /home/laba/env_isaaclab/bin/activate
DISPLAY=:1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
  python /home/laba/project_singularity/δ3/scripts/play_hylion_v6_BG.py \
    --ckpt_path /home/laba/project_singularity/δ3/checkpoints/stage_d2_5_hylion_v6/best.pt \
    --task Velocity-Hylion-BG-v0 \
    --num_envs 1 \
    --lin_vel_x 0.3
```

### 4-2. Gazebo Sim-to-Sim 검증 (NUC 이전 전 교차 검증)

**목적**: PhysX(학습) ↔ Gazebo(실제 환경과 유사) 물리 차이 확인

**절차**:
```
① URDF 준비
   - 학습에 사용한 hylion_v6.urdf 사용 (δ1 & ε2/urdf/hylion_v6.urdf)
   - collision mesh 동일하게 유지

② ROS2 + Gazebo 환경에서 URDF 로드
   ros2 launch hylion_description gazebo.launch.py

③ policy → ROS2 bridge
   - 관측값(obs): /joint/state, /imu/data → policy 입력 (45차원)
   - 행동(action): policy 출력 → /joint/cmd (12 DOF)
   - 루프 주기: 50Hz (Isaac Lab 학습 주기와 동일)

④ 속도 명령 테스트
   ros2 topic pub /gait/cmd geometry_msgs/msg/Twist \
     "{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {z: 0.0}}"

⑤ 외력 인가 테스트
   Gazebo apply_body_wrench 로 ±10N, ±30N 인가
```

**성공 기준**:
- Gazebo에서도 Isaac Lab 대비 동일한 보행 패턴
- orientation 종료 없이 10초 이상 유지
- 외력 인가 후 2초 내 복원

### 4-3. Sim-to-Sim 검증 일정

| 날짜 | 작업 |
|------|------|
| 4/25 (D5 완료 후) | Isaac Lab play 확인 (±10N 외력 테스트) |
| 4/25~4/26 | Gazebo bridge 구성 + 보행 확인 |
| 4/26 | 머리 탈 USD로 play 재실행 (외형 확인) |
| 4/26~4/27 | NUC 이전 및 최종 패키징 |

---

## 5. NUC 배포 계획

BHL 공식 아키텍처 기준 (https://berkeley-humanoid-lite.gitbook.io/docs):

```
[Orin (상위)] → /gait/cmd (ROS2 Twist, 25Hz)
                      ↓
[NUC (하위)] — policy inference (50Hz)
             — ESC × 10 (CAN 1Mbps)
                      ↓
             [모터 × 10]
```

### NUC 이전 체크리스트
```
□ 1. 최종 best.pt 확인
      ls δ3/checkpoints/stage_e4_hylion_v6/best.pt  (또는 도달 가능한 최고 스테이지)

□ 2. 패키징
      tar czf hylion_v6_final_$(date +%F).tar.gz \
        δ3/checkpoints/ \
        δ3/hylion/ \
        δ3/scripts/ \
        δ3/usd/hylion_v6/

□ 3. NUC로 전송
      scp hylion_v6_final_*.tar.gz [NUC_USER]@[NUC_IP]:~/

□ 4. NUC 환경 확인
      isaaclab 동일 버전 설치 여부 확인
      python δ3/scripts/play_hylion_v6_BG.py --ckpt_path ... --num_envs 1

□ 5. 안전 게이트 설정
      최대 속도: lin_vel_x ≤ 0.3 m/s (데모 시)
      관절 토크 한계: stiffness 20.0, damping 2.0 유지
      넘어짐 감지 → 즉시 정지 (orientation > 45°)
```

### sim-to-real 보정 여유
- 학습: ±30N → 실제 로봇: ~15~20N 대응 가능 (sim-to-real 손실 포함)
- domain randomization: mass ±1.5kg, 외력 ±30N, 관절노이즈 ±0.05rad

---

## 6. 전체 타임라인 요약

```
4/22 (오늘)  D3~D4 자동 완료 (orchestrator 실행 중)
4/23         D4.5~D5 자동 완료
4/23~4/24    E1~E4 학습 (별도 스크립트 필요)
4/25         D5/E4 Isaac Lab play 검증
4/25~4/26    Gazebo sim-to-sim 검증 + 머리 탈 USD 확인
4/26~4/27    NUC 이전 + 최종 패키징
2026-06-01   발표 🎯
```

---

*작성: 2026-04-22 | 상태: 운영 중*
