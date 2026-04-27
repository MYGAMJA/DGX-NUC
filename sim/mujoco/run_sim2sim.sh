#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_sim2sim.sh  —  Hylion v6 sim-to-sim (IsaacLab → MuJoCo) 실행 스크립트
#
# 사용법:
#   chmod +x run_sim2sim.sh
#   ./run_sim2sim.sh [mode]
#
# mode 목록:
#   baseline     policy 없이 PD만으로 기본자세 유지 (로봇이 서 있을 수 있는지 확인)
#   diag         진단 출력 포함, 걷기 명령 vx=0.3 (obs/action/torque 매 25스텝 출력)
#   walk         걷기 명령 vx=0.3, GUI 뷰어 포함
#   walk_noise   obs 노이즈 + 학습 환경 마찰 중간값 (가장 현실적인 sim-to-sim)
#   walk_hard    effort_limit=20으로 올려서 걷기 (토크 부족 가설 검증)
#   walk_arm     armature 적용 + 걷기 (물리 매칭 강화)
#   walk_full    obs_noise + armature + friction=0.8 (학습 환경 최대 모방)
#   headless     GUI 없이 headless 실행 (SSH 환경)
#   custom       아래 CUSTOM_* 변수 직접 수정 후 사용
# ──────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
CKPT="$REPO_ROOT/checkpoints/biped/stage_d5_hylion_v6/best.pt"
URDF="$REPO_ROOT/sim/isaaclab/robot/hylion_v6.urdf"
MJCF="$REPO_ROOT/sim/isaaclab/robot/hylion_v6.xml"
PY="$SCRIPT_DIR/play_mujoco.py"

# ── obs 45-dim 구성 (IsaacLab env_cfg.py 기준) ────────────────────────────────
#  [0:3]   velocity_commands : --vx  --vy  --wz  (아래에서 설정)
#  [3:6]   base_ang_vel      : IMU gyro → MuJoCo qvel[3:6] (body frame 변환)
#  [6:9]   projected_gravity : quaternion → R^T @ [0,0,-1]
#  [9:21]  joint_pos_rel     : qpos - DEFAULT_JOINT_POS  (12 leg joints)
#  [21:33] joint_vel         : qvel  (12 leg joints)
#  [33:45] last_action       : 직전 policy 출력 (자동 관리)
#
# ─ 실제로 사용자가 제어할 수 있는 입력 변수 ─────────────────────────────────────
VX=0.3       # linear velocity x (m/s) — 앞으로 걷기: 0.3~0.5
VY=0.0       # linear velocity y (m/s) — 옆으로 걷기
WZ=0.0       # angular velocity z (rad/s) — 회전: 0.3~0.5

KP=20.0          # PD stiffness (IsaacLab robot_cfg_BG.py 기준)
KD=2.0           # PD damping
EFFORT=6.0       # torque limit (Nm) — 너무 작으면 motor saturation
DURATION=10.0    # 시뮬레이션 시간 (s)
DIAG=0           # 진단 출력 주기 (0=꺼짐, 25=25스텝마다)

# CUSTOM 모드용 변수
CUSTOM_VX=0.3
CUSTOM_VY=0.0
CUSTOM_WZ=0.0
CUSTOM_EFFORT=6.0
CUSTOM_DURATION=10.0

MODE="${1:-walk}"

echo "========================================="
echo " Hylion v6 sim-to-sim  mode: $MODE"
echo "========================================="

case "$MODE" in

  baseline)
    echo "[MODE] ZERO-ACTION: policy 비활성, PD로 기본자세만 유지"
    echo "[INFO] 로봇이 쓰러지면 URDF/충돌/물리 설정 문제"
    python3 "$PY" \
      --urdf "$URDF" \
      --zero-action \
      --kp $KP --kd $KD --effort-limit $EFFORT \
      --duration $DURATION \
      --diag 25
    ;;

  diag)
    echo "[MODE] DIAG: 걷기 vx=$VX, 진단 출력 25스텝마다"
    python3 "$PY" \
      --ckpt "$CKPT" --urdf "$URDF" \
      --vx $VX --vy $VY --wz $WZ \
      --kp $KP --kd $KD --effort-limit $EFFORT \
      --duration $DURATION \
      --diag 25
    ;;

  walk)
    echo "[MODE] WALK: vx=$VX, GUI 뷰어 포함 (MJCF 사용)"
    python3 "$PY" \
      --ckpt "$CKPT" --mjcf "$MJCF" \
      --vx $VX --vy $VY --wz $WZ \
      --kp $KP --kd $KD --effort-limit $EFFORT \
      --duration $DURATION
    ;;

  walk_noise)
    echo "[MODE] WALK_NOISE: obs 노이즈 + 마찰 0.8 (학습 환경 모방)"
    python3 "$PY" \
      --ckpt "$CKPT" --mjcf "$MJCF" \
      --vx $VX --vy $VY --wz $WZ \
      --kp $KP --kd $KD --effort-limit $EFFORT \
      --obs-noise --friction 0.8 \
      --duration $DURATION \
      --diag 25
    ;;

  walk_hard)
    echo "[MODE] WALK_HARD: effort_limit=20 (토크 부족 가설 검증)"
    python3 "$PY" \
      --ckpt "$CKPT" --urdf "$URDF" \
      --vx $VX --vy $VY --wz $WZ \
      --kp $KP --kd $KD --effort-limit 20.0 \
      --duration $DURATION \
      --diag 25
    ;;

  walk_arm)
    echo "[MODE] WALK_ARM: armature 적용 (hip/knee=0.007, ankle=0.002)"
    python3 "$PY" \
      --ckpt "$CKPT" --urdf "$URDF" \
      --vx $VX --vy $VY --wz $WZ \
      --kp $KP --kd $KD --effort-limit $EFFORT \
      --armature \
      --duration $DURATION \
      --diag 25
    ;;
  walk_full)
    echo "[MODE] WALK_FULL: obs_noise + armature + friction=0.8 (학습 환경 최대 모방)"
    python3 "$PY" \
      --ckpt "$CKPT" --mjcf "$MJCF" \
      --vx $VX --vy $VY --wz $WZ \
      --kp $KP --kd $KD --effort-limit $EFFORT \
      --obs-noise --armature --friction 0.8 \
      --duration $DURATION \
      --diag 25
    ;;
  headless)
    echo "[MODE] HEADLESS: GUI 없음 (SSH 환경)"
    python3 "$PY" \
      --ckpt "$CKPT" --urdf "$URDF" \
      --vx $VX --vy $VY --wz $WZ \
      --kp $KP --kd $KD --effort-limit $EFFORT \
      --duration $DURATION \
      --no-viewer \
      --diag 25
    ;;

  custom)
    echo "[MODE] CUSTOM: VX=$CUSTOM_VX, effort=$CUSTOM_EFFORT"
    python3 "$PY" \
      --ckpt "$CKPT" --urdf "$URDF" \
      --vx $CUSTOM_VX --vy $CUSTOM_VY --wz $CUSTOM_WZ \
      --kp $KP --kd $KD --effort-limit $CUSTOM_EFFORT \
      --duration $CUSTOM_DURATION \
      --diag 25
    ;;

  *)
    echo "[ERROR] 알 수 없는 mode: $MODE"
    echo "사용 가능한 mode: baseline | diag | walk | walk_noise | walk_hard | walk_arm | walk_full | headless | custom"
    exit 1
    ;;
esac
