"""Sim-to-sim: Hylion v6 IsaacLab policy → MuJoCo 검증 (MJCF 기반).

MJCF(sim/isaaclab/robot/hylion_v6.xml)를 직접 로드하므로
URDF 패칭·freejoint 추가·IMU 수동 변환이 불필요하다.

물리 설정 (robot_cfg_BG.py 기준):
  kp=20, kd=2, effort_limit=6 Nm
  armature: hip/knee=0.007, ankle=0.002  ← MJCF에 포함
  sim_dt=1/200 Hz, decimation=8  →  control Hz=25

obs 45-dim:
  [0:3]   velocity_commands  (vx, vy, wz)
  [3:6]   base_ang_vel       (imu_gyro sensor → body frame)
  [6:9]   projected_gravity  (imu_quat sensor → body frame 중력벡터)
  [9:21]  joint_pos_rel      (12 leg joints, default offset 제거)
  [21:33] joint_vel          (12 leg joints)
  [33:45] last_action        (12)

action 12-dim:
  target_pos = default_pos + action * 0.25
  torque = kp*(target_pos - q) - kd*qdot  (clipped ±effort_limit)

주요 CLI 플래그:
  --mjcf PATH        MJCF 경로 (default: sim/isaaclab/robot/hylion_v6.xml)
  --effort-limit N   토크 한도 (default 6)
  --kp / --kd        PD 게인 (default 20 / 2)
  --zero-action      policy 무시, PD가 default 자세만 유지 (기준선 측정)
  --diag N           N 스텝마다 obs/action/torque 진단 출력
  --no-viewer        headless 실행 (SSH 환경)
"""

import argparse
import os
import sys
import numpy as np

REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_CKPT = os.path.join(REPO_ROOT, "checkpoints/biped/stage_d5_hylion_v6/best.pt")
DEFAULT_MJCF = os.path.join(REPO_ROOT, "sim/isaaclab/robot/hylion_v6.xml")

# IsaacLab env_cfg.py 기준 다리 joint 순서 (action 12-dim 대응)
LEG_JOINTS = [
    "leg_left_hip_roll_joint",
    "leg_left_hip_yaw_joint",
    "leg_left_hip_pitch_joint",
    "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint",
    "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint",
    "leg_right_hip_yaw_joint",
    "leg_right_hip_pitch_joint",
    "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint",
    "leg_right_ankle_roll_joint",
]

# robot_cfg_BG.py InitialStateCfg 기준 default joint positions
DEFAULT_JOINT_POS = np.array([
    0.0, 0.0, -0.2, 0.4, -0.3, 0.0,   # left:  roll yaw pitch knee ankle_p ankle_r
    0.0, 0.0, -0.2, 0.4, -0.3, 0.0,   # right
], dtype=np.float32)

DEFAULT_KP           = 20.0
DEFAULT_KD           = 2.0
DEFAULT_EFFORT_LIMIT = 6.0
ACTION_SCALE         = 0.25
CONTROL_HZ           = 25
SIM_DT               = 1.0 / 200.0
N_SUBSTEPS           = 8


def load_policy(ckpt_path, obs_dim=45, act_dim=12, device="cpu"):
    """RSL-RL checkpoint에서 actor MLP 추출."""
    import torch
    import torch.nn as nn

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    class ActorMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ELU(),
                nn.Linear(256, 128),    nn.ELU(),
                nn.Linear(128, 128),    nn.ELU(),
                nn.Linear(128, act_dim),
            )
        def forward(self, x):
            return self.net(x)

    actor = ActorMLP().to(device)
    if "actor_state_dict" in ckpt:
        raw = ckpt["actor_state_dict"]
        state = {"net." + k[len("mlp."):]: v for k, v in raw.items() if k.startswith("mlp.")}
    elif "model_state_dict" in ckpt:
        raw = ckpt["model_state_dict"]
        state = {"net." + k[len("actor."):]: v for k, v in raw.items() if k.startswith("actor.")}
    else:
        state = {k: v for k, v in ckpt.items() if not k.startswith("critic")}
    missing, _ = actor.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing actor keys: {missing}")
    return actor


def projected_gravity_vec(quat_wxyz):
    """quaternion (w,x,y,z) → body frame 중력 단위벡터."""
    w, x, y, z = quat_wxyz
    return np.array([
        -2.0 * (x*z - w*y),
        -2.0 * (y*z + w*x),
        -(1.0 - 2.0*(x*x + y*y)),
    ], dtype=np.float32)


def run(args):
    import mujoco
    import mujoco.viewer
    import torch

    kp, kd, effort_limit = args.kp, args.kd, args.effort_limit

    # ── 모델 로드 ──
    print(f"[SIM2SIM] Loading MJCF: {args.mjcf}")
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    # RK4는 contact 시스템에 불안정 → implicitfast(MuJoCo 3.x default)로 교체
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # 학습 로봇 총 질량 보정: MJCF는 SO-ARM 팔 body 없이 13kg, 학습은 19.89kg
    # 누락 질량을 base body에 lumped mass로 추가
    TRAINING_TOTAL_MASS = 19.89
    mass_deficit = TRAINING_TOTAL_MASS - model.body_mass.sum()
    if abs(mass_deficit) > 0.1:
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        model.body_mass[base_id] += mass_deficit
        print(f"[SIM2SIM] Mass correction: +{mass_deficit:.2f} kg to base "
              f"(total {model.body_mass.sum():.2f} kg)")

    if effort_limit != DEFAULT_EFFORT_LIMIT:
        for i in range(model.nu):
            model.actuator_ctrlrange[i] = [-effort_limit, effort_limit]
    data = mujoco.MjData(model)
    print(f"[SIM2SIM] nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # ── 센서 주소 (MJCF에 정의된 imu_gyro / imu_quat) ──
    def _sensor_adr(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return model.sensor_adr[sid]

    gyro_adr = _sensor_adr("imu_gyro")   # body-frame angular velocity (dim=3)
    quat_adr = _sensor_adr("imu_quat")   # orientation quaternion (w,x,y,z) (dim=4)

    # ── joint / actuator 인덱스 ──
    qpos_ids = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in LEG_JOINTS]
    qvel_ids = [model.jnt_dofadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in LEG_JOINTS]
    act_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in LEG_JOINTS]
    hip_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_left_hip_roll")

    # ── 초기 자세: 발 높이 자동 보정 ──
    mujoco.mj_resetData(model, data)
    for i, qi in enumerate(qpos_ids):
        data.qpos[qi] = DEFAULT_JOINT_POS[i]
    data.qpos[2] = 2.0   # FK 계산용 임시 높이
    data.qpos[3] = 1.0   # quaternion w
    mujoco.mj_forward(model, data)

    active_col_ids = [i for i in range(model.ngeom)
                      if model.geom_contype[i] > 0
                      and model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE]
    foot_geom_id  = min(active_col_ids, key=lambda i: data.geom_xpos[i, 2])
    foot_center_z = data.geom_xpos[foot_geom_id, 2]
    gtype = model.geom_type[foot_geom_id]
    size  = model.geom_size[foot_geom_id]
    # world-frame z half-extent: |R[2,:]| @ local_half  (AABB 공식)
    # geom이 회전돼있으면 local size[2]가 world z extent가 아님
    local_half = np.array([size[0], size[0], size[1]]) \
                 if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER else size[:3].copy()
    R = data.geom_xmat[foot_geom_id].reshape(3, 3)
    world_half_z = float(np.abs(R[2, :]) @ local_half)
    data.qpos[2] = 2.0 - (foot_center_z - world_half_z) + 0.005
    mujoco.mj_forward(model, data)
    print(f"[SIM2SIM] base_z={data.qpos[2]:.3f}m, hip_z={data.xpos[hip_bid,2]:.3f}m")

    # ── policy 로드 ──
    if args.zero_action:
        policy = None
        print("[SIM2SIM] Mode: ZERO-ACTION (pure PD at default pose)")
    else:
        print(f"[SIM2SIM] Loading policy: {args.ckpt}")
        policy = load_policy(args.ckpt, device=args.device)
        policy.eval()

    cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    print(f"[SIM2SIM] Command: vx={args.vx} vy={args.vy} wz={args.wz}")
    print(f"[SIM2SIM] KP={kp}, KD={kd}, effort_limit={effort_limit} Nm")
    print(f"[SIM2SIM] Control Hz={CONTROL_HZ}, substeps={N_SUBSTEPS}")

    last_action = np.zeros(12, dtype=np.float32)
    total_steps = int(args.duration * CONTROL_HZ)
    survived    = 0

    def get_obs():
        # freejoint qvel[3:6] = world-frame angular velocity
        # R.T @ omega_world = body-frame angular velocity (IsaacLab 기준과 동일)
        quat_wxyz = data.qpos[3:7].astype(np.float32)
        w, x, y, z = quat_wxyz
        R = np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ], dtype=np.float32)
        o_angvel  = R.T @ data.qvel[3:6].astype(np.float32)
        o_grav    = projected_gravity_vec(quat_wxyz)
        o_qpos    = np.array([data.qpos[qi] for qi in qpos_ids], np.float32) - DEFAULT_JOINT_POS
        o_qvel    = np.array([data.qvel[vi] for vi in qvel_ids], np.float32)
        return np.concatenate([cmd, o_angvel, o_grav, o_qpos, o_qvel, last_action]).astype(np.float32)

    print(f"[SIM2SIM] Running {total_steps} steps ({args.duration}s) ...")

    def _run_loop(viewer=None):
        nonlocal survived
        for step in range(total_steps):
            obs_np = get_obs()

            if policy is not None:
                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=args.device).unsqueeze(0)
                with torch.inference_mode():
                    action_np = policy(obs_t).squeeze(0).cpu().numpy()
                action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                action_np = np.zeros(12, dtype=np.float32)
            last_action[:] = action_np

            target_pos = DEFAULT_JOINT_POS + action_np * ACTION_SCALE
            torques = np.zeros(12, dtype=np.float32)
            for _ in range(N_SUBSTEPS):
                for i, aid in enumerate(act_ids):
                    q  = data.qpos[qpos_ids[i]]
                    qd = data.qvel[qvel_ids[i]]
                    t  = float(np.clip(kp * (target_pos[i] - q) - kd * qd,
                                       -effort_limit, effort_limit))
                    data.ctrl[aid] = t
                    torques[i]     = t
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

            hip_z = data.xpos[hip_bid, 2]

            if args.diag > 0 and step % args.diag == 0:
                quat_wxyz = data.sensordata[quat_adr:quat_adr+4].astype(np.float32)
                g = projected_gravity_vec(quat_wxyz)
                pitch = float(np.degrees(np.arcsin(np.clip(-g[0], -1, 1))))
                jn = ["Lhr","Lhy","Lhp","Lk","Lap","Lar",
                      "Rhr","Rhy","Rhp","Rk","Rap","Rar"]
                print(f"\n--- step {step:4d}  pitch={pitch:+6.1f}°  hip_z={hip_z:.3f}m ---")
                print(f"  grav_body : {obs_np[6:9].round(3)}")
                print(f"  ang_vel   : {obs_np[3:6].round(3)}")
                print(f"  qpos_rel  : {dict(zip(jn, obs_np[9:21].round(3)))}")
                print(f"  action    : {dict(zip(jn, action_np.round(3)))}")
                print(f"  torque    : {dict(zip(jn, torques.round(2)))}")
                sat = np.abs(torques) >= effort_limit - 0.01
                if sat.any():
                    print(f"  SATURATED : {[jn[i] for i in range(12) if sat[i]]}")

            if hip_z < 0.15:
                print(f"[SIM2SIM] FALL at step {step} (hip_z={hip_z:.3f})")
                return
            survived = step + 1

    if args.no_viewer:
        _run_loop()
    else:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                _run_loop(viewer)
        except Exception as e:
            print(f"[SIM2SIM] Viewer error: {e}")

    print(f"[SIM2SIM] Done. Survived {survived}/{total_steps} steps "
          f"({survived/CONTROL_HZ:.1f}s / {args.duration:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hylion v6 sim-to-sim (IsaacLab → MuJoCo, MJCF)")
    parser.add_argument("--ckpt",         type=str,   default=DEFAULT_CKPT,
                        help="RSL-RL checkpoint 경로")
    parser.add_argument("--mjcf",         type=str,   default=DEFAULT_MJCF,
                        help="MJCF .xml 경로")
    parser.add_argument("--vx",           type=float, default=0.0)
    parser.add_argument("--vy",           type=float, default=0.0)
    parser.add_argument("--wz",           type=float, default=0.0)
    parser.add_argument("--duration",     type=float, default=10.0,
                        help="시뮬레이션 시간 (초)")
    parser.add_argument("--kp",           type=float, default=DEFAULT_KP)
    parser.add_argument("--kd",           type=float, default=DEFAULT_KD)
    parser.add_argument("--effort-limit", type=float, default=DEFAULT_EFFORT_LIMIT,
                        dest="effort_limit", help="토크 한도 Nm (default 6)")
    parser.add_argument("--zero-action",  action="store_true", dest="zero_action",
                        help="policy 무시, pure PD (기준선 측정)")
    parser.add_argument("--diag",         type=int,   default=0, metavar="N",
                        help="N 스텝마다 obs/action/torque 출력 (0=off)")
    parser.add_argument("--device",       type=str,   default="cpu")
    parser.add_argument("--no-viewer",    action="store_true", dest="no_viewer",
                        help="headless 실행 (SSH 환경)")
    args = parser.parse_args()

    if not args.zero_action and not os.path.isfile(args.ckpt):
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        sys.exit(1)
    if not os.path.isfile(args.mjcf):
        print(f"[ERROR] MJCF not found: {args.mjcf}")
        sys.exit(1)

    run(args)
