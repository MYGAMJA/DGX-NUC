"""Sim-to-sim: Hylion v6 IsaacLab policy → MuJoCo 검증.

학습 환경 (IsaacLab/PhysX) 에서 훈련된 RSL-RL policy를 MuJoCo에서 실행.

물리 설정 (robot_cfg_BG.py 기준):
  kp=20, kd=2, effort_limit=6 Nm  (legs & ankles 동일)
  armature: hip/knee=0.007, ankle=0.002  (--armature 플래그로 활성화)

obs 구성 (45-dim) — env_cfg.py ObservationsCfg 기준:
  [0:3]   velocity_commands  (vx, vy, wz)
  [3:6]   base_ang_vel       (body frame)  noise: Uniform ±0.3
  [6:9]   projected_gravity  (body frame)  noise: Uniform ±0.05
  [9:21]  joint_pos_rel      (12 leg joints, default offset 제거)  noise: Uniform ±0.05
  [21:33] joint_vel          (12 leg joints)  noise: Uniform ±2.0
  [33:45] last_action        (12)  noise: 없음

action (12-dim):
  target_pos = default_pos + action * 0.25
  torque = kp*(target_pos - q) - kd*qdot  (clipped to ±effort_limit)

Physics:
  sim_dt=1/200 Hz, decimation=8  →  control Hz=25
  바닥 마찰: static/dynamic 0.4~1.2 (env_cfg.py EventsCfg randomize_rigid_body_material)

Termination (env_cfg.py TerminationsCfg):
  base 기울기 > 0.78 rad (45°)  ← 학습 환경 기준

주요 CLI 플래그:
  --effort-limit N   토크 한도 (default 6, 테스트: 20)
  --kp / --kd        PD 게인 (default 20 / 2)
  --armature         훈련 config의 armature 적용 (hip/knee=0.007, ankle=0.002)
  --obs-noise        obs 노이즈 주입 (학습 환경과 동일한 UniformNoise)
  --friction F       바닥 마찰 고정값 (default 0.8; 학습 범위: 0.4~1.2)
  --zero-action      policy 무시, PD가 default 자세만 유지 (기준선 측정)
  --diag N           N 스텝마다 obs/action/torque 진단 출력
"""

import argparse
import os
import sys
import re
import numpy as np

REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_CKPT = os.path.join(REPO_ROOT, "checkpoints/biped/stage_d5_hylion_v6/best.pt")
DEFAULT_URDF = os.path.join(REPO_ROOT, "sim/isaaclab/robot/hylion_v6.urdf")
DEFAULT_MJCF = os.path.join(REPO_ROOT, "sim/isaaclab/robot/hylion_v6.xml")

# IsaacLab env_cfg.py 기준 다리 joint 순서
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
    0.0,   # leg_left_hip_roll_joint
    0.0,   # leg_left_hip_yaw_joint
   -0.2,   # leg_left_hip_pitch_joint
    0.4,   # leg_left_knee_pitch_joint
   -0.3,   # leg_left_ankle_pitch_joint
    0.0,   # leg_left_ankle_roll_joint
    0.0,   # leg_right_hip_roll_joint
    0.0,   # leg_right_hip_yaw_joint
   -0.2,   # leg_right_hip_pitch_joint
    0.4,   # leg_right_knee_pitch_joint
   -0.3,   # leg_right_ankle_pitch_joint
    0.0,   # leg_right_ankle_roll_joint
], dtype=np.float32)

# robot_cfg_BG.py actuator config 기준 (CLI args로 오버라이드 가능)
DEFAULT_KP           = 20.0
DEFAULT_KD           = 2.0
DEFAULT_EFFORT_LIMIT = 6.0

# robot_cfg_BG.py armature (--armature 플래그로 적용)
ARMATURE_HIP_KNEE = 0.007
ARMATURE_ANKLE    = 0.002

ACTION_SCALE = 0.25   # JointPositionActionCfg scale
CONTROL_HZ   = 25
SIM_DT       = 1.0 / 200.0
N_SUBSTEPS   = 8      # decimation


def build_mjcf_model(mjcf_path: str, effort_limit: float = DEFAULT_EFFORT_LIMIT):
    """Hylion v6 MJCF (.xml) → MuJoCo model 직접 로드.
    URDF 패칭 없이 네이티브 MuJoCo XML 로드.
    """
    try:
        import mujoco
    except ImportError:
        raise ImportError("pip install mujoco")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    # effort_limit CLI 인자로 ctrlrange 오버라이드
    for i in range(model.nu):
        model.actuator_ctrlrange[i, 0] = -effort_limit
        model.actuator_ctrlrange[i, 1] =  effort_limit
    data = mujoco.MjData(model)
    return model, data


def build_mujoco_model(urdf_path: str, effort_limit: float = DEFAULT_EFFORT_LIMIT):
    """Hylion v6 URDF → MuJoCo model 빌드.

    URDF에는 이미 box/cylinder collision 지오메트리가 정의되어 있음.
    mesh visual geom만 비활성화하고 URDF 기존 collision geom을 그대로 사용.
    """
    try:
        import mujoco
    except ImportError:
        raise ImportError("pip install mujoco")

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    with open(urdf_path, "r") as f:
        urdf_xml = f.read()

    # STL assets → basename 키로 로드 (MuJoCo URDF 로더의 경로 해석 방식)
    stl_refs = re.findall(r'filename="([^"]+\.[Ss][Tt][Ll])"', urdf_xml)
    assets = {}
    for rel in stl_refs:
        abs_path = os.path.normpath(os.path.join(urdf_dir, rel))
        key = os.path.basename(rel)
        if key not in assets and os.path.isfile(abs_path):
            with open(abs_path, "rb") as f:
                assets[key] = f.read()
    urdf_patched = re.sub(
        r'filename="([^"]+\.[Ss][Tt][Ll])"',
        lambda m: f'filename="{os.path.basename(m.group(1))}"',
        urdf_xml,
    )

    spec = mujoco.MjSpec.from_string(urdf_patched, assets=assets)

    # ── 1. ground plane 추가 ──
    gnd = spec.worldbody.add_geom()
    gnd.name = "ground"
    gnd.type = mujoco.mjtGeom.mjGEOM_PLANE
    gnd.size = np.array([0.0, 0.0, 0.05])
    gnd.rgba = np.array([0.4, 0.4, 0.4, 1.0])

    # ── 2. base body에 freejoint 추가 ──
    base_body = spec.worldbody.bodies[0]  # "base"
    fj = base_body.add_freejoint()
    fj.name = "root"

    # ── 3. mesh visual geom collision 비활성화 ──
    # URDF에 이미 box/cylinder collision geom이 정의되어 있으므로
    # mesh(visual) geom만 비활성화하면 됨
    def _disable_mesh_geoms(body):
        for geom in body.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                geom.contype = 0
                geom.conaffinity = 0
        for child in body.bodies:
            _disable_mesh_geoms(child)

    _disable_mesh_geoms(base_body)

    # ── 4. 다리 12관절에 motor actuator 추가 ──
    # motor 타입: data.ctrl = torque → PD는 run()에서 수동 계산
    for jname in LEG_JOINTS:
        act = spec.add_actuator()
        act.name    = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target  = jname
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange   = np.array([-effort_limit, effort_limit])

    model = spec.compile()

    # IsaacLab: enabled_self_collisions=False → 로봇 geom 간 self-collision 비활성화
    # 로봇 geom: contype=1, conaffinity=0  (지면만 conaffinity=1이어서 지면과만 충돌)
    gnd_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    for i in range(model.ngeom):
        if i == gnd_id:
            model.geom_contype[i]     = 0
            model.geom_conaffinity[i] = 1
        else:
            model.geom_contype[i]     = 1
            model.geom_conaffinity[i] = 0

    # PhysX articulation solver 설정 대응 (solver_position_iteration_count=8)
    model.opt.iterations     = 50   # contact solver iterations (PhysX 8 pos + 4 vel)
    model.opt.noslip_iterations = 4
    model.opt.ls_iterations  = 50

    data = mujoco.MjData(model)
    return model, data


def get_joint_indices(model, joint_names):
    import mujoco
    ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found")
        ids.append(model.jnt_qposadr[jid])
    return ids


def get_joint_vel_indices(model, joint_names):
    import mujoco
    ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found")
        ids.append(model.jnt_dofadr[jid])
    return ids


def load_policy(ckpt_path: str, obs_dim: int = 45, act_dim: int = 12, device: str = "cpu"):
    """RSL-RL checkpoint에서 actor network 추출.
    체크포인트 구조: {'actor_state_dict': {'mlp.X.*': ...}}
    """
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

    model = ActorMLP().to(device)

    if "actor_state_dict" in ckpt:
        raw = ckpt["actor_state_dict"]
        # mlp.X.* → net.X.*  (distribution.* 제외)
        state = {"net." + k[len("mlp."):]: v
                 for k, v in raw.items() if k.startswith("mlp.")}
    elif "model_state_dict" in ckpt:
        raw = ckpt["model_state_dict"]
        state = {"net." + k[len("actor."):]: v
                 for k, v in raw.items() if k.startswith("actor.")}
    elif "actor_critic" in ckpt:
        raw = ckpt["actor_critic"]
        state = {"net." + k[len("actor."):]: v
                 for k, v in raw.items() if k.startswith("actor.")}
    else:
        state = {k: v for k, v in ckpt.items() if not k.startswith("critic")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing actor keys: {missing}")
    return model


def projected_gravity_vec(quat_wxyz: np.ndarray) -> np.ndarray:
    """quaternion (w,x,y,z) → body frame 중력 단위벡터."""
    w, x, y, z = quat_wxyz
    gx = -2.0 * (x*z - w*y)
    gy = -2.0 * (y*z + w*x)
    gz = -(1.0 - 2.0*(x*x + y*y))
    return np.array([gx, gy, gz], dtype=np.float32)


def run(args):
    import mujoco
    import mujoco.viewer
    import torch

    kp           = args.kp
    kd           = args.kd
    effort_limit = args.effort_limit

    # ── 모델 로드 (MJCF 우선, 없으면 URDF) ──
    if args.mjcf and os.path.isfile(args.mjcf):
        print(f"[SIM2SIM] Loading Hylion v6 MJCF: {args.mjcf}")
        model, data = build_mjcf_model(args.mjcf, effort_limit=effort_limit)
    else:
        print(f"[SIM2SIM] Loading Hylion v6 URDF: {args.urdf}")
        model, data = build_mujoco_model(args.urdf, effort_limit=effort_limit)
    model.opt.timestep = SIM_DT
    print(f"[SIM2SIM] nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # ── armature 적용 (--armature) ──
    if args.armature:
        for jname in LEG_JOINTS:
            jid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            dofadr = model.jnt_dofadr[jid]
            model.dof_armature[dofadr] = (
                ARMATURE_ANKLE if "ankle" in jname else ARMATURE_HIP_KNEE
            )
        print(f"[SIM2SIM] Armature: hip/knee={ARMATURE_HIP_KNEE}, ankle={ARMATURE_ANKLE}")

    # 활성 collision geom 목록 (mesh/plane 제외, 로봇 geom은 contype=1)
    active_col_ids = [
        i for i in range(model.ngeom)
        if (model.geom_contype[i] > 0
            and model.geom_type[i] not in (
                mujoco.mjtGeom.mjGEOM_PLANE,
                mujoco.mjtGeom.mjGEOM_MESH,
            ))
    ]
    print(f"[SIM2SIM] Active collision geoms: {len(active_col_ids)}")

    qpos_ids = get_joint_indices(model, LEG_JOINTS)
    qvel_ids = get_joint_vel_indices(model, LEG_JOINTS)

    # actuator index
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
               for n in LEG_JOINTS]

    # fall detection: hip body
    hip_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_left_hip_roll")

    # ── 초기 자세: URDF collision geom 기준 foot 높이 자동 계산 ──
    mujoco.mj_resetData(model, data)
    for i, qi in enumerate(qpos_ids):
        data.qpos[qi] = DEFAULT_JOINT_POS[i]
    data.qpos[2] = 2.0
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    foot_geom_id  = min(active_col_ids, key=lambda i: data.geom_xpos[i, 2])
    foot_center_z = data.geom_xpos[foot_geom_id, 2]
    gtype         = model.geom_type[foot_geom_id]
    foot_half_z   = float(model.geom_size[foot_geom_id, 1 if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER else 2])

    data.qpos[2] = 2.0 - foot_center_z + foot_half_z + 0.005
    mujoco.mj_forward(model, data)
    print(f"[SIM2SIM] Auto base_z={data.qpos[2]:.3f}m, initial hip_z={data.xpos[hip_bid,2]:.3f}m")

    # ── policy 로드 ──
    device = args.device
    if args.zero_action:
        policy = None
        print("[SIM2SIM] Mode: ZERO-ACTION (policy disabled, pure PD at default pose)")
    else:
        print(f"[SIM2SIM] Loading policy: {args.ckpt}")
        policy = load_policy(args.ckpt, device=device)
        policy.eval()

    cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    print(f"[SIM2SIM] Command: vx={args.vx} vy={args.vy} wz={args.wz}")
    print(f"[SIM2SIM] KP={kp}, KD={kd}, effort_limit={effort_limit} Nm")
    print(f"[SIM2SIM] Control Hz={CONTROL_HZ}, substeps={N_SUBSTEPS}")
    print(f"[SIM2SIM] obs_noise={'ON' if args.obs_noise else 'OFF'}, friction={args.friction}")
    if args.diag > 0:
        print(f"[SIM2SIM] Diagnostics every {args.diag} steps")

    # ── 바닥 마찰 설정 (env_cfg.py: static/dynamic 0.4~1.2) ──────────────
    # MJCF/URDF 로드 후 ground geom의 friction을 CLI 값으로 설정
    import mujoco as _mj
    gnd_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_GEOM, "ground")
    if gnd_id >= 0:
        model.geom_friction[gnd_id, 0] = args.friction   # sliding
        model.geom_friction[gnd_id, 1] = args.friction * 0.1  # torsional
        model.geom_friction[gnd_id, 2] = args.friction * 0.1  # rolling
        print(f"[SIM2SIM] Ground friction set to {args.friction:.2f}")

    last_action = np.zeros(12, dtype=np.float32)
    total_steps = int(args.duration * CONTROL_HZ)

    # ── obs 노이즈 (env_cfg.py ObservationsCfg UniformNoiseCfg 기준) ──────
    # 학습 시 policy는 이 노이즈가 있는 obs를 받았으므로 sim-to-sim도 동일하게
    rng = np.random.default_rng(seed=42)
    def _uniform(lo, hi, size):
        return rng.uniform(lo, hi, size).astype(np.float32) if args.obs_noise else np.zeros(size, np.float32)

    # 진단용 roll/pitch 계산 (body x/y 축의 중력 투영)
    def _pitch_deg(quat_wxyz):
        g = projected_gravity_vec(quat_wxyz)
        return float(np.degrees(np.arcsin(np.clip(-g[0], -1, 1))))

    def get_obs():
        quat_wxyz = data.qpos[3:7].astype(np.float32)
        w, x, y, z = quat_wxyz
        R = np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ], dtype=np.float32)
        o_angvel = R.T @ data.qvel[3:6].astype(np.float32) + _uniform(-0.3,  0.3,  3)  # env_cfg noise
        o_grav   = projected_gravity_vec(quat_wxyz)         + _uniform(-0.05, 0.05, 3)  # env_cfg noise
        o_qpos   = np.array([data.qpos[qi] for qi in qpos_ids], dtype=np.float32) - DEFAULT_JOINT_POS \
                   + _uniform(-0.05, 0.05, 12)  # env_cfg noise
        o_qvel   = np.array([data.qvel[vi] for vi in qvel_ids], dtype=np.float32) \
                   + _uniform(-2.0,  2.0,  12)  # env_cfg noise
        return np.concatenate([cmd, o_angvel, o_grav, o_qpos, o_qvel, last_action]).astype(np.float32)

    print(f"[SIM2SIM] Running {total_steps} steps ({args.duration}s) ...")
    survived = 0

    def _run_loop(viewer=None):
        nonlocal survived
        for step in range(total_steps):
            obs_np = get_obs()

            if policy is not None:
                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.inference_mode():
                    action_np = policy(obs_t).squeeze(0).cpu().numpy()
                action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                action_np = np.zeros(12, dtype=np.float32)
            last_action[:] = action_np

            target_pos = DEFAULT_JOINT_POS + action_np * ACTION_SCALE
            torques    = np.zeros(12, dtype=np.float32)
            for _ in range(N_SUBSTEPS):
                for i, aid in enumerate(act_ids):
                    if aid >= 0:
                        q  = data.qpos[qpos_ids[i]]
                        qd = data.qvel[qvel_ids[i]]
                        t  = kp * (target_pos[i] - q) - kd * qd
                        t  = float(np.clip(t, -effort_limit, effort_limit))
                        data.ctrl[aid] = t
                        torques[i]     = t
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

            hip_z = data.xpos[hip_bid, 2]

            # ── Termination: IsaacLab env_cfg.py TerminationsCfg 기준 ────
            # bad_orientation: base 기울기 > 0.78 rad (45°)
            quat_wxyz = data.qpos[3:7].astype(np.float32)
            grav_body = projected_gravity_vec(quat_wxyz)   # (0,0,-1) in world → body
            tilt = float(np.arccos(np.clip(-grav_body[2], -1.0, 1.0)))  # 0 = upright
            if tilt > 0.78:
                print(f"[SIM2SIM] TILT at step {step} (tilt={np.degrees(tilt):.1f}°, threshold=44.7°)")
                return

            if args.diag > 0 and step % args.diag == 0:
                pitch = _pitch_deg(data.qpos[3:7].astype(np.float32))
                qpos_rel = obs_np[9:21]
                jnames_short = ["Lhr","Lhy","Lhp","Lk","Lap","Lar",
                                "Rhr","Rhy","Rhp","Rk","Rap","Rar"]
                print(f"\n--- step {step:4d}  pitch={pitch:+6.1f}°  hip_z={hip_z:.3f}m ---")
                print(f"  grav_body : {obs_np[6:9].round(3)}")
                print(f"  ang_vel   : {obs_np[3:6].round(3)}")
                print(f"  qpos_rel  : {dict(zip(jnames_short, qpos_rel.round(3)))}")
                print(f"  action    : {dict(zip(jnames_short, action_np.round(3)))}")
                print(f"  torque    : {dict(zip(jnames_short, torques.round(2)))}")
                sat = np.abs(torques) >= effort_limit - 0.01
                if sat.any():
                    print(f"  SATURATED : {[jnames_short[i] for i in range(12) if sat[i]]}")

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
    parser = argparse.ArgumentParser(description="Hylion v6 sim-to-sim (IsaacLab → MuJoCo)")
    # 경로
    parser.add_argument("--ckpt",         type=str,   default=DEFAULT_CKPT)
    parser.add_argument("--urdf",         type=str,   default=DEFAULT_URDF)
    parser.add_argument("--mjcf",         type=str,   default=DEFAULT_MJCF,
                        help="MJCF .xml 경로 (지정시 URDF 대신 사용)")
    # 속도 명령
    parser.add_argument("--vx",           type=float, default=0.0)
    parser.add_argument("--vy",           type=float, default=0.0)
    parser.add_argument("--wz",           type=float, default=0.0)
    parser.add_argument("--duration",     type=float, default=10.0)
    # 물리 파라미터 (training defaults: kp=20, kd=2, effort_limit=6)
    parser.add_argument("--kp",           type=float, default=DEFAULT_KP,
                        help="PD stiffness (default 20)")
    parser.add_argument("--kd",           type=float, default=DEFAULT_KD,
                        help="PD damping (default 2)")
    parser.add_argument("--effort-limit", type=float, default=DEFAULT_EFFORT_LIMIT,
                        dest="effort_limit",
                        help="Torque limit Nm (default 6; try 20 to test hypothesis B)")
    parser.add_argument("--armature",     action="store_true",
                        help="Apply training armature: hip/knee=0.007, ankle=0.002")
    # 모드
    parser.add_argument("--zero-action",  action="store_true", dest="zero_action",
                        help="Disable policy; pure PD at default pose (baseline)")
    parser.add_argument("--diag",         type=int,   default=0,
                        metavar="N",
                        help="Print obs/action/torque every N steps (0=off)")
    parser.add_argument("--device",       type=str,   default="cpu")
    parser.add_argument("--no-viewer",    action="store_true", dest="no_viewer",
                        help="Headless mode (no GUI, for SSH)")
    # 학습 환경 매칭 플래그
    parser.add_argument("--obs-noise",    action="store_true", dest="obs_noise",
                        help="IsaacLab 학습 시 적용된 UniformNoise를 obs에 주입 (ang_vel±0.3, grav±0.05, qpos±0.05, qvel±2.0)")
    parser.add_argument("--friction",     type=float, default=0.8, dest="friction",
                        help="바닥 마찰 (default 0.8; 학습 환경 범위 0.4~1.2, 낮게: 0.4, 높게: 1.2)")
    args = parser.parse_args()

    if not args.zero_action and not os.path.isfile(args.ckpt):
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        sys.exit(1)
    if not os.path.isfile(args.urdf):
        print(f"[ERROR] URDF not found: {args.urdf}")
        sys.exit(1)

    run(args)
