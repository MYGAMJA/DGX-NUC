"""Sim2sim runner — BHL biped or Hylion MJCF, standalone (no gamepad).

Run from Berkeley-Humanoid-Lite-main/ directory:
    python3 run_sim.py [--vx 0.4] [--config ...] [--mjcf ...] [--no-viewer]

Observations (45-dim):
    [0:3]   command_velocity  (vx, vy, wz)
    [3:6]   base_ang_vel
    [6:9]   projected_gravity
    [9:21]  joint_pos_rel   (12 joints, default-subtracted)
    [21:33] joint_vel
    [33:45] prev_actions
"""

import argparse
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MJCF   = os.path.join(ROOT, "robot_assets", "mjcf", "bhl_biped_scene.xml")
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "policy_biped_25hz_a.yaml")

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

DEFAULT_JOINT_POS = np.array([
    0.0, 0.0, -0.2, 0.4, -0.3, 0.0,
    0.0, 0.0, -0.2, 0.4, -0.3, 0.0,
], dtype=np.float32)


def load_config(path):
    from omegaconf import OmegaConf
    with open(path) as f:
        return OmegaConf.load(f)


def load_onnx_policy(onnx_path):
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    input_shape = session.get_inputs()[0].shape
    try:
        session.run(None, {"obs": np.zeros(input_shape, dtype=np.float32)})
        key = "obs"
    except Exception:
        key = "onnx::Gemm_0"
    print(f"[SIM] ONNX input key: '{key}', shape: {input_shape}")
    return session, key


def projected_gravity(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], dtype=np.float32)


def run(args):
    import mujoco
    import mujoco.viewer

    cfg = load_config(args.config)
    kp = np.array(cfg.joint_kp, dtype=np.float32)
    kd = np.array(cfg.joint_kd, dtype=np.float32)
    effort = np.array(cfg.effort_limits, dtype=np.float32)
    action_scale = float(cfg.action_scale)
    physics_dt = float(cfg.physics_dt)
    policy_dt = float(cfg.policy_dt)
    n_substeps = max(1, round(policy_dt / physics_dt))
    policy_hz = 1.0 / policy_dt

    print(f"[SIM] Policy Hz={policy_hz:.0f}, substeps={n_substeps}, physics_dt={physics_dt}")

    # Load MJCF
    mjcf_path = args.mjcf
    print(f"[SIM] Loading MJCF: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    model.opt.timestep = physics_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    print(f"[SIM] nq={model.nq}, nv={model.nv}, nu={model.nu}, nsensordata={model.nsensordata}")

    # Sensor addresses
    def sensor_adr(name, required=True):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            if required:
                raise ValueError(f"Sensor '{name}' not found in MJCF")
            return None
        return model.sensor_adr[sid]

    quat_adr = sensor_adr("imu_quat")
    gyro_adr = sensor_adr("imu_gyro")
    pos_adr  = sensor_adr("frame_pos", required=False)  # optional: IMU world position

    # Fall detection: frame_pos sensor > imu site > base body xpos (in priority)
    imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
    if imu_site_id < 0:
        imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_bid < 0:
        base_bid = 1

    if pos_adr is not None:
        height_src = "frame_pos sensor"
    elif imu_site_id >= 0:
        height_src = f"imu site (id={imu_site_id})"
    else:
        height_src = "base xpos"
    print(f"[SIM] imu_quat@{quat_adr}, imu_gyro@{gyro_adr}, height via: {height_src}")

    # Joint indices
    qpos_ids = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in LEG_JOINTS]
    qvel_ids = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in LEG_JOINTS]
    act_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in LEG_JOINTS]

    # Initial pose: set default joint angles, then auto-correct base height
    mujoco.mj_resetData(model, data)
    data.qpos[3] = 1.0  # quat w
    data.qpos[2] = 2.0  # temp height for FK
    for i, qi in enumerate(qpos_ids):
        data.qpos[qi] = DEFAULT_JOINT_POS[i]
    mujoco.mj_forward(model, data)

    active_geoms = [i for i in range(model.ngeom)
                    if model.geom_contype[i] > 0
                    and model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE]
    foot_id = min(active_geoms, key=lambda i: data.geom_xpos[i, 2])
    foot_z = data.geom_xpos[foot_id, 2]
    size = model.geom_size[foot_id]
    R = data.geom_xmat[foot_id].reshape(3, 3)
    gtype = model.geom_type[foot_id]
    local_half = np.array([size[0], size[0], size[1]]) if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER else size[:3].copy()
    world_half_z = float(np.abs(R[2, :]) @ local_half)
    data.qpos[2] = 2.0 - (foot_z - world_half_z) + 0.003
    mujoco.mj_forward(model, data)
    print(f"[SIM] base_z={data.qpos[2]:.3f} m")

    # Load policy — resolve relative to config file, then ROOT as fallback
    ckpt_raw = cfg.policy_checkpoint_path
    config_dir = os.path.dirname(os.path.abspath(args.config))
    ckpt = ckpt_raw if os.path.isabs(ckpt_raw) else os.path.join(config_dir, "..", ckpt_raw)
    if not os.path.isfile(ckpt):
        ckpt = os.path.join(ROOT, ckpt_raw)
    print(f"[SIM] Loading policy: {ckpt}")
    session, onnx_key = load_onnx_policy(ckpt)

    cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    print(f"[SIM] Command: vx={args.vx} vy={args.vy} wz={args.wz}")

    prev_actions = np.zeros(12, dtype=np.float32)
    total_steps = int(args.duration * policy_hz)
    survived = 0

    def get_obs():
        quat = data.sensordata[quat_adr:quat_adr + 4].astype(np.float32)
        ang_vel = data.sensordata[gyro_adr:gyro_adr + 3].astype(np.float32)
        proj_grav = projected_gravity(quat)
        j_pos = np.array([data.qpos[qi] for qi in qpos_ids], np.float32) - DEFAULT_JOINT_POS
        j_vel = np.array([data.qvel[vi] for vi in qvel_ids], np.float32)
        return np.concatenate([cmd, ang_vel, proj_grav, j_pos, j_vel, prev_actions])

    def run_loop(viewer=None):
        nonlocal survived
        for step in range(total_steps):
            t_step_start = time.perf_counter()

            obs = get_obs().reshape(1, -1).astype(np.float32)
            actions = session.run(None, {onnx_key: obs})[0].flatten()
            actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
            prev_actions[:] = actions

            target_pos = DEFAULT_JOINT_POS + actions * action_scale
            for _ in range(n_substeps):
                j_pos = np.array([data.qpos[qpos_ids[i]] for i in range(12)], np.float32)
                j_vel = np.array([data.qvel[qvel_ids[i]] for i in range(12)], np.float32)
                torques = np.clip(kp * (target_pos - j_pos) + kd * (-j_vel), -effort, effort)
                for i, aid in enumerate(act_ids):
                    data.ctrl[aid] = float(torques[i])
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                # Real-time throttle: sleep remaining time of this control step
                elapsed = time.perf_counter() - t_step_start
                sleep_t = policy_dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            # Height for fall detection
            if pos_adr is not None:
                height = float(data.sensordata[pos_adr + 2])
            elif imu_site_id >= 0:
                height = float(data.site_xpos[imu_site_id, 2])
            else:
                height = float(data.xpos[base_bid, 2])

            if args.diag > 0 and step % args.diag == 0:
                quat = data.sensordata[quat_adr:quat_adr + 4]
                g = projected_gravity(quat)
                pitch = float(np.degrees(np.arcsin(np.clip(-g[0], -1, 1))))
                print(f"  step={step:4d}  z={height:.3f}  pitch={pitch:+.1f}°  "
                      f"act=[{actions.min():.2f},{actions.max():.2f}]")

            if height < args.fall_threshold:
                print(f"[SIM] FALL at step {step} (z={height:.3f})")
                return
            survived = step + 1

    print(f"[SIM] Running {total_steps} steps ({args.duration:.0f}s) ...")
    if args.no_viewer:
        run_loop()
    else:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                run_loop(viewer)
        except Exception as e:
            print(f"[SIM] Viewer error: {e}\nRetrying headless ...")
            run_loop()

    print(f"[SIM] Done: survived {survived}/{total_steps} steps "
          f"({survived / policy_hz:.1f} / {args.duration:.0f} s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default=DEFAULT_CONFIG)
    parser.add_argument("--mjcf",            default=DEFAULT_MJCF)
    parser.add_argument("--vx",              type=float, default=0.4)
    parser.add_argument("--vy",              type=float, default=0.0)
    parser.add_argument("--wz",              type=float, default=0.0)
    parser.add_argument("--duration",        type=float, default=15.0)
    parser.add_argument("--diag",            type=int,   default=25)
    parser.add_argument("--fall-threshold",  type=float, default=0.25, dest="fall_threshold")
    parser.add_argument("--no-viewer",       action="store_true", dest="no_viewer")
    args = parser.parse_args()

    if not os.path.isfile(args.mjcf):
        print(f"[ERROR] MJCF not found: {args.mjcf}")
        sys.exit(1)

    run(args)
