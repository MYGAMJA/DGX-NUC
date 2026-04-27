"""Evaluate Hylion v6 policy survival under fixed commands.

This script measures how often the policy falls within a fixed horizon.
It supports a deterministic demo-style reset mode and a randomized reset mode.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl")

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate Hylion v6 policy survival.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Velocity-Hylion-BG-v0")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-15_13-36-12/model_11998.pt",
)
parser.add_argument("--lin_vel_x", type=float, default=0.2)
parser.add_argument("--lin_vel_y", type=float, default=0.0)
parser.add_argument("--ang_vel_z", type=float, default=0.0)
parser.add_argument("--horizon_sec", type=float, default=10.0)
parser.add_argument("--warmup_sec", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--base_mass_add_kg", type=float, default=0.0)
parser.add_argument("--leg_gain_scale", type=float, default=1.2)
parser.add_argument("--feet_air_threshold", type=float, default=0.2)
parser.add_argument(
    "--randomized_resets",
    action="store_true",
    default=False,
    help="Keep training-like reset randomization instead of deterministic demo resets.",
)
parser.add_argument(
    "--keep_startup_randomization",
    action="store_true",
    default=False,
    help="Keep startup randomization terms instead of freezing them for demo evaluation.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import berkeley_humanoid_lite.tasks  # noqa: F401

sys.path.insert(0, PROJECT_ROOT)
from hylion import agents
from hylion.env_cfg_BG import HylionEnvCfg_BG

gym.register(
    id="Velocity-Hylion-BG-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HylionEnvCfg_BG,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HylionPPORunnerCfg,
    },
)


def _freeze_startup_randomization(env_cfg):
    if not hasattr(env_cfg, "events"):
        return
    for term in ("physics_material", "add_all_joint_default_pos", "base_external_force_torque"):
        if hasattr(env_cfg.events, term):
            setattr(env_cfg.events, term, None)


def _freeze_reset_randomization(env_cfg):
    if not hasattr(env_cfg, "events"):
        return
    if hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
        params = getattr(env_cfg.events.reset_base, "params", None)
        if isinstance(params, dict):
            params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
            params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    if hasattr(env_cfg.events, "reset_robot_joints") and env_cfg.events.reset_robot_joints is not None:
        params = getattr(env_cfg.events.reset_robot_joints, "params", None)
        if isinstance(params, dict):
            params["position_range"] = (1.0, 1.0)
            params["velocity_range"] = (0.0, 0.0)


def _configure_command(env_cfg, lin_vel_x, lin_vel_y, ang_vel_z):
    if not (hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity")):
        return
    cmd = env_cfg.commands.base_velocity
    if hasattr(cmd, "heading_command"):
        cmd.heading_command = False
    if hasattr(cmd, "rel_standing_envs"):
        cmd.rel_standing_envs = 0.0
    ranges = getattr(cmd, "ranges", None)
    if ranges is not None:
        if hasattr(ranges, "lin_vel_x"):
            ranges.lin_vel_x = (lin_vel_x, lin_vel_x)
        if hasattr(ranges, "lin_vel_y"):
            ranges.lin_vel_y = (lin_vel_y, lin_vel_y)
        if hasattr(ranges, "ang_vel_z"):
            ranges.ang_vel_z = (ang_vel_z, ang_vel_z)
        if hasattr(ranges, "heading"):
            ranges.heading = (0.0, 0.0)


def _handle_cfg_for_rsl_rl_v5(agent_cfg):
    cfg_dict = agent_cfg.to_dict()
    policy = cfg_dict.pop("policy", None)
    if policy is not None:
        cfg_dict["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("actor_hidden_dims", [256, 128, 128]),
            "activation": policy.get("activation", "elu"),
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": policy.get("init_noise_std", 0.5),
            },
        }
        cfg_dict["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("critic_hidden_dims", [256, 128, 128]),
            "activation": policy.get("activation", "elu"),
            "obs_normalization": False,
        }
    obs_groups = cfg_dict.get("obs_groups") or {}
    obs_groups.setdefault("policy", ["policy"])
    obs_groups.setdefault("actor", ["policy"])
    obs_groups.setdefault("critic", ["critic"])
    cfg_dict["obs_groups"] = obs_groups
    return cfg_dict


def _set_command(base_env, lin_vel_x: float, lin_vel_y: float, ang_vel_z: float):
    cmd_manager = base_env.command_manager
    term = cmd_manager._terms.get("base_velocity")
    if term is None:
        term = next(iter(cmd_manager._terms.values()), None)
    if term is not None and hasattr(term, "_command"):
        term._command[:, 0] = lin_vel_x
        term._command[:, 1] = lin_vel_y
        term._command[:, 2] = ang_vel_z


def main():
    os.environ["HYLION_BASE_MASS_ADD_KG"] = str(args_cli.base_mass_add_kg)
    os.environ["HYLION_LEG_GAIN_SCALE"] = str(args_cli.leg_gain_scale)
    os.environ["HYLION_FEET_AIR_THRESHOLD"] = str(args_cli.feet_air_threshold)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.seed = args_cli.seed
    if not args_cli.keep_startup_randomization:
        _freeze_startup_randomization(env_cfg)
    if not args_cli.randomized_resets:
        _freeze_reset_randomization(env_cfg)
    # Command range을 목표 속도로 고정 (0으로 잠그면 서있기 명령만 테스트됨)
    _configure_command(env_cfg, args_cli.lin_vel_x, args_cli.lin_vel_y, args_cli.ang_vel_z)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    rsl_rl_version = metadata.version("rsl-rl-lib")
    major = int(rsl_rl_version.split(".")[0])
    agent_cfg_dict = _handle_cfg_for_rsl_rl_v5(agent_cfg) if major >= 4 else agent_cfg.to_dict()

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.ckpt_path, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    control_hz = 25
    warmup_steps = int(args_cli.warmup_sec * control_hz)
    eval_steps = int(args_cli.horizon_sec * control_hz)
    total_steps = warmup_steps + eval_steps

    obs = env.get_observations()
    base_env = env.unwrapped
    num_envs = env.num_envs
    device = env.device

    failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    timeout_only = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_fail_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    for step in range(total_steps):
        if step < warmup_steps:
            vx = 0.5 * args_cli.lin_vel_x
            vy = 0.5 * args_cli.lin_vel_y
            wz = 0.5 * args_cli.ang_vel_z
        else:
            vx = args_cli.lin_vel_x
            vy = args_cli.lin_vel_y
            wz = args_cli.ang_vel_z
        _set_command(base_env, vx, vy, wz)

        with torch.inference_mode():
            actions = policy(obs)
            # env action_scale=0.25 이므로 raw 출력을 ±1.0으로 자르면 안 됨
            # NaN/Inf만 방어하고 스케일 클리핑은 제거
            actions = torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)
            obs, _, dones, extras = env.step(actions)

        time_outs = extras.get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(dones, dtype=torch.bool)
        else:
            time_outs = time_outs.to(torch.bool)
        done_mask = dones.to(torch.bool)
        fail_now = done_mask & ~time_outs
        timeout_now = done_mask & time_outs

        new_fail = fail_now & ~failed
        first_fail_step[new_fail] = step
        failed |= fail_now
        timeout_only |= timeout_now

    success_mask = ~failed
    success_count = int(success_mask.sum().item())
    fail_count = int(failed.sum().item())
    timeout_count = int(timeout_only.sum().item())
    success_rate = success_count / num_envs
    if fail_count > 0:
        fail_steps = first_fail_step[failed].float()
        fail_mean = float(fail_steps.mean().item())
        fail_min = int(fail_steps.min().item())
        fail_max = int(fail_steps.max().item())
    else:
        fail_mean = -1.0
        fail_min = -1
        fail_max = -1

    mode = "randomized_resets" if args_cli.randomized_resets else "demo_resets"
    startup = "startup_randomized" if args_cli.keep_startup_randomization else "startup_frozen"
    import sys
    print("[EVAL] mode=", mode, flush=True)
    print("[EVAL] startup=", startup, flush=True)
    print(f"[EVAL] command vx={args_cli.lin_vel_x} vy={args_cli.lin_vel_y} wz={args_cli.ang_vel_z}", flush=True)
    print(f"[EVAL] horizon_sec={args_cli.horizon_sec} warmup_sec={args_cli.warmup_sec} num_envs={num_envs}", flush=True)
    print(f"[EVAL] success={success_count}/{num_envs} success_rate={success_rate:.3f}", flush=True)
    print(f"[EVAL] fail_count={fail_count} timeout_count={timeout_count}", flush=True)
    if fail_count > 0:
        print(f"[EVAL] first_fail_step min={fail_min} mean={fail_mean:.1f} max={fail_max}", flush=True)
    sys.stdout.flush()

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()