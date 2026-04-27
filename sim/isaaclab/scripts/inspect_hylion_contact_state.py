"""Inspect Hylion v6 foot contact state without running PPO.

This launches the PhysX Hylion environment, steps it forward with zero actions,
and prints root height, ankle heights, net contact forces, and air/contact times.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect Hylion foot contact state.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--task", type=str, default="Velocity-Hylion-BG-v0")
parser.add_argument("--hylion_usd_path", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.hylion_usd_path:
    os.environ["HYLION_USD_PATH"] = args_cli.hylion_usd_path

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import warp as wp

from isaaclab_tasks.utils.hydra import hydra_task_config

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


def _to_torch(x):
    return x if isinstance(x, torch.Tensor) else wp.to_torch(x)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 123

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

    foot_ids, foot_names = robot.find_bodies(["leg_left_ankle_roll", "leg_right_ankle_roll"], preserve_order=True)
    sensor_ids, sensor_names = contact_sensor.find_sensors(["leg_left_ankle_roll", "leg_right_ankle_roll"], preserve_order=True)

    print("[INSPECT] robot.body_names=", robot.body_names)
    print("[INSPECT] contact_sensor.body_names=", contact_sensor.body_names)
    print("[INSPECT] foot_ids=", foot_ids, "foot_names=", foot_names)
    print("[INSPECT] sensor_ids=", sensor_ids, "sensor_names=", sensor_names)

    action_dim = env.unwrapped.action_manager.total_action_dim
    zero_action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)

    for step in range(args_cli.steps):
        env.step(zero_action)

        root_pos = _to_torch(robot.data.root_pos_w)
        body_pos = _to_torch(robot.data.body_pos_w)
        net_force_hist = _to_torch(contact_sensor.data.net_forces_w_history)
        air_time = _to_torch(contact_sensor.data.current_air_time)
        contact_time = _to_torch(contact_sensor.data.current_contact_time)

        env0_root_z = float(root_pos[0, 2].item())
        env0_foot_z = body_pos[0, foot_ids, 2].tolist()
        env0_force = net_force_hist[0, 0, sensor_ids, :].norm(dim=-1).tolist()
        env0_air = air_time[0, sensor_ids].tolist()
        env0_contact = contact_time[0, sensor_ids].tolist()

        if step < 20 or step % 10 == 0 or step == args_cli.steps - 1:
            print(
                f"[STEP {step:03d}] root_z={env0_root_z:.4f} "
                f"foot_z={[round(v, 4) for v in env0_foot_z]} "
                f"force={[round(v, 4) for v in env0_force]} "
                f"air={[round(v, 4) for v in env0_air]} "
                f"contact={[round(v, 4) for v in env0_contact]}"
            )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()